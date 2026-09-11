import torch

from lightllm.common.basemodel.triton_kernel.norm.qk_norm import qk_rmsnorm_forward
from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.models.qwen3_dflash.model import Qwen3DFlashModel
from lightllm.models.windowed_mtp.kv_store import WindowKVStore
from lightllm.models.windowed_mtp.layer_infer import WindowedAttentionMixin
from lightllm.utils.windowed_mtp import window_capacity, validate_windowed_mtp
from lightllm.common.basemodel.attention.base_att import BaseAttBackend, BaseDecodeAttState
from lightllm.utils.log_utils import init_logger
from lightllm.utils.torch_memory_saver_utils import MemoryTag


logger = init_logger(__name__)


class WindowedDecodeState(BaseDecodeAttState):
    def init_state(self):
        pass

    def decode_att(self, *args, **kwargs):
        raise RuntimeError("Windowed draft uses bounded FA3 attention directly")


class WindowedAttBackend(BaseAttBackend):
    def create_att_decode_state(self, infer_state):
        return WindowedDecodeState(backend=self, infer_state=infer_state)


class WindowedDraftMixin:
    uses_windowed_draft_kv = True

    def _verify_params(self):
        super()._verify_params()
        validate_windowed_mtp(self.args)

    def _init_custom(self):
        super()._init_custom()
        # _init_custom runs after the base model's KV allocation region.
        with self.torch_memory_saver.region(tag=MemoryTag.KV_CACHE):
            self.kv_store = WindowKVStore(
                self.args.running_max_req_size + 1,
                window_capacity(self.args),
                self.config["n_layer"],
                self.tp_k_head_num_,
                self.head_dim_,
                self.data_type,
                self._cos_cached.device,
                self.args.mtp_draft_window,
                self.args.mtp_draft_sinks,
            )
        for i, layer in enumerate(self.layers_infer):
            layer.windowed_layer_index = i
        store = self.kv_store
        pool_bytes = sum(t.numel() * t.element_size() for t in (store.kv, store.ends, store.counts))
        logger.info(
            f"{self.args.mtp_mode} windowed: full-context draft KV layers=0, "
            f"window KV layers={self.config['n_layer']}, "
            f"capacity={store.capacity}, KV pool bytes={pool_bytes}, target KV layers={len(self.mem_manager.kv_buffer)}"
        )

    @torch.no_grad()
    def commit_features(self, model_input, hidden, starts=None, accept_len=None):
        # The fused feature is transient; only its per-layer K/V is retained.
        projected = self.pre_post_weight.fc_weight_.mm(hidden, use_custom_tensor_mananger=False)
        projected = self.pre_post_weight.hidden_norm_weight_(projected, eps=self.pre_infer.eps_, alloc_func=torch.empty)
        if model_input.is_prefill:
            reqs = model_input.b_req_idx.long()
            starts = model_input.b_prefill_start_loc
            first = model_input.b_ready_cache_len
            lengths = model_input.b_seq_len - first
            max_new = model_input.max_q_seq_len
        else:
            reqs = model_input.b_req_idx.index_select(0, starts.long()).long()
            first = model_input.b_seq_len.index_select(0, starts.long()) - 1
            lengths = accept_len
            max_new = self.args.mtp_step + 1
        features, positions = self.kv_store.prepare(reqs, projected, starts, first, lengths, max_new)
        cos = self._cos_cached.index_select(0, positions.clamp_min(0))
        sin = self._sin_cached.index_select(0, positions.clamp_min(0))
        for i, (layer, weight) in enumerate(zip(self.layers_infer, self.trans_layers_weight)):
            kv = weight.kv_proj.mm(features, use_custom_tensor_mananger=False)
            qk_rmsnorm_forward(
                kv[:, : self.tp_k_head_num_ * self.head_dim_], weight.qk_norm_weight_.k_weight, layer.eps_
            )
            kv = kv.view(-1, 2 * self.tp_k_head_num_, self.head_dim_)
            rotary_emb_fwd(
                kv[:, : self.tp_k_head_num_], None, cos, sin, partial_rotary_factor=layer.partial_rotary_factor
            )
            self.kv_store.write(i, reqs, positions, kv)

    def _token_forward(self, infer_state):
        # These reads run inside CUDA Graph, using the current request IDs/state.
        reqs = infer_state.b_req_idx[:: self.block_size].long()
        infer_state.windowed_context = (
            self.kv_store,
            reqs,
            self.kv_store.counts.index_select(0, reqs),
            self.block_size,
        )
        try:
            return super()._token_forward(infer_state)
        finally:
            del infer_state.windowed_context

    def _init_att_backend(self):
        self.prefill_att_backend = self.decode_att_backend = WindowedAttBackend(model=self)

    def _decode(self, model_input):
        # Window KV is committed by the proposer. Draft block rows never own
        # target cache slots and must not overwrite its request-to-token table.
        origin_batch_size = model_input.batch_size
        assert origin_batch_size > 0 and origin_batch_size % self.block_size == 0
        infer_batch_size = origin_batch_size
        use_cuda_graph = self.graph is not None and self.graph.can_run(
            batch_size=infer_batch_size,
            max_len_in_batch=max(2, model_input.max_kv_seq_len),
        )
        need_capture = False
        if use_cuda_graph:
            infer_batch_size = self.graph.find_closest_graph_batch_size(batch_size=infer_batch_size)
            need_capture = self.graph.need_capture(infer_batch_size)
        model_input = self._create_padded_decode_model_input(model_input, infer_batch_size)
        infer_state = self._create_inferstate(model_input)
        infer_state.is_cuda_graph = need_capture
        infer_state.init_some_extra_state(self)
        infer_state.init_att_state()
        if use_cuda_graph:
            if need_capture:
                output = self.graph.capture_decode(self._token_forward, infer_state)
            else:
                output = self.graph.replay(infer_state)
        else:
            output = self._token_forward(infer_state)
        return self._create_unpad_decode_model_output(output, origin_batch_size=origin_batch_size)


def windowed_model_class(original):
    if not issubclass(original, Qwen3DFlashModel):
        raise ValueError("Windowed draft KV currently supports the Qwen3 parallel block backbone")
    layer_class = type(
        "Windowed" + original.transformer_layer_infer_class.__name__,
        (WindowedAttentionMixin, original.transformer_layer_infer_class),
        {},
    )
    return type(
        "Windowed" + original.__name__, (WindowedDraftMixin, original), {"transformer_layer_infer_class": layer_class}
    )
