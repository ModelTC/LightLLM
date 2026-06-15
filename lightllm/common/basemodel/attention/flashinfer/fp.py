import dataclasses
import torch
from ..base_att import BaseAttBackend, BasePrefillAttState, BaseDecodeAttState, AttControl
from lightllm.utils.dist_utils import get_dp_world_size, get_current_device_id
from ...triton_kernel.repack_kv_index import repack_kv_index
from .env_utils import set_flashinfer_envs


def _fast_plan_tensor_core_decode(
    decode_wrapper,
    indptr,
    indices,
    last_page_len,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    page_size,
    indptr_host,
    kv_lens_arr_host,
    max_kv_len,
):
    batch_size = len(last_page_len)
    if batch_size != decode_wrapper._fixed_batch_size:
        raise ValueError(
            "The batch size should be fixed in cudagraph mode, the runtime batch size {} "
            "mismatches the batch size set during initialization {}".format(
                batch_size, decode_wrapper._fixed_batch_size
            )
        )
    if len(indices) > len(decode_wrapper._paged_kv_indices_buf):
        raise ValueError("The size of indices should be less than or equal to the allocated buffer")

    qo_indptr_host = getattr(decode_wrapper, "_lightllm_qo_indptr_host", None)
    if qo_indptr_host is None or len(qo_indptr_host) != batch_size + 1:
        from flashinfer.decode import _get_range_buf

        qo_indptr_host = _get_range_buf(batch_size + 1, "cpu")
        decode_wrapper._lightllm_qo_indptr_host = qo_indptr_host

    if indptr_host is None:
        indptr_host = indptr.to("cpu")
    if kv_lens_arr_host is None:
        from flashinfer.decode import get_seq_lens

        last_page_len_host = last_page_len.to("cpu")
        kv_lens_arr_host = get_seq_lens(indptr_host, last_page_len_host, page_size)
    if max_kv_len is None:
        max_kv_len = max(kv_lens_arr_host).item()

    decode_wrapper._batch_size = batch_size
    decode_wrapper._num_qo_heads = num_qo_heads
    decode_wrapper._num_kv_heads = num_kv_heads
    decode_wrapper._block_tables = None
    decode_wrapper._max_kv_len = max_kv_len

    args = [
        decode_wrapper._float_workspace_buffer,
        decode_wrapper._int_workspace_buffer,
        decode_wrapper._pin_memory_int_workspace_buffer,
        qo_indptr_host,
        indptr_host,
        kv_lens_arr_host,
        batch_size,
        batch_size,
        num_qo_heads,
        num_kv_heads,
        page_size,
        decode_wrapper.is_cuda_graph_enabled,
        head_dim,
        head_dim,
        False,
        -1,
    ]
    if decode_wrapper._backend == "fa2":
        args.extend([-1, False, 0])
    decode_wrapper._plan_info = decode_wrapper._cached_module.plan(*args)
    decode_wrapper._pos_encoding_mode = "NONE"
    decode_wrapper._window_left = -1
    decode_wrapper._logits_soft_cap = 0.0
    decode_wrapper._sm_scale = None
    decode_wrapper._rope_scale = None
    decode_wrapper._rope_theta = None


class FlashInferAttBackend(BaseAttBackend):
    def __init__(self, model):
        set_flashinfer_envs()
        super().__init__(model=model)
        tp_world_size = get_dp_world_size()
        self.tp_q_head_num = model.config["num_attention_heads"] // tp_world_size
        self.tp_kv_head_num = max(model.config["num_key_value_heads"] // tp_world_size, 1)
        head_dim = model.config["hidden_size"] // model.config["num_attention_heads"]
        self.head_dim = model.config.get("head_dim", head_dim)
        self.workspace_buffer = torch.empty(512 * 1024 * 1024, dtype=torch.int8, device=get_current_device_id())
        self.max_seq_length = model.max_seq_length
        self.kv_indices_buffer = [
            torch.empty(
                model.graph_max_batch_size * self.max_seq_length, dtype=torch.int32, device=get_current_device_id()
            ),
            torch.empty(
                model.graph_max_batch_size * self.max_seq_length, dtype=torch.int32, device=get_current_device_id()
            ),
        ]
        self.kv_starts_host_buffer = [
            torch.empty((model.graph_max_batch_size + 1,), dtype=torch.int32, device="cpu"),
            torch.empty((model.graph_max_batch_size + 1,), dtype=torch.int32, device="cpu"),
        ]
        self.q_data_type = model.data_type
        self.kv_data_type = model.data_type

    def create_att_prefill_state(self, infer_state) -> "FlashInferPrefillAttState":
        return FlashInferPrefillAttState(backend=self, infer_state=infer_state)

    def create_att_decode_state(self, infer_state) -> "FlashInferDecodeAttState":
        return FlashInferDecodeAttState(backend=self, infer_state=infer_state)


@dataclasses.dataclass
class FlashInferPrefillAttState(BasePrefillAttState):
    prefill_wrapper: object = None

    def init_state(self):
        self.backend: FlashInferAttBackend = self.backend

        import flashinfer

        batch_size = self.infer_state.batch_size
        device = self.infer_state.input_ids.device

        q_starts = self.infer_state.b1_cu_q_seq_len.int()
        kv_starts = self.infer_state.b1_cu_kv_seq_len.int()
        kv_last_page_len = torch.full((batch_size,), 1, dtype=torch.int32, device=device)
        kv_indices = torch.empty(
            batch_size * self.backend.max_seq_length,
            dtype=torch.int32,
            device=device,
        )
        repack_kv_index(
            self.infer_state.req_manager.req_to_token_indexs,
            self.infer_state.b_req_idx,
            self.infer_state.b_seq_len,
            kv_starts[:-1],
            self.infer_state.max_kv_seq_len,
            kv_indices,
        )
        self.prefill_wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
            self.backend.workspace_buffer,
            qo_indptr_buf=q_starts,
            paged_kv_indptr_buf=kv_starts,
            paged_kv_indices_buf=kv_indices,
            paged_kv_last_page_len_buf=kv_last_page_len,
        )
        self.prefill_wrapper.plan(
            q_starts,
            kv_starts,
            kv_indices,
            kv_last_page_len,
            self.backend.tp_q_head_num,
            self.backend.tp_kv_head_num,
            self.backend.head_dim,
            1,
            causal=True,
            pos_encoding_mode="NONE",
            logits_soft_cap=0.0,
            q_data_type=self.backend.q_data_type,
            kv_data_type=self.backend.kv_data_type,
        )

    def prefill_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
    ) -> torch.Tensor:
        assert (
            att_control.use_alibi is False
            and att_control.use_sliding_window is False
            and att_control.use_att_sink is False
        )
        return self._nomarl_prefill_att(
            q=q,
            k=k,
            v=v,
            alloc_func=alloc_func,
        )

    def _nomarl_prefill_att(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, alloc_func=torch.empty
    ) -> torch.Tensor:
        self.backend: FlashInferAttBackend = self.backend  # for typing
        o_tensor = alloc_func(q.shape, q.dtype, device="cuda")
        self.prefill_wrapper.run(
            q,
            (k.unsqueeze(1), v.unsqueeze(1)),
            out=o_tensor,
        )
        return o_tensor


@dataclasses.dataclass
class FlashInferDecodeAttState(BaseDecodeAttState):
    kv_last_page_len_buffer: torch.Tensor = None
    kv_indices: torch.Tensor = None
    kv_starts: torch.Tensor = None
    kv_starts_host: torch.Tensor = None
    kv_seq_lens_host: torch.Tensor = None
    decode_wrapper: object = None

    def init_state(self):
        self.backend: FlashInferAttBackend = self.backend
        device = self.infer_state.input_ids.device
        model = self.backend.model
        self.kv_last_page_len_buffer = torch.full((self.infer_state.batch_size,), 1, dtype=torch.int32, device=device)
        if (
            self.infer_state.batch_size <= model.graph_max_batch_size
            and self.infer_state.max_kv_seq_len <= model.graph_max_len_in_batch
        ):
            self.kv_indices = self.backend.kv_indices_buffer[self.infer_state.microbatch_index][
                : self.infer_state.batch_size * self.backend.max_seq_length
            ]
        else:
            self.kv_indices = torch.empty(
                self.infer_state.batch_size * self.backend.max_seq_length,
                dtype=torch.int32,
                device=device,
            )

        repack_kv_index(
            self.infer_state.req_manager.req_to_token_indexs,
            self.infer_state.b_req_idx,
            self.infer_state.b_seq_len,
            self.infer_state.b_kv_start_loc,
            self.infer_state.max_kv_seq_len,
            self.kv_indices,
            zero_output=False,
        )
        self.kv_starts = self.infer_state.b1_cu_kv_seq_len.int()
        if self.infer_state.b_seq_len_cpu is not None:
            self.kv_seq_lens_host = self.infer_state.b_seq_len_cpu
            self.kv_starts_host = self.backend.kv_starts_host_buffer[self.infer_state.microbatch_index][
                : self.infer_state.batch_size + 1
            ]
            self.kv_starts_host[0] = 0
            torch.cumsum(self.infer_state.b_seq_len_cpu, dim=0, out=self.kv_starts_host[1:])
        if self.infer_state.skip_decode_att_wrapper_init:
            return

        import flashinfer

        assert self.decode_wrapper is None
        self.decode_wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
            self.backend.workspace_buffer,
            "NHD",
            use_cuda_graph=True,
            use_tensor_cores=True,
            paged_kv_indptr_buffer=self.kv_starts,
            paged_kv_indices_buffer=self.kv_indices,
            paged_kv_last_page_len_buffer=self.kv_last_page_len_buffer,
        )
        self.decode_wrapper.plan(
            self.kv_starts,
            self.kv_indices,
            self.kv_last_page_len_buffer,
            self.backend.tp_q_head_num,
            self.backend.tp_kv_head_num,
            self.backend.head_dim,
            1,
            q_data_type=self.backend.q_data_type,
            kv_data_type=self.backend.kv_data_type,
            non_blocking=True,
        )
        return

    def copy_for_decode_cuda_graph(self, new_state: "FlashInferDecodeAttState"):
        super().copy_for_decode_cuda_graph(new_state)
        _fast_plan_tensor_core_decode(
            self.decode_wrapper,
            new_state.kv_starts,
            new_state.kv_indices,
            new_state.kv_last_page_len_buffer,
            new_state.backend.tp_q_head_num,
            new_state.backend.tp_kv_head_num,
            new_state.backend.head_dim,
            1,
            new_state.kv_starts_host,
            new_state.kv_seq_lens_host,
            new_state.infer_state.max_kv_seq_len,
        )

    def decode_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
    ):
        assert (
            att_control.use_alibi is False
            and att_control.use_sliding_window is False
            and att_control.use_att_sink is False
        )
        return self._normal_decode_att(
            q=q,
            k=k,
            v=v,
            alloc_func=alloc_func,
        )

    def _normal_decode_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        alloc_func=torch.empty,
    ):
        o_tensor = alloc_func(q.shape, q.dtype)
        self.decode_wrapper.run(
            q,
            (k.unsqueeze(1), v.unsqueeze(1)),
            out=o_tensor,
        )
        return o_tensor
