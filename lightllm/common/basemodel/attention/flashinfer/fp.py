import dataclasses
import torch
import triton
from ..base_att import BaseAttBackend, BasePrefillAttState, BaseDecodeAttState, AttControl
from lightllm.utils.dist_utils import get_dp_world_size, get_current_device_id
from ...triton_kernel.gen_prefill_params import gen_cumsum_pad0_tensor
from ...triton_kernel.repack_kv_index import repack_kv_index
from .env_utils import set_flashinfer_envs
from .utils import should_init_decode_wrapper


class FlashInferAttBackend(BaseAttBackend):
    workspace_buffer_key = "flashinfer_fp"
    workspace_buffer_size = 512 * 1024 * 1024

    def __init__(self, model):
        set_flashinfer_envs()
        super().__init__(model=model)
        self._init_infer_page_size()
        tp_world_size = get_dp_world_size()
        self.tp_q_head_num = model.config["num_attention_heads"] // tp_world_size
        self.tp_kv_head_num = max(model.config["num_key_value_heads"] // tp_world_size, 1)
        head_dim = model.config["hidden_size"] // model.config["num_attention_heads"]
        self.head_dim = model.config.get("head_dim", head_dim)
        self.max_seq_length = model.max_seq_length
        self.max_page_num = triton.cdiv(self.max_seq_length, self.infer_page_size)
        self.kv_indices_buffer = [
            torch.empty(
                model.graph_max_batch_size * self.max_page_num, dtype=torch.int32, device=get_current_device_id()
            ),
            torch.empty(
                model.graph_max_batch_size * self.max_page_num, dtype=torch.int32, device=get_current_device_id()
            ),
        ]
        self.q_data_type = model.data_type
        self.kv_data_type = model.data_type

    def _init_infer_page_size(self):
        self.infer_page_size = self.model.args.page_size
        assert self.model.args.page_size % self.infer_page_size == 0, (
            f"model page_size {self.model.args.page_size} "
            f"must be divisible by infer_page_size {self.infer_page_size}"
        )

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
        # TODO: 将页数、页数前缀和及末页有效 token 数的计算融合为一个 Triton 算子。
        # token 长度除以页大小并向上取整，末页不足一页也计为一页。
        b_page_len = (self.infer_state.b_seq_len + (self.backend.infer_page_size - 1)) // self.backend.infer_page_size
        kv_starts, _ = gen_cumsum_pad0_tensor(b_page_len, b_page_len)
        kv_last_page_len = self.infer_state.b_seq_len - (b_page_len - 1) * self.backend.infer_page_size
        kv_indices = torch.empty(
            batch_size * self.backend.max_page_num,
            dtype=torch.int32,
            device=device,
        )
        repack_kv_index(
            req_to_token_indexs=self.infer_state.req_manager.req_to_token_indexs,
            b_req_idx=self.infer_state.b_req_idx,
            b_token_len=self.infer_state.b_seq_len,
            b_page_start_loc=kv_starts[:-1],
            max_token_len=self.infer_state.max_kv_seq_len,
            out_page_indices=kv_indices,
            page_size=self.backend.infer_page_size,
        )
        self.prefill_wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
            self.backend.get_gpu_workspace_buffer(
                key_name=self.backend.workspace_buffer_key,
                workspace_size=self.backend.workspace_buffer_size,
            ),
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
            self.backend.infer_page_size,
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
            (
                k.view(-1, self.backend.infer_page_size, k.shape[1], k.shape[2]),
                v.view(-1, self.backend.infer_page_size, v.shape[1], v.shape[2]),
            ),
            out=o_tensor,
        )
        return o_tensor


@dataclasses.dataclass
class FlashInferDecodeAttState(BaseDecodeAttState):
    kv_last_page_len_buffer: torch.Tensor = None
    kv_indices: torch.Tensor = None
    kv_starts: torch.Tensor = None
    decode_wrapper: object = None

    def _should_init_decode_wrapper(self) -> bool:
        return should_init_decode_wrapper(self.backend.model, self.infer_state)

    def init_state(self):
        import flashinfer

        self.backend: FlashInferAttBackend = self.backend
        device = self.infer_state.input_ids.device
        model = self.backend.model
        # TODO: 将页数、页数前缀和及末页有效 token 数的计算融合为一个 Triton 算子。
        # token 长度除以页大小并向上取整，末页不足一页也计为一页。
        b_page_len = (self.infer_state.b_seq_len + (self.backend.infer_page_size - 1)) // self.backend.infer_page_size
        self.kv_last_page_len_buffer = self.infer_state.b_seq_len - (b_page_len - 1) * self.backend.infer_page_size
        if (
            self.infer_state.batch_size <= model.graph_max_batch_size
            and self.infer_state.max_kv_seq_len <= model.graph_max_len_in_batch
        ):
            self.kv_indices = self.backend.kv_indices_buffer[self.infer_state.microbatch_index][
                : self.infer_state.batch_size * self.backend.max_page_num
            ]
        else:
            self.kv_indices = torch.empty(
                self.infer_state.batch_size * self.backend.max_page_num,
                dtype=torch.int32,
                device=device,
            )

        self.kv_starts, _ = gen_cumsum_pad0_tensor(b_page_len, b_page_len)
        repack_kv_index(
            req_to_token_indexs=self.infer_state.req_manager.req_to_token_indexs,
            b_req_idx=self.infer_state.b_req_idx,
            b_token_len=self.infer_state.b_seq_len,
            b_page_start_loc=self.kv_starts[:-1],
            max_token_len=self.infer_state.max_kv_seq_len,
            out_page_indices=self.kv_indices,
            page_size=self.backend.infer_page_size,
        )
        if not self._should_init_decode_wrapper():
            # 处于 graph replay 回放阶段，不需要特殊初始化 decode wrapper。
            return

        assert self.decode_wrapper is None
        self.decode_wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
            self.backend.get_gpu_workspace_buffer(
                key_name=self.backend.workspace_buffer_key,
                workspace_size=self.backend.workspace_buffer_size,
            ),
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
            self.backend.infer_page_size,
            q_data_type=self.backend.q_data_type,
            kv_data_type=self.backend.kv_data_type,
            non_blocking=True,
        )
        return

    def copy_for_decode_cuda_graph(self, new_state: "FlashInferDecodeAttState"):
        super().copy_for_decode_cuda_graph(new_state)
        self._refresh_cuda_graph_decode_plan(new_state.infer_state.max_kv_seq_len)
        return

    def _refresh_cuda_graph_decode_plan(self, max_kv_len: int):
        from flashinfer.decode import fast_decode_plan

        uniform_kv_indptr_cpu = (
            torch.arange(
                self.infer_state.batch_size + 1,
                dtype=torch.int32,
                device="cpu",
            )
            * triton.cdiv(max_kv_len, self.backend.infer_page_size)
        )

        fast_decode_plan(
            self.decode_wrapper,
            indptr=self.kv_starts,
            indices=self.kv_indices,
            last_page_len=self.kv_last_page_len_buffer,
            num_qo_heads=self.backend.tp_q_head_num,
            num_kv_heads=self.backend.tp_kv_head_num,
            head_dim=self.backend.head_dim,
            page_size=self.backend.infer_page_size,
            q_data_type=self.backend.q_data_type,
            kv_data_type=self.backend.kv_data_type,
            non_blocking=True,
            global_override_indptr_cpu=uniform_kv_indptr_cpu,
        )
        return

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
            (
                k.view(-1, self.backend.infer_page_size, k.shape[1], k.shape[2]),
                v.view(-1, self.backend.infer_page_size, v.shape[1], v.shape[2]),
            ),
            out=o_tensor,
        )
        return o_tensor
