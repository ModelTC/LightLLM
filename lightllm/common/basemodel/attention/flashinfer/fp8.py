import dataclasses
import torch
from ..base_att import AttControl
from .fp import FlashInferAttBackend, FlashInferPrefillAttState, FlashInferDecodeAttState


class Fp8FlashInferAttBackend(FlashInferAttBackend):
    def __init__(self, model):
        super().__init__(model=model)
        self.kv_data_type = torch.float8_e4m3fn

    def create_att_prefill_state(self, infer_state) -> "Fp8FlashInferPrefillAttState":
        return Fp8FlashInferPrefillAttState(backend=self, infer_state=infer_state)

    def create_att_decode_state(self, infer_state) -> "Fp8FlashInferDecodeAttState":
        return Fp8FlashInferDecodeAttState(backend=self, infer_state=infer_state)


@dataclasses.dataclass
class Fp8FlashInferPrefillAttState(FlashInferPrefillAttState):
    offline_scales: torch.Tensor = None

    def init_state(self):
        super().init_state()
        self.offline_scales = self.infer_state.mem_manager.scales_list

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
        return self._fp8_prefill_att(
            q=q,
            k=k,
            v=v,
            alloc_func=alloc_func,
        )

    def _fp8_prefill_att(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, alloc_func=torch.empty
    ) -> torch.Tensor:
        o_tensor = alloc_func(q.shape, q.dtype, device="cuda")
        k = k.unsqueeze(1).view(torch.float8_e4m3fn)
        v = v.unsqueeze(1).view(torch.float8_e4m3fn)
        layer_index = self.backend._find_layer_index(k=k, v=v, att_state=self)
        offline_scales = self.offline_scales
        k_descale = offline_scales[layer_index][0] if offline_scales is not None else None
        v_descale = offline_scales[layer_index][1] if offline_scales is not None else None
        self.prefill_wrapper.run(
            q,
            (k, v),
            k_scale=k_descale,
            v_scale=v_descale,
            out=o_tensor,
        )
        return o_tensor


@dataclasses.dataclass
class Fp8FlashInferDecodeAttState(FlashInferDecodeAttState):
    offline_scales: torch.Tensor = None

    def init_state(self):
        super().init_state()
        self.offline_scales = self.infer_state.mem_manager.scales_list

    def copy_for_decode_cuda_graph(self, new_state):
        return super().copy_for_decode_cuda_graph(new_state)

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
        return self._fp8_decode_att(
            q=q,
            k=k,
            v=v,
            alloc_func=alloc_func,
        )

    def _fp8_decode_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        alloc_func=torch.empty,
    ):
        o_tensor = alloc_func(q.shape, q.dtype, device="cuda")

        k = k.unsqueeze(1).view(torch.float8_e4m3fn)
        v = v.unsqueeze(1).view(torch.float8_e4m3fn)
        offline_scales = self.offline_scales
        layer_index = self.backend._find_layer_index(k=k, v=v, att_state=self)

        k_descale = offline_scales[layer_index][0] if offline_scales is not None else None
        v_descale = offline_scales[layer_index][1] if offline_scales is not None else None
        self.decode_wrapper.run(
            q,
            (k, v),
            k_scale=k_descale,
            v_scale=v_descale,
            out=o_tensor,
        )
        return o_tensor
