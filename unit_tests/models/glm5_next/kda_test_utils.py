"""Build KDA layer inputs and attention the way GLM-5-Next layers do (see _kda_layer_inputs)."""
from typing import Any
from unit_tests.common.attention.backends.step_fixture import create_att_state

from types import SimpleNamespace

import torch

from unit_tests.common.attention.backends.step_fixture import make_infer_state
from lightllm.common.basemodel.infer_struct import InferStateInfo


def kda_params(
    conv_states: Any,
    ssm_states: Any,
    layer_weight: Any,
    mixed_qkv: Any,
    raw_gate: Any,
    raw_beta: Any,
    *,
    state: Any = None,
    logic_layer_index: Any = 0,
):
    return dict(
        mixed_qkv=mixed_qkv,
        raw_gate=raw_gate,
        raw_beta=raw_beta,
        conv_weight=layer_weight.get_merged_kda_conv_weight(),
        A_log=layer_weight.linear_A_log.weight,
        dt_bias=layer_weight.linear_dt_bias.weight,
        conv_states=conv_states,
        ssm_states=ssm_states,
        state=state,
        logic_layer_index=logic_layer_index,
    )


def prepare_prefill_infer(infer, device="cuda"):
    """Give a test stub the InferStateInfo defaults of the fields it does not set, as a prefill step."""
    for name, value in vars(InferStateInfo()).items():
        if not hasattr(infer, name):
            setattr(infer, name, value)
    infer.is_prefill = True
    infer.batch_size = infer.b_req_idx.shape[0]
    if infer.req_manager is None:
        infer.req_manager = SimpleNamespace(req_to_token_indexs=None)
    return infer


def prefill_batch(max_q_seq_len, device="cpu"):
    b_req_idx = torch.zeros(1, dtype=torch.int32, device=device)
    return make_infer_state(
        is_prefill=True,
        req_to_token_indexs=None,
        b_req_idx=b_req_idx,
        b_seq_len=b_req_idx,
        batch_size=1,
        max_kv_seq_len=max_q_seq_len,
        total_token_num=max_q_seq_len,
        b1_cu_q_seq_len=torch.tensor([0, max_q_seq_len], dtype=torch.int32, device=device),
        b_ready_cache_len=b_req_idx,
        max_q_seq_len=max_q_seq_len,
    )


def kda_att_config(
    *,
    heads: Any,
    dim: Any = 128,
    replay: Any = False,
    mtp_step: Any = 0,
    prefill_kernel: Any = "triton",
    ssm_dtype: Any = torch.float32,
    conv_kernel_size: Any = 4,
    lower_bound: Any = -5.0,
    dtype: Any = torch.bfloat16,
):
    """GLM-5 层线性注意力分支的配置。"""
    from lightllm.common.attention.backends.kda import KDAAttConfig

    return KDAAttConfig(
        num_heads=heads,
        head_dim=dim,
        conv_kernel_size=conv_kernel_size,
        lower_bound=lower_bound,
        dtype=dtype,
        state_mtp_step=mtp_step,
        replay=replay,
        ssm_dtype=ssm_dtype,
        prefill_kernel=prefill_kernel,
    )


class KDAAttention:
    """A KDA layer's attention outside a model: the impls and step builders its layer group uses.

    step_config describes the model's decode rows; each step adds the role's recurrent state index.
    """

    def __init__(self, att_config: Any, step_config: Any = None, *, prefill_impl: Any = None, role: Any = "main"):
        from lightllm.common.attention.backends.kda import triton as kda
        from unit_tests.common.attention.backends.step_fixture import AttentionStepConfig

        self.step_config = AttentionStepConfig() if step_config is None else step_config
        replay = att_config.replay
        prefill_cls = kda.ReplayKDAPrefillBackend if replay else kda.TritonKDAPrefillBackend
        decode_cls = kda.ReplayKDADecodeBackend if replay else kda.TritonKDADecodeBackend
        self.impls = {
            "prefill": prefill_impl if prefill_impl is not None else prefill_cls(att_config, role=role),
            "decode": decode_cls(att_config, role=role),
        }
        self.config = self.impls["prefill"].att_config
        self.flash_kda_fwd = self.impls["prefill"]._flash_kda_fwd

    def create_att_prefill_state(self, infer):
        return KDAStep(self, infer, "prefill")

    def create_att_decode_state(self, infer):
        return KDAStep(self, infer, "decode")


class KDAStep:
    """One step of a KDA layer with the backend state for that microbatch."""

    def __init__(self, attention, infer, phase):
        self.attention, self.infer, self.phase = attention, infer, phase
        self.att_state = None

    def init_state(self):
        from unit_tests.common.attention.backends.step_fixture import configure_attention_step

        infer, attention = self.infer, self.attention
        batch = configure_attention_step(infer_state=infer, step_config=attention.step_config)
        self.att_state = create_att_state(backend=attention.impls[self.phase], infer_state=batch)

    def run(self, params, alloc_func=torch.empty):
        """params 是 kda_params(...); 返回 [rows, num_heads, head_dim]."""
        from lightllm.common.attention.backends.kda import KDAAttInputs

        config = self.attention.config
        inputs = KDAAttInputs(
            mixed_qkv=params["mixed_qkv"],
            raw_gate=params["raw_gate"],
            raw_beta=params["raw_beta"],
            conv_weight=params["conv_weight"],
            A_log=params["A_log"],
            dt_bias=params["dt_bias"],
            conv_states=params["conv_states"],
            ssm_states=params["ssm_states"],
            replay_state=params["state"] if config.replay else None,
            logic_layer_index=params["logic_layer_index"],
        )
        x = inputs.mixed_qkv
        out = alloc_func((x.shape[0], config.num_heads, config.head_dim), dtype=x.dtype, device=x.device)
        backend = self.attention.impls[self.phase]
        attention = backend.prefill if self.phase == "prefill" else backend.decode
        return attention(att_state=self.att_state, layer_inputs=inputs, out=out, alloc_func=alloc_func)
