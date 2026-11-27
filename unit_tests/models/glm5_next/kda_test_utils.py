"""Build KDA layer inputs and attention the way GLM-5-Next layers do (see _kda_layer_inputs)."""

from types import SimpleNamespace

import torch

from unit_tests.common.attention.backends.step_fixture import make_infer_state
from lightllm.common.basemodel.infer_struct import InferStateInfo


def kda_params(conv_states, ssm_states, layer_weight, mixed_qkv, raw_gate, raw_beta, *, state=None, logical_layer_id=0):
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
        logical_layer_id=logical_layer_id,
    )


def prepare_prefill_infer(infer, device="cuda"):
    """Give a test stub the InferStateInfo defaults of the fields it does not set, as a prefill step.

    Properties such as uses_cuda_graph exist only on InferStateInfo itself, so build stubs from InferStateInfo().
    """
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


def kda_layer_config(
    *,
    heads,
    dim=128,
    replay=False,
    mtp_step=0,
    prefill_kernel="triton",
    ssm_dtype=torch.float32,
    conv_kernel_size=4,
    lower_bound=-5.0,
    dtype=torch.bfloat16,
):
    """The config for the linear attention branch of a GLM-5 layer."""
    from lightllm.common.attention.backends.kda import KDALayerConfig

    return KDALayerConfig(
        num_heads=heads,
        head_dim=dim,
        conv_kernel_size=conv_kernel_size,
        gate_lower_bound=lower_bound,
        dtype=dtype,
        state_draft_capacity=mtp_step,
        use_ssm_replay=replay,
        ssm_dtype=ssm_dtype,
        prefill_kernel=prefill_kernel,
    )


class KDAAttention:
    """A KDA layer's attention outside a model: the set-up backends its layer group uses."""

    def __init__(self, layer_config):
        from lightllm.common.attention.backends.kda import recurrent as kda

        self.layer_config = layer_config
        replay = layer_config.use_ssm_replay
        backend_classes = {
            "prefill": kda.ReplayKDAPrefillBackend if replay else kda.KDAPrefillBackend,
            "decode": kda.ReplayKDADecodeBackend if replay else kda.TritonKDADecodeBackend,
        }
        self.backends = {phase: backend_cls(layer_config) for phase, backend_cls in backend_classes.items()}
        for backend in self.backends.values():
            backend.init_runtime(None)
        self.flash_kda_fwd = self.backends["prefill"].flash_kda_fwd

    def create_att_prefill_state(self, infer):
        return KDAStep(self, infer, "prefill")

    def create_att_decode_state(self, infer):
        return KDAStep(self, infer, "decode")


class KDAStep:
    """One step of a KDA layer with the backend state for that microbatch."""

    def __init__(self, attention, infer, phase):
        self.attention, self.infer, self.phase = attention, infer, phase
        self.att_metadata = None

    def init_state(self):
        self.att_metadata = self.attention.backends[self.phase].build_att_metadata(self.infer)

    def run(self, params, alloc_func=torch.empty):
        """params is kda_params(...); returns [rows, num_heads, head_dim]."""
        from lightllm.common.attention.backends.kda import KDALayerInputs

        config = self.attention.layer_config
        inputs = KDALayerInputs(
            mixed_qkv=params["mixed_qkv"],
            raw_gate=params["raw_gate"],
            raw_beta=params["raw_beta"],
            conv_weight=params["conv_weight"],
            A_log=params["A_log"],
            dt_bias=params["dt_bias"],
            conv_states=params["conv_states"],
            ssm_states=params["ssm_states"],
            replay_state=params["state"] if config.use_ssm_replay else None,
            logical_layer_id=params["logical_layer_id"],
        )
        x = inputs.mixed_qkv
        out = alloc_func((x.shape[0], config.num_heads, config.head_dim), dtype=x.dtype, device=x.device)
        return self.attention.backends[self.phase].forward(self.att_metadata, inputs, out=out, alloc_func=alloc_func)
