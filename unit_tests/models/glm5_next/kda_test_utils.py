"""Build KDA layer inputs and attention the way GLM-5-Next layers do (see _kda_layer_inputs)."""

from types import SimpleNamespace

import torch

from lightllm.common.attention.backend import AttentionMetadata


def kda_params(conv_states, ssm_states, layer_weight, mixed_qkv, raw_gate, raw_beta, *, state=None, layer_num=0):
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
        layer_num=layer_num,
    )


def prepare_prefill_infer(infer, device="cuda"):
    """Fill the request fields a prefill step builder reads but a test stub does not set."""
    infer.is_prefill = True
    for name in ("is_cuda_graph_capture", "is_cuda_graph_replay"):
        setattr(infer, name, getattr(infer, name, False))
    infer.batch_size = infer.b_req_idx.shape[0]
    infer.b_seq_len = getattr(infer, "b_seq_len", None)
    infer.max_kv_seq_len = getattr(infer, "max_kv_seq_len", None)
    infer.total_token_num = getattr(infer, "total_token_num", None)
    infer.b1_cu_kv_seq_len = getattr(infer, "b1_cu_kv_seq_len", None)
    infer.b_q_start_loc = getattr(infer, "b_q_start_loc", None)
    infer.b_ready_cache_len = getattr(infer, "b_ready_cache_len", None)
    infer.max_cache_len = getattr(infer, "max_cache_len", 0)
    infer.mem_index = getattr(infer, "mem_index", None)
    infer.microbatch_index = getattr(infer, "microbatch_index", 0)
    infer.req_manager = getattr(infer, "req_manager", SimpleNamespace(req_to_token_indexs=None))
    return infer


def prefill_batch(max_q_seq_len, device="cpu"):
    b_req_idx = torch.zeros(1, dtype=torch.int32, device=device)
    return AttentionMetadata(
        req_to_token_indexs=None,
        b_req_idx=b_req_idx,
        b_seq_len=b_req_idx,
        batch_size=1,
        max_kv_seq_len=max_q_seq_len,
        total_token_num=max_q_seq_len,
        cu_q_seq_len=torch.tensor([0, max_q_seq_len], dtype=torch.int32, device=device),
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
    """The layer config a GLM-5 KDA layer creates (Glm5NextTransformerLayerInfer._init_kda_attention)."""
    from lightllm.common.attention.backends.kda import KDALayerConfig

    return KDALayerConfig(
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

    model supplies what build_attention_metadata reads (mtp_manager, is_mtp_draft_model for decode rows).
    """

    def __init__(self, layer_config, model=None, *, dynamic_layout=False):
        from lightllm.common.attention.backends.kda import triton as kda

        self.layer_config, self.dynamic_layout = layer_config, dynamic_layout
        self.model = SimpleNamespace(is_mtp_draft_model=False, mtp_manager=None) if model is None else model
        replay = layer_config.replay
        backends = {
            "prefill": kda.ReplayKDAPrefillBackend if replay else kda.TritonKDAPrefillBackend,
            "decode": kda.ReplayKDADecodeBackend if replay else kda.TritonKDADecodeBackend,
        }
        self.impls = {phase: backend.get_impl_cls()(layer_config) for phase, backend in backends.items()}
        self.step_builders = {
            phase: backend.create_step_builder(layer_config, None) for phase, backend in backends.items()
        }
        self.flash_kda_fwd = self.step_builders["prefill"].flash_kda_fwd

    def create_att_prefill_state(self, infer):
        return KDAStep(self, infer, "prefill")

    def create_att_decode_state(self, infer):
        return KDAStep(self, infer, "decode")


class KDAStep:
    """One step of a KDA layer: init_state builds the attention state as InferStateInfo.init_att_state does."""

    def __init__(self, attention, infer, phase):
        self.attention, self.infer, self.phase = attention, infer, phase
        self.att_state = None

    def init_state(self):
        from lightllm.common.basemodel.attention_runtime import build_attention_metadata, recurrent_mtp_state_index

        infer, attention = self.infer, self.attention
        index = None
        if getattr(infer, "att_state_manager", None) is not None:
            role = SimpleNamespace(att_state_manager=infer.att_state_manager, att_state_role=infer.att_state_role)
            index = recurrent_mtp_state_index(role)
        batch = build_attention_metadata(
            infer, attention.model, lambda: attention.dynamic_layout, req_to_mtp_state_index=index
        )
        self.att_state = attention.step_builders[self.phase].build(batch)

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
            replay_state=params["state"] if config.replay else None,
            layer_num=params["layer_num"],
        )
        x = inputs.mixed_qkv
        out = alloc_func((x.shape[0], config.num_heads, config.head_dim), dtype=x.dtype, device=x.device)
        return self.attention.impls[self.phase].forward(self.att_state, inputs, out=out, alloc_func=alloc_func)
