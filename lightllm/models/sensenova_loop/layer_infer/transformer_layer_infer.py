from lightllm.models.sensenova_loop.triton_kernel.mhc import hc_contract, hc_expand, hc_post, hc_pre_norm
from lightllm.models.sensenova_loop.triton_kernel.mhc.pre_norm import has_deepgemm_mhc
from lightllm.models.qwen3_5.layer_infer.transformer_layer_infer import Qwen35TransformerLayerInfer


class SenseNovaLoopTransformerLayerInfer(Qwen35TransformerLayerInfer):
    def __init__(self, layer_num, network_config):
        super().__init__(layer_num, network_config)
        self.is_linear_attention_layer = network_config["layer_types"][layer_num] == "linear_attention"
        self.physical_layer_num = network_config["loop_physical_indices"][layer_num]
        self.mhc = network_config.get("mhc")
        self.use_mhc = self.mhc is not None and self.mhc["start"] <= self.physical_layer_num < self.mhc["end"]
        if not self.use_mhc:
            return
        self.use_deepgemm_mhc = self.mhc["streams"] == 4 and has_deepgemm_mhc()
        self.mhc_entry = self.physical_layer_num == self.mhc["start"]
        self.mhc_exit = self.physical_layer_num == self.mhc["end"] - 1

    def context_forward(self, hidden, infer_state, layer_weight):
        if not self.use_mhc:
            return super().context_forward(hidden, infer_state, layer_weight)
        return self._mhc_forward(hidden, infer_state, layer_weight, prefill=True)

    def token_forward(self, hidden, infer_state, layer_weight):
        if not self.use_mhc:
            return super().token_forward(hidden, infer_state, layer_weight)
        return self._mhc_forward(hidden, infer_state, layer_weight, prefill=False)

    def _mhc_pre(self, residual, router_weight, norm_weight):
        return hc_pre_norm(
            x=residual,
            fn=router_weight.weight,
            base=router_weight.bias,
            norm_weight=norm_weight.weight,
            streams=self.mhc["streams"],
            rms_eps=1e-6,
            norm_eps=self.eps_,
            sinkhorn_iters=self.mhc["iterations"],
            use_deepgemm=self.use_deepgemm_mhc,
        )

    def _mhc_post(self, residual, branch, mix, post):
        # SenseNova routers map output -> input; the common kernel uses input -> output.
        return hc_post(branch, residual, mix.transpose(1, 2), post, self.mhc["streams"])

    def _mhc_forward(self, hidden, infer_state, layer_weight, *, prefill):
        streams = self.mhc["streams"]
        if self.mhc_entry:
            hidden = hc_expand(hidden, streams)

        branch, mix, post = self._mhc_pre(hidden, layer_weight.mhc_attn, layer_weight.att_norm_weight_)
        if prefill:
            branch = self.context_attention_forward(branch, infer_state, layer_weight)
        else:
            branch = self.token_attention_forward(branch, infer_state, layer_weight)
        branch = branch.view(-1, self.embed_dim_)
        hidden = self._mhc_post(hidden, branch, mix, post)
        branch, mix, post = self._mhc_pre(hidden, layer_weight.mhc_ffn, layer_weight.ffn_norm_weight_)
        branch = self._ffn(branch, infer_state, layer_weight).view(-1, self.embed_dim_)
        hidden = self._mhc_post(hidden, branch, mix, post)
        if self.mhc_exit:
            return hc_contract(hidden, streams)
        return hidden
