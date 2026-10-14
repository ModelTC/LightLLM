import torch
from lightllm.common.basemodel.batch_objs import PostLayerOutput

from lightllm.models.llama.layer_infer.post_layer_infer import LlamaPostLayerInfer


class Qwen3DFlashPostLayerInfer(LlamaPostLayerInfer):
    def token_forward(self, input_embdings: torch.Tensor, infer_state, layer_weight):
        if infer_state.is_prefill:
            # 提交 prefill 时只写入 draft KV，但 BaseModel 仍要求返回 logits 字段。
            return PostLayerOutput(logits=input_embdings.new_empty((0,)))
        return super().token_forward(
            input_embdings=input_embdings,
            infer_state=infer_state,
            layer_weight=layer_weight,
        )
