import rpyc
import socket
import torch
import torch.distributed as dist

from lightllm.models.llama.layer_weights.pre_and_post_layer_weight import LlamaPreAndPostLayerWeight
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.models.llama.layer_infer.pre_layer_infer import LlamaPreLayerInfer
from lightllm.server.embed_cache.utils import get_shm_name_embed, load_tensor_afs
from lightllm.common.basemodel.triton_kernel.multimodal_emb import multimodal_emb
from lightllm.distributed.communication_op import all_reduce
from lightllm.utils.envs_utils import get_env_start_args


"""
infer_state.multimodal_params: batch list of MultimodalParams-dict like:
   {
       "images": [
           {
               "uuid": int,
               "token_id": int, image token start id,
               "token_num": int, image token num,
           },
       ]
       ...
   }
"""


class LlamaMultimodalPreLayerInfer(LlamaPreLayerInfer):
    def __init__(self, network_config):
        super().__init__(network_config)
        self.args = get_env_start_args()
        if self.args.enable_remote_vit:
            self.cache_client = rpyc.connect("localhost", self.args.cache_port, config={"allow_pickle": True})
            self.cache_client._channel.stream.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return

    def _copy_loaded_embed_to_cache(
        self, embed_tensor: torch.Tensor, cpu_embed_cache_tensor: torch.Tensor, start_index: int
    ):
        if embed_tensor.ndim == 2:
            embed_tensor = embed_tensor.unsqueeze(1)

        token_num, layer_num, hidden_size = embed_tensor.shape
        cpu_embed_cache_tensor[start_index : start_index + token_num, :layer_num, :hidden_size].copy_(embed_tensor)
        return

    def context_forward(self, input_ids, infer_state: LlamaInferStateInfo, layer_weight: LlamaPreAndPostLayerWeight):
        img_start_token_ids = []
        img_token_lens = []
        img_start_locs_in_cache = []
        unique_uids = []
        all_uids = []
        device = layer_weight.wte_weight_.weight.device
        dtype = layer_weight.wte_weight_.weight.dtype
        hidden_size = layer_weight.wte_weight_.weight.shape[1]

        for _, p in enumerate(infer_state.multimodal_params):
            for img in p["images"] + p["audios"]:
                all_uids.append(img["uuid"])
                # skip the same image
                if img["token_id"] in img_start_token_ids:
                    continue
                img_start_token_ids.append(img["token_id"])
                img_token_lens.append(img["token_num"])
                img_start_locs_in_cache.append(img["start_index_in_embed_cache"])
                unique_uids.append(img["uuid"])
        out = torch.zeros((len(input_ids), hidden_size), dtype=dtype, device=device)

        from lightllm.server.router.model_infer.infer_batch import g_infer_context

        cpu_embed_cache_client = g_infer_context.cpu_embed_cache_client
        cpu_embed_cache_tensor = (
            torch.empty((0, 0, hidden_size), dtype=dtype, device=device)
            if cpu_embed_cache_client is None
            else cpu_embed_cache_client.cpu_embed_cache_tensor
        )

        if self.args.enable_remote_vit:
            for uid, start_index_in_embed_cache in zip(unique_uids, img_start_locs_in_cache):
                embed_tensor = load_tensor_afs(get_shm_name_embed(uid), self.args.image_embed_dir)
                self._copy_loaded_embed_to_cache(embed_tensor, cpu_embed_cache_tensor, start_index_in_embed_cache)

            if all_uids:
                self.cache_client.root.release(all_uids)

        assert cpu_embed_cache_tensor.shape[2] == hidden_size, (
            f"Dimension mismatch: text weight dimension is {hidden_size}, "
            f"but image embed dimension is {cpu_embed_cache_tensor.shape[2]}"
        )
        # each tp will fill the img embeds, should divide by world_size
        img_start_token_ids = torch.tensor(img_start_token_ids, dtype=torch.long, device="cpu", pin_memory=True).cuda(
            non_blocking=True
        )
        img_token_lens = torch.tensor(img_token_lens, dtype=torch.long, device="cpu", pin_memory=True).cuda(
            non_blocking=True
        )
        img_start_locs_in_cache = torch.tensor(
            img_start_locs_in_cache, dtype=torch.long, device="cpu", pin_memory=True
        ).cuda(non_blocking=True)

        multimodal_emb(
            out=out,
            prompt_ids=input_ids,
            text_weight_embs=layer_weight.wte_weight_.weight,
            embed_cache=cpu_embed_cache_tensor,
            img_token_lens=img_token_lens,
            img_start_token_ids=img_start_token_ids,
            img_start_locs_in_cache=img_start_locs_in_cache,
            tp_text_start_token_id=layer_weight.wte_weight_.tp_vocab_start_id,
            tp_text_end_token_id=layer_weight.wte_weight_.tp_vocab_end_id,
            tp_world_size=self.tp_world_size_,
        )
        if self.tp_world_size_ > 1:
            all_reduce(out, group=infer_state.dist_group, op=dist.ReduceOp.SUM, async_op=False)
        return out
