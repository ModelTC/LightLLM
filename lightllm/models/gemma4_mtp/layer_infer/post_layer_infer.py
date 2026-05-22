import torch
import numpy as np
from lightllm.models.llama.layer_infer.post_layer_infer import LlamaPostLayerInfer
from lightllm.distributed.communication_op import all_gather


class Gemma4MTPPostLayerInfer(LlamaPostLayerInfer):
    def __init__(self, network_config):
        super().__init__(network_config)
        cap = network_config.get("final_logit_softcapping")
        self.final_logit_softcapping = float(cap) if cap else None

        self.use_ordered_embeddings_ = bool(network_config.get("use_ordered_embeddings"))
        self._post_projection_weight_ = None
        if self.use_ordered_embeddings_:
            self.num_centroids_ = network_config["num_centroids"]
            self.centroid_top_k_ = network_config["centroid_intermediate_top_k"]
            self.vocab_size_ = network_config["vocab_size"]
            assert (
                self.vocab_size_ % self.num_centroids_ == 0
            ), f"vocab_size={self.vocab_size_} must be divisible by num_centroids={self.num_centroids_}"
            self._vocab_per_centroid_ = self.vocab_size_ // self.num_centroids_
            # token -> centroid mapping is derived lazily from the loaded
            # token_ordering buffer (weights are not loaded yet at __init__).
            self._centroid_of_token_ = None

    def _dense_logits(self, last_hidden, token_num, input_embdings_dtype, infer_state, layer_weight):
        lm_input = last_hidden.permute(1, 0).view(-1, token_num)
        logic_batch = layer_weight.lm_head_weight_(input=lm_input, alloc_func=self.alloc_tensor)
        vocab_size = layer_weight.lm_head_weight_.vocab_size
        if self.tp_world_size_ == 1:
            gather_data = logic_batch
        else:
            gather_data = self.alloc_tensor((vocab_size, token_num), dtype=input_embdings_dtype)
            split_indexes = np.linspace(0, vocab_size, self.tp_world_size_ + 1, dtype=np.int64)
            all_gather(
                [gather_data[split_indexes[i] : split_indexes[i + 1], :] for i in range(self.tp_world_size_)],
                logic_batch,
                group=infer_state.dist_group,
                async_op=False,
            )
        logic_batch = None
        ans_logics = self.alloc_tensor((token_num, vocab_size), dtype=torch.float32)
        ans_logics[:, :] = gather_data.permute(1, 0)
        gather_data = None
        return ans_logics

    def _centroid_logits(self, last_hidden, token_num, layer_weight):
        """Gather lm_head rows for the per-token top-K centroid blocks,
        dot with the post-norm hidden, scatter into a [N, vocab] -inf tensor
        at the original vocab positions. Mathematically equivalent to
        dense logits + mask but avoids the [N, vocab] bool tensor and matches
        the reference implementations exactly.
        """
        centroid_scores = layer_weight.centroids_weight_.mm(last_hidden)  # [N, num_centroids]
        topk_centroids = torch.topk(centroid_scores, k=self.centroid_top_k_, dim=-1).indices  # [N, K]
        # token_ordering[i] = original vocab id at reordered position i;
        # row c of the (C, vpc) view holds the vocab ids of centroid c.
        token_ordering = layer_weight.token_ordering_.weight  # [vocab] int64
        clusters = token_ordering.view(self.num_centroids_, self._vocab_per_centroid_)  # [C, vpc]
        selected_vocab = clusters[topk_centroids]  # [N, K, vpc] - original vocab ids
        num_selected = self.centroid_top_k_ * self._vocab_per_centroid_
        selected_vocab = selected_vocab.reshape(token_num, num_selected)  # [N, num_selected]
        # Gather lm_head rows for the selected vocab ids.
        lm_head_w = layer_weight.lm_head_weight_.weight  # [vocab, draft_hidden]
        H = lm_head_w.shape[1]
        selected_embeddings = lm_head_w.index_select(0, selected_vocab.view(-1)).view(token_num, num_selected, H)
        # Sparse logits: dot product per token vs its selected rows.
        selected_logits = torch.einsum("nh,nsh->ns", last_hidden, selected_embeddings)
        # Scatter to [N, vocab] with -inf elsewhere.
        output = torch.full(
            (token_num, self.vocab_size_),
            float("-inf"),
            dtype=selected_logits.dtype,
            device=selected_logits.device,
        )
        output.scatter_(-1, selected_vocab, selected_logits)
        return output

    def token_forward(self, input_embdings, infer_state, layer_weight):
        last_hidden, token_num = self._slice_get_last_input(input_embdings, infer_state)
        last_hidden = self._norm(last_hidden, infer_state, layer_weight)
        if self.use_ordered_embeddings_:
            logits = self._centroid_logits(last_hidden, token_num, layer_weight)
        else:
            logits = self._dense_logits(last_hidden, token_num, input_embdings.dtype, infer_state, layer_weight)
        if self.final_logit_softcapping is not None and self.final_logit_softcapping > 0:
            cap = self.final_logit_softcapping
            logits = torch.tanh(logits / cap) * cap
        assert self._post_projection_weight_ is not None, "post_projection weight is not initialized"
        return logits, self._post_projection_weight_.mm(last_hidden)
