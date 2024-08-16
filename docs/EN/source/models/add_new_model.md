# How to Add New Model Support

## 1. Introduction of inference architecture

In the lightllm/common/basemodel directory, you will find the base class implementation for the entire inference architecture.

~~~shell
├── basemodel.py   # Model architecture class
├── infer_struct.py # State class for inference
├── __init__.py
├── layer_infer # Inference layer base class
│   ├── base_layer_infer.py
│   ├── __init__.py
│   ├── post_layer_infer.py
│   ├── pre_layer_infer.py
│   ├── template # Template implementation of the inference layer. 
│   │   ├── __init__.py
│   │   ├── post_layer_infer_template.py
│   │   ├── pre_layer_infer_template.py
│   │   └── transformer_layer_infer_template.py
│   └── transformer_layer_infer.py
├── layer_weights # base class of weight
│   ├── base_layer_weight.py
│   ├── hf_load_utils.py
│   ├── __init__.py
│   ├── pre_and_post_layer_weight.py
│   └── transformer_layer_weight.py
└── triton_kernel # Some commonly used triton kernel operators
    ├── apply_penalty.py
    ├── destindex_copy_kv.py
    └── __init__.py
~~~

As shown above, the current model inference architecture mainly consists of two parts: weight and inference.

### Weight

The layer_weights directory contains weight-related codes. In theory, a newly added model needs to inherit the PreAndPostLayerWeight and TransformerLayerWeight classes in pre_and_post_layer_weight.py and transformer_layer_weight.py to load weights.

| Weight base class      | Responsibilities                                                         |
| ---------------------- | ------------------------------------------------------------ |
| PreAndPostLayerWeight  | Responsible for loading the weights of the first Embedding layer and the last post-processing layer of the LLM model and splitting the weights according to the tp parameters used |
| TransformerLayerWeight | Responsible for loading the weights of the LLM model transformer layer and splitting the weights according to the tp parameters used |

### Inference

The layer_infer directory contains the base classes for inference processing, and some templates are provided in the template directory. Inheriting from the template class can reduce some unnecessary duplication of code and simplify the implementation. There are three inference classes that need to be inherited in this directory.

| Inference base class  | Responsibilities                    |
| --------------------- | ------------------------------------------ |
| PreLayerInfer         | Responsible for inference of the Embedding layer                  |
| TransformerLayerInfer | Responsible for inference of th transformer layer                |
| PostLayerInfer        | Responsible for inference of converting the final hidden layer output of the network into logits  |

The base class BaseLayerInfer of the above three classes provides two most important external service function interfaces. All inference behaviors will be entered through these two interfaces.

| interface                                                    | Responsibilities                                           |
| ------------------------------------------------------------ | ---------------------------------------------- |
| def context_forward(self, input_ids, infer_state: InferStateInfo, layer_weight: BaseLayerWeight): | the first inference of batch（prefill） |
| def token_forward(self, input_ids, infer_state: InferStateInfo, layer_weight: BaseLayerWeight): | the inference of  decode      |

### Operator

The triton_kernel directory contains some operators needed for inference implemented using openai triton.

### State class

The InferStateInfo class in infer_struct.py is a state class that passes some important information between layers when performing a model inference. Different models can inherit and implement this class to add unique state information that each model needs to pass. The InferStateInfo class provides an inherited init_some_extra_state interface for initializing the transmission of additional unique information.

~~~python
    def init_some_extra_state(self, 
            model, 
            batch_size, 
            total_token_num,
            max_len_in_batch,
            input_ids : torch.Tensor,
            b_loc : torch.Tensor,
            b_start_loc : torch.Tensor,
            b_seq_len : torch.Tensor,
            is_prefill):
        pass
~~~

### Model class

The TpPartBaseModel class in basemodel.py is the entry point of the entire model. Each type of model needs to inherit and implement this class. This class uses the inference class, weight class, and state class to complete the model loading and inference functions in a similar way to building blocks. Many of its interfaces can be inherited and implemented to complete the unique operations of each model type.

~~~python
class TpPartBaseModel:
    # weight class
    pre_and_post_weight_class = None
    transformer_weight_class = None

    # infer class
    pre_layer_infer_class = None
    post_layer_infer_class = None
    transformer_layer_infer_class = None

    # infer state class
    infer_state_class = InferStateInfo

    def __init__(self, tp_rank, world_size, weight_dir, max_total_token_num, load_way="HF", mode=[]):
        self.tp_rank_ = tp_rank
        self.world_size_ = world_size
        self.weight_dir_ = weight_dir
        self.max_total_token_num = max_total_token_num
        self.load_way = load_way
        self.mode = mode

        self._init_config()
        self._verify_must()
        self._verify_params()
        self._init_weights()
        self._init_mem_manager()
        self._init_infer_layer()
        self._init_some_value()
        self._init_custom()
        return
   ...
   ...
~~~

Common interfaces that need to be inherited and implemented

| interfaces                   | effect                                                         |
| ---------------------------- | ------------------------------------------------------------ |
| def _init_config(self)：     | Read the config.json of the initialization model and perform some key name legalization operations |
| def _verify_params(self)：   | Verification parameters                                                     |
| def _init_mem_manager(self): | Initialize the mem manager object used by token attention               |
| def _init_some_value(self):  | Initialize the values ​​of some member variables used by the inference framework                       |
| def _init_custom(self):      | Some models have their own personalized initialization, such as llama initializing its own Rotary value  |

## 2. the example  of adding bloom model

The specific implementation is in the ***lightllm/models/bloom*** directory. Please read the corresponding source code for the following code snippets. The triton_kernel directory contains some kernels used by the inference class, which will not be introduced in detail below. At the same time, the bloom model uses the default state class because it does not need to pass special state information. If you want to understand the entire framework more deeply, you can further refer to the access implementation source code of models such as llama and llama2.

### （1） Add implementation weight class

***pre_and_post_layer_weight.py***

~~~python
import torch
import numpy as np
from lightllm.common.basemodel import PreAndPostLayerWeight

class BloomPreAndPostLayerWeight(PreAndPostLayerWeight):
    def __init__(self, tp_rank, world_size, data_type, network_config, mode):
        super().__init__(tp_rank, world_size, data_type, network_config, mode)

    def load_hf_weights(self, weights):

        if "word_embeddings_layernorm.weight" in weights:
            self.pre_norm_weight_ = self._cuda(weights['word_embeddings_layernorm.weight'])
        if "word_embeddings_layernorm.bias" in weights:
            self.pre_norm_bias_ = self._cuda(weights['word_embeddings_layernorm.bias'])
        if "ln_f.weight" in weights:
            self.final_norm_weight_ = self._cuda(weights['ln_f.weight'])
        if "ln_f.bias" in weights:
            self.final_norm_bias_ = self._cuda(weights["ln_f.bias"])
        if "word_embeddings.weight" in weights:
            vob_size = self.network_config_["vocab_size"]
            split_vob_size = vob_size // self.world_size_
            self.wte_weight_ = self._cuda(weights["word_embeddings.weight"][split_vob_size *
                                                                 self.tp_rank_: split_vob_size * (self.tp_rank_ + 1), :])
            self.lm_head_weight_ = self.wte_weight_
        return
    
    def verify_load(self):
        errors = "weights load not ok"
        weights = [self.pre_norm_weight_, 
                   self.pre_norm_bias_, 
                   self.final_norm_weight_, 
                   self.final_norm_bias_,
                   self.wte_weight_,
                   self.lm_head_weight_]
        for i in range(len(weights)):
            assert weights[i] is not None, "index:" + str(i) + " " + errors
        return 

~~~

***transformer_layer_weight.py***

~~~python
import torch
import math
import numpy as np
from lightllm.common.basemodel import TransformerLayerWeight


class BloomTransformerLayerWeight(TransformerLayerWeight):
    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config, mode):
        super().__init__(layer_num, tp_rank, world_size, data_type, network_config, mode)
        return

    def init_static_params(self):
        head_num = self.network_config_["num_attention_heads"]
        tp_head_num = head_num // self.world_size_
        tmp_alibi = self._generate_alibi(head_num, dtype=torch.float32)
        assert head_num % self.world_size_ == 0
        self.tp_alibi = tmp_alibi[self.tp_rank_ * tp_head_num: (self.tp_rank_ + 1) * tp_head_num].contiguous().cuda()
        return
    
    def load_hf_weights(self, weights):

        self._load_qkvo_weights(weights)
        self._load_ffn_weights(weights)
        return
    
    def verify_load(self):
        errors = "weights load not ok"
        weights = [self.att_norm_weight_,
                   self.att_norm_bias_,
                   self.q_weight_,
                   self.k_weight_,
                   self.v_weight_,
                   self.q_bias_,
                   self.k_bias_,
                   self.v_bias_,
                   self.o_weight_,
                   self.o_bias_,

                   self.ffn_norm_weight_,
                   self.ffn_norm_bias_,
                   self.ffn_1_weight_,
                   self.ffn_1_bias_,
                   self.ffn_2_weight_,
                   self.ffn_2_bias_,
                   ]
        for i in range(len(weights)):
            assert weights[i] is not None, "index:" + str(i) + " " + errors
        return 

    def _load_qkvo_weights(self, weights):
        if f"h.{self.layer_num_}.input_layernorm.weight" in weights:
            self.att_norm_weight_ = self._cuda(weights[f"h.{self.layer_num_}.input_layernorm.weight"])
        if f"h.{self.layer_num_}.input_layernorm.bias" in weights:
            self.att_norm_bias_ = self._cuda(weights[f"h.{self.layer_num_}.input_layernorm.bias"])

        if f"h.{self.layer_num_}.self_attention.query_key_value.weight" in weights:
            n_embed = self.network_config_["n_embed"]
            split_n_embed = n_embed // self.world_size_
            head_num = self.network_config_["num_attention_heads"]
            att_qkv_dense_weight = weights[f"h.{self.layer_num_}.self_attention.query_key_value.weight"].reshape(head_num, 3, -1, n_embed)
            self.q_weight_ = self._cuda(att_qkv_dense_weight[:,
                                                  0,
                                                  :,
                                                  :].reshape(-1,
                                                             n_embed)[split_n_embed * self.tp_rank_: split_n_embed * (self.tp_rank_ + 1),
                                                                      :].transpose(0,
                                                                                   1))
            self.k_weight_ = self._cuda(att_qkv_dense_weight[:,
                                                  1,
                                                  :,
                                                  :].reshape(-1,
                                                             n_embed)[split_n_embed * self.tp_rank_: split_n_embed * (self.tp_rank_ + 1),
                                                                      :].transpose(0,
                                                                                   1))
            self.v_weight_ = self._cuda(att_qkv_dense_weight[:,
                                                  2,
                                                  :,
                                                  :].reshape(-1,
                                                             n_embed)[split_n_embed * self.tp_rank_: split_n_embed * (self.tp_rank_ + 1),
                                                                      :].transpose(0,
                                                                                   1))
        if f"h.{self.layer_num_}.self_attention.query_key_value.bias" in weights:
            n_embed = self.network_config_["n_embed"]
            split_n_embed = n_embed // self.world_size_
            head_num = self.network_config_["num_attention_heads"]
            att_qkv_dense_bias = weights[f"h.{self.layer_num_}.self_attention.query_key_value.bias"].reshape(head_num, 3, -1)
            self.q_bias_ = self._cuda(att_qkv_dense_bias[:, 0, :].reshape(-1)[split_n_embed *
                                                                   self.tp_rank_: split_n_embed * (self.tp_rank_ + 1)])
            self.k_bias_ = self._cuda(att_qkv_dense_bias[:, 1, :].reshape(-1)[split_n_embed *
                                                                   self.tp_rank_: split_n_embed * (self.tp_rank_ + 1)])
            self.v_bias_ = self._cuda(att_qkv_dense_bias[:, 2, :].reshape(-1)[split_n_embed *
                                                                   self.tp_rank_: split_n_embed * (self.tp_rank_ + 1)])

        if f"h.{self.layer_num_}.self_attention.dense.weight" in weights:
            n_embed = self.network_config_["n_embed"]
            split_n_embed = n_embed // self.world_size_
            self.o_weight_ = self._cuda(weights[f"h.{self.layer_num_}.self_attention.dense.weight"][:,
                                                                                                     split_n_embed * self.tp_rank_: split_n_embed * (self.tp_rank_ + 1)].transpose(0, 1))
        if f"h.{self.layer_num_}.self_attention.dense.bias" in weights:
            self.o_bias_ = self._cuda(weights[f"h.{self.layer_num_}.self_attention.dense.bias"])
        return 

    def _load_ffn_weights(self, weights):
        if f"h.{self.layer_num_}.post_attention_layernorm.weight" in weights:
            self.ffn_norm_weight_ = self._cuda(weights[f"h.{self.layer_num_}.post_attention_layernorm.weight"])
            self.ffn_norm_bias_ = self._cuda(weights[f"h.{self.layer_num_}.post_attention_layernorm.bias"])

        # ffn params
        if f"h.{self.layer_num_}.mlp.dense_h_to_4h.weight" in weights:
            n_embed = self.network_config_["n_embed"] * 4
            split_n_embed = n_embed // self.world_size_
            self.ffn_1_weight_ = weights[f"h.{self.layer_num_}.mlp.dense_h_to_4h.weight"]
            self.ffn_1_weight_ = self._cuda(self.ffn_1_weight_[split_n_embed * self.tp_rank_: split_n_embed *
                                                    (self.tp_rank_ + 1), :].transpose(0, 1))

        if f"h.{self.layer_num_}.mlp.dense_h_to_4h.bias" in weights:
            n_embed = self.network_config_["n_embed"] * 4
            split_n_embed = n_embed // self.world_size_
            self.ffn_1_bias_ = self._cuda(weights[f"h.{self.layer_num_}.mlp.dense_h_to_4h.bias"][split_n_embed *
                                                                                      self.tp_rank_: split_n_embed * (self.tp_rank_ + 1)])
        if f"h.{self.layer_num_}.mlp.dense_4h_to_h.weight" in weights:
            n_embed = self.network_config_["n_embed"] * 4
            split_n_embed = n_embed // self.world_size_
            self.ffn_2_weight_ = weights[f"h.{self.layer_num_}.mlp.dense_4h_to_h.weight"]
            self.ffn_2_weight_ = self._cuda(self.ffn_2_weight_[:, split_n_embed *
                                                    self.tp_rank_: split_n_embed * (self.tp_rank_ + 1)].transpose(0, 1))

        if f"h.{self.layer_num_}.mlp.dense_4h_to_h.bias" in weights:
            self.ffn_2_bias_ = self._cuda(weights[f"h.{self.layer_num_}.mlp.dense_4h_to_h.bias"])
        return 

    def _generate_alibi(self, n_head, dtype=torch.float16):
        def get_slopes(n):
            def get_slopes_power_of_2(n):
                start = 2 ** (-(2 ** -(math.log2(n) - 3)))
                ratio = start
                return [start * ratio**i for i in range(n)]

            if math.log2(n).is_integer():
                return get_slopes_power_of_2(n)
            else:
                closest_power_of_2 = 2 ** math.floor(math.log2(n))
                return (
                    get_slopes_power_of_2(closest_power_of_2)
                    + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
                )

        slopes = torch.Tensor(get_slopes(n_head))
        head_alibi = slopes.to(dtype)
        return head_alibi
~~~

### (2) Add implementation inference class

***pre_layer_infer.py***

~~~python
import torch
import torch.distributed as dist
from lightllm.common.basemodel import PreLayerInferTpl
from lightllm.common.basemodel import InferStateInfo
from lightllm.models.bloom.layer_weights.pre_and_post_layer_weight import BloomPreAndPostLayerWeight
from lightllm.utils.infer_utils import mark_cost_time
from lightllm.models.bloom.triton_kernel.layernorm import layernorm_forward

class BloomPreLayerInfer(PreLayerInferTpl):
    """
    """
    def __init__(self, tp_rank, world_size, network_config, mode):
        super().__init__(tp_rank, world_size, network_config, mode)
        self.eps_ = network_config["layer_norm_epsilon"]
        tp_vocab_size_ = network_config["vocab_size"] // self.world_size_
        self.vob_start_id_ = tp_vocab_size_ * self.tp_rank_
        self.vob_end_id_ = tp_vocab_size_ * (self.tp_rank_ + 1)
        return
    
    def _norm(self, input, infer_state, layer_weight : BloomPreAndPostLayerWeight) -> torch.Tensor:
        return layernorm_forward(input, layer_weight.pre_norm_weight_, layer_weight.pre_norm_bias_, eps=self.eps_)

    def context_forward(self, input_ids, infer_state: InferStateInfo, layer_weight: BloomPreAndPostLayerWeight):
        total_token_num = infer_state.total_token_num
        input_ids = input_ids[0:total_token_num]

        input_mask = torch.logical_or(self.vob_start_id_ > input_ids, input_ids >= self.vob_end_id_)
        tmp_input_ids = (input_ids - self.vob_start_id_)
        tmp_input_ids[input_mask] = 0
        input_embdings = torch.embedding(layer_weight.wte_weight_, tmp_input_ids, padding_idx=-1)
        input_embdings[input_mask] = 0.0
        if self.world_size_ > 1:
            dist.all_reduce(input_embdings, op=dist.ReduceOp.SUM, async_op=False)
        input_embdings = self._norm(input_embdings, infer_state, layer_weight)
        return input_embdings

    def token_forward(self, input_ids, infer_state: InferStateInfo, layer_weight: BloomPreAndPostLayerWeight):
        input_mask = torch.logical_or(self.vob_start_id_ > input_ids, input_ids >= self.vob_end_id_)
        tmp_input_ids = (input_ids - self.vob_start_id_)
        tmp_input_ids[input_mask] = 0
        input_embdings = torch.embedding(layer_weight.wte_weight_, tmp_input_ids, padding_idx=-1)
        input_embdings[input_mask] = 0.0
        if self.world_size_ > 1:
            dist.all_reduce(input_embdings, op=dist.ReduceOp.SUM, async_op=False)
        input_embdings = self._norm(input_embdings, infer_state, layer_weight)
        return input_embdings
~~~

***transformer_layer_infer.py***

~~~python
import time
import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np

from lightllm.common.basemodel import TransformerLayerInferTpl
from lightllm.models.bloom.layer_weights.transformer_layer_weight import BloomTransformerLayerWeight
from lightllm.models.bloom.triton_kernel.context_flashattention_nopad import context_attention_fwd
from lightllm.models.bloom.triton_kernel.token_flashattention_nopad import token_attention_fwd
from lightllm.models.bloom.triton_kernel.layernorm import layernorm_forward
from lightllm.common.basemodel import InferStateInfo
from lightllm.utils.infer_utils import mark_cost_time

class BloomTransformerLayerInfer(TransformerLayerInferTpl):
    """
    """

    def __init__(self, layer_num, tp_rank, world_size, network_config, mode):
        super().__init__(layer_num, tp_rank, world_size, network_config, mode)
        self.eps_ = network_config["layer_norm_epsilon"]
        self.tp_q_head_num_ = network_config["num_attention_heads"] // self.world_size_
        self.tp_k_head_num_ = self.tp_q_head_num_
        self.tp_v_head_num_ = self.tp_q_head_num_
        self.tp_o_head_num_ = self.tp_q_head_num_
        self.head_dim_ = network_config["n_embed"] // network_config["num_attention_heads"]
        self.embed_dim_ = network_config["n_embed"]
        return
    
    def _att_norm(self, input, infer_state:InferStateInfo, layer_weight: BloomTransformerLayerWeight)->torch.Tensor:
        return layernorm_forward(
            input.view(-1, self.embed_dim_),
            weight=layer_weight.att_norm_weight_,
            bias=layer_weight.att_norm_bias_,
            eps=self.eps_)
    
    def _ffn_norm(self, input, infer_state:InferStateInfo, layer_weight: BloomTransformerLayerWeight)->torch.Tensor:
        return layernorm_forward(
            input.view(-1, self.embed_dim_),
            weight=layer_weight.ffn_norm_weight_,
            bias=layer_weight.ffn_norm_bias_,
            eps=self.eps_)
    
    def _get_qkv(self, input, cache_k, cache_v, infer_state:InferStateInfo, layer_weight: BloomTransformerLayerWeight)->torch.Tensor:
        q = torch.addmm(layer_weight.q_bias_, input.view(-1, self.embed_dim_), layer_weight.q_weight_, beta=1.0, alpha=1.0)
        torch.addmm(layer_weight.k_bias_, input.view(-1, self.embed_dim_), layer_weight.k_weight_, beta=1.0,
                    alpha=1.0, out=cache_k.view(-1, self.tp_k_head_num_ * self.head_dim_))
        torch.addmm(layer_weight.v_bias_, input.view(-1, self.embed_dim_), layer_weight.v_weight_, beta=1.0,
                    alpha=1.0, out=cache_v.view(-1, self.tp_v_head_num_ * self.head_dim_))
        return q
    
    def _context_attention_kernel(self, q, kv, infer_state:InferStateInfo, layer_weight: BloomTransformerLayerWeight)->torch.Tensor:
        o_tensor = torch.empty_like(q)
        context_attention_fwd(q.view(-1, self.tp_q_head_num_, self.head_dim_),
                              kv[:, 0: self.tp_k_head_num_, :],
                              kv[:, self.tp_k_head_num_ : self.tp_k_head_num_ + self.tp_v_head_num_, :],
                              o_tensor.view(-1, self.tp_q_head_num_, self.head_dim_),
                              layer_weight.tp_alibi,
                              infer_state.b_start_loc,
                              infer_state.b_seq_len,
                              infer_state.max_len_in_batch)
        return o_tensor
    
    def _token_attention_kernel(self, q, infer_state:InferStateInfo, layer_weight: BloomTransformerLayerWeight)->torch.Tensor:
        o_tensor = torch.empty_like(q)
        token_attention_fwd(q.view(-1, self.tp_q_head_num_, self.head_dim_),
                            infer_state.mem_manager.kv_buffer[self.layer_num_][:, 0: self.tp_k_head_num_, :],
                            infer_state.mem_manager.kv_buffer[self.layer_num_][:, self.tp_k_head_num_: self.tp_k_head_num_+ self.tp_v_head_num_, :],
                            o_tensor.view(-1, self.tp_q_head_num_, self.head_dim_),
                            layer_weight.tp_alibi,
                            infer_state.b_loc,
                            infer_state.b_start_loc,
                            infer_state.b_seq_len,
                            infer_state.max_len_in_batch)
        return o_tensor
    
    def _get_o(self, input, infer_state:InferStateInfo, layer_weight: BloomTransformerLayerWeight)->torch.Tensor:
        o = torch.addmm(layer_weight.o_bias_,
                        input.view(-1, self.tp_q_head_num_ * self.head_dim_),
                        layer_weight.o_weight_,
                        beta=1.0 / self.world_size_)
        return o
    
    def _ffn(self, input, infer_state:InferStateInfo, layer_weight: BloomTransformerLayerWeight)->torch.Tensor:
        ffn1_out = torch.addmm(layer_weight.ffn_1_bias_, input.view(-1, self.embed_dim_), layer_weight.ffn_1_weight_)
        input = None
        gelu_out = torch.nn.functional.gelu(ffn1_out, approximate='tanh')
        ffn1_out = None
        ffn2_out = torch.addmm(layer_weight.ffn_2_bias_, gelu_out, layer_weight.ffn_2_weight_, beta=1.0 / self.world_size_)
        gelu_out = None
        return ffn2_out
~~~

***post_layer_infer.py***

~~~python
import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np

from lightllm.models.bloom.layer_weights.pre_and_post_layer_weight import BloomPreAndPostLayerWeight
from einops import rearrange
from lightllm.common.basemodel import InferStateInfo, PostLayerInferTpl
from lightllm.models.bloom.triton_kernel.layernorm import layernorm_forward


class BloomPostLayerInfer(PostLayerInferTpl):
    """
    """

    def __init__(self, tp_rank, world_size, network_config, mode):
        super().__init__(tp_rank, world_size, network_config, mode)
        assert (network_config["vocab_size"] % self.world_size_ == 0)
        self.eps_ = network_config["layer_norm_epsilon"]
        self.vocab_size_ = network_config["vocab_size"]
        self.embed_dim_ = network_config["n_embed"]
        return
    
    def _norm(self, input, infer_state, layer_weight : BloomPreAndPostLayerWeight) -> torch.Tensor:
        return layernorm_forward(input, layer_weight.final_norm_weight_, layer_weight.final_norm_bias_, eps=self.eps_)

    def soft_max(self, data):
        return torch.softmax(data.permute(1, 0).float(), dim=-1)

    def token_forward(self, input_embdings, infer_state: InferStateInfo, layer_weight: BloomPreAndPostLayerWeight, return_logics=False):
        batch_size = infer_state.batch_size
        last_input = torch.empty((batch_size, self.embed_dim_), device=input_embdings.device, dtype=torch.float16)
        if infer_state.is_prefill:
            last_index = torch.cumsum(infer_state.b_seq_len, dim=0, dtype=torch.long) - 1
            last_input[:, :] = input_embdings[last_index, :]
        else:
            last_input[:, :] = input_embdings[-batch_size:, :]

        input_embdings = None
        last_input = self._norm(last_input, infer_state, layer_weight)
        last_input = rearrange(last_input, "batch embed_dim -> embed_dim batch").contiguous().reshape(-1, batch_size)
        logic_batch = torch.mm(layer_weight.lm_head_weight_, last_input)
        last_input = None
        if self.world_size_ == 1:
            gather_data = logic_batch
        else:
            gather_data = torch.empty((self.vocab_size_, batch_size), device=logic_batch.device, dtype=torch.float16)
            split_size = self.vocab_size_ // self.world_size_
            dist.all_gather([gather_data[i * split_size: (i + 1) * split_size, :]
                            for i in range(self.world_size_)], logic_batch, group=None, async_op=False)
        logic_batch = None

        if not return_logics:
            prob_out = self.soft_max(gather_data)
            gather_data = None
            return prob_out
        else:
            ans_logics = gather_data.permute(1, 0).float()
            gather_data = None
            return ans_logics
~~~

### （3） Add implementation model class

***model.py***

~~~python
import os
import json
from lightllm.models.bloom.layer_infer.pre_layer_infer import BloomPreLayerInfer
from lightllm.models.bloom.layer_infer.post_layer_infer import BloomPostLayerInfer
from lightllm.models.bloom.layer_infer.transformer_layer_infer import BloomTransformerLayerInfer
from lightllm.models.bloom.layer_weights.pre_and_post_layer_weight import BloomPreAndPostLayerWeight
from lightllm.models.bloom.layer_weights.transformer_layer_weight import BloomTransformerLayerWeight
from lightllm.common.basemodel import InferStateInfo, TpPartBaseModel

from lightllm.common.build_utils import repair_config

class BloomTpPartModel(TpPartBaseModel):
    # weight class
    pre_and_post_weight_class = BloomPreAndPostLayerWeight
    transformer_weight_class = BloomTransformerLayerWeight

    # infer class
    pre_layer_infer_class = BloomPreLayerInfer
    post_layer_infer_class = BloomPostLayerInfer
    transformer_layer_infer_class = BloomTransformerLayerInfer

    # infer state class
    infer_state_class = InferStateInfo

    def __init__(self, tp_rank, world_size, weight_dir, max_total_token_num, load_way="HF", mode=[]):
        super().__init__(tp_rank, world_size, weight_dir, max_total_token_num, load_way, mode)
        return

    def _init_config(self):
        super()._init_config()
        # rename key
        # repair_config()
        return 
~~~

### (4) Add support for models in the server service layer

***lightllm/server/router/model_infer/model_rpc.py***

~~~python
import asyncio
import rpyc
import torch
import traceback
from datetime import timedelta
from typing import Dict, List, Tuple
from transformers.configuration_utils import PretrainedConfig
from lightllm.server.router.model_infer.infer_batch import InferBatch
from rpyc.utils.classic import obtain

from lightllm.models.bloom.model import BloomTpPartModel
from lightllm.utils.infer_utils import set_random_seed
from lightllm.utils.infer_utils import calculate_time, mark_start, mark_end
from lightllm.common.configs.config import setting
from .post_process import sample

class ModelRpcServer(rpyc.Service):

    def exposed_init_model(self, rank_id, world_size, weight_dir, max_total_token_num, load_way, mode):
        import torch
        import torch.distributed as dist
        if world_size != 1:
            trans_list = [obtain(e) for e in (rank_id, world_size, weight_dir, max_total_token_num, load_way, mode)]
            rank_id, world_size, weight_dir, max_total_token_num, load_way, mode = trans_list

        self.tp_rank = rank_id
        self.world_size = world_size
        self.load_way = load_way
        self.mode = mode
        self.cache = {}

        dist.init_process_group('nccl', init_method=f'tcp://127.0.0.1:{setting["nccl_port"]}', rank=rank_id, world_size=world_size)
        torch.cuda.set_device(rank_id)

        model_cfg, _ = PretrainedConfig.get_config_dict(
            weight_dir
        )
        try:
            self.model_type = model_cfg["model_type"]
            if self.model_type == "bloom":
                self.model = BloomTpPartModel(rank_id, world_size, weight_dir, max_total_token_num, load_way, mode)
                raise Exception(f"can not support {self.model_type} now")
        except Exception as e:
            print("#" * 16)
            print("load model error:", str(e), e, type(e))
            raise e
        
        set_random_seed(2147483647)
        return
    ...
~~~









