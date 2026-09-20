# 如何添加新的模型支持

## 1. 当前的推理架构介绍

在 ***lightllm/common/basemodel*** 目录下，是整个推理架构的基类实现

~~~shell
├── basemodel.py   # 模型框架类
├── infer_struct.py # 推理用的状态类
├── __init__.py
├── layer_infer # 推理层的基类实现
│   ├── base_layer_infer.py
│   ├── __init__.py
│   ├── post_layer_infer.py
│   ├── pre_layer_infer.py
│   ├── template # 推理层的模板实现，继承实现模板可以减少开发量和重复代码
│   │   ├── __init__.py
│   │   ├── post_layer_infer_template.py
│   │   ├── pre_layer_infer_template.py
│   │   └── transformer_layer_infer_template.py
│   └── transformer_layer_infer.py
├── layer_weights # 权重基类的实现
│   ├── base_layer_weight.py
│   ├── hf_load_utils.py
│   ├── __init__.py
│   ├── pre_and_post_layer_weight.py
│   └── transformer_layer_weight.py
└── triton_kernel # 一些公共使用的 triton kernel 算子
    ├── apply_penalty.py
    ├── destindex_copy_kv.py
    └── __init__.py
~~~

如上所示，目前模型推理架构主要由权重和推理两个部分组成。

### 权重

layer_weights 目录下是权重相关的代码，理论上对于一个新添加的模型需要继承实现 pre_and_post_layer_weight.py 和 transformer_layer_weight.py 中的 PreAndPostLayerWeight 和 TransformerLayerWeight 类来实现权重的加载。

| 权重基类               | 职责                                                         |
| ---------------------- | ------------------------------------------------------------ |
| PreAndPostLayerWeight  | 负责对LLM模型的第一层Embedding层和最后一层后处理层的权重加载并按照所使用的tp参数对权重进行拆分 |
| TransformerLayerWeight | 负责对LLM模型transformer层进行权重的加载按照所使用的tp参数对权重进行拆分 |

### 推理

layer_infer 目录下是进行推理处理的相关基类，并在template目录下提供了一些模板，从模板类进行继承实现可以减少一些不必要的重复代码，简化实现，该目录下需要继承实现的推理类有三个。

| 推理基类              | 职责                                       |
| --------------------- | ------------------------------------------ |
| PreLayerInfer         | 负责对 Embedding 层的推理                  |
| TransformerLayerInfer | 负责 transformer 层的推理                  |
| PostLayerInfer        | 负责将网络最后的隐层输出转化为logits的推理 |

上述三个类的基类 BaseLayerInfer 提供了两个最重要的对外服务函数接口，所有的推理行为都会由这两个接口进入。

| 接口                                                         | 职责                                           |
| ------------------------------------------------------------ | ---------------------------------------------- |
| def context_forward(self, input_ids, infer_state: InferStateInfo, layer_weight: BaseLayerWeight): | Batch进行第一次推理（在代码中又被叫做prefill） |
| def token_forward(self, input_ids, infer_state: InferStateInfo, layer_weight: BaseLayerWeight): | 单步decode阶段的推理                           |

### 算子

triton_kernel 目录下是一些使用 openai triton 实现的推理需要用到的算子。

### 状态类

infer_struct.py 中的 InferStateInfo 类是进行一次模型推理时，在层间传递一些重要信息的状态类，不同的模型可以继承实现该类，添加每个模型需要传递的独特状态信息， InferStateInfo 类提供了一个供继承的init_some_extra_state接口，用于传递额外独特信息的初始化。

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

### 模型框架类

basemodel.py 中的 TpPartBaseModel 类，是整个模型的入口，每个类型的模型都需要继承实现该类。该类通过类似搭积木的方式，使用推理类，权重类，状态类完成模型的加载，推理功能，其中有很多接口可以被继承实现，以完成每个模型类型自己独特的操作。

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

    def __init__(self, weight_dir, max_total_token_num, load_way="HF", mode=[]):
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

常用需要继承实现的接口

| 接口                         | 功能                                                         |
| ---------------------------- | ------------------------------------------------------------ |
| def _init_config(self)：     | 读取初始化模型的 config.json,  并进行一些 key 名的同名合法化操作 |
| def _verify_params(self)：   | 校验参数                                                     |
| def _init_mem_manager(self): | 初始化 token attention 使用的 mem manager 对象               |
| def _init_some_value(self):  | 初始化推理框架会使用的一些成员变量的值                       |
| def _init_custom(self):      | 一些模型自己的个性化初始化，比如 llama 初始化自己的Rotary值  |

## 2. 添加 bloom 模型的示例说明

具体实现在 ***lightllm/models/bloom*** 目录下，下面的代码片段请对应源码进行阅读，其中 triton_kernel 目录下为推理类使用的一些 kernel，下文中不做详细介绍，同时 bloom 模型因为不需要传递特殊状态信息使用默认的状态类即可。如想更深入的理解整个框架，可以进一步参考 llama 和 llama2 等模型的接入实现源码。

### （1） 添加实现权重类

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
            split_vob_size = vob_size // self.tp_world_size_
            self.wte_weight_ = self._cuda(weights["word_embeddings.weight"][split_vob_size *
                                                                 self.tp_rank_: split_vob_size * (self.tp_rank_ + 1), :])
            self.lm_head_weight_ = self.wte_weight_
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
        tp_head_num = head_num // self.tp_world_size_
        tmp_alibi = self._generate_alibi(head_num, dtype=torch.float32)
        assert head_num % self.tp_world_size_ == 0
        self.tp_alibi = tmp_alibi[self.tp_rank_ * tp_head_num: (self.tp_rank_ + 1) * tp_head_num].contiguous().cuda()
        return
    
    def load_hf_weights(self, weights):

        self._load_qkvo_weights(weights)
        self._load_ffn_weights(weights)
        return

    def _load_qkvo_weights(self, weights):
        if f"h.{self.layer_num_}.input_layernorm.weight" in weights:
            self.att_norm_weight_ = self._cuda(weights[f"h.{self.layer_num_}.input_layernorm.weight"])
        if f"h.{self.layer_num_}.input_layernorm.bias" in weights:
            self.att_norm_bias_ = self._cuda(weights[f"h.{self.layer_num_}.input_layernorm.bias"])

        if f"h.{self.layer_num_}.self_attention.query_key_value.weight" in weights:
            n_embed = self.network_config_["n_embed"]
            split_n_embed = n_embed // self.tp_world_size_
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
            split_n_embed = n_embed // self.tp_world_size_
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
            split_n_embed = n_embed // self.tp_world_size_
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
            split_n_embed = n_embed // self.tp_world_size_
            self.ffn_1_weight_ = weights[f"h.{self.layer_num_}.mlp.dense_h_to_4h.weight"]
            self.ffn_1_weight_ = self._cuda(self.ffn_1_weight_[split_n_embed * self.tp_rank_: split_n_embed *
                                                    (self.tp_rank_ + 1), :].transpose(0, 1))

        if f"h.{self.layer_num_}.mlp.dense_h_to_4h.bias" in weights:
            n_embed = self.network_config_["n_embed"] * 4
            split_n_embed = n_embed // self.tp_world_size_
            self.ffn_1_bias_ = self._cuda(weights[f"h.{self.layer_num_}.mlp.dense_h_to_4h.bias"][split_n_embed *
                                                                                      self.tp_rank_: split_n_embed * (self.tp_rank_ + 1)])
        if f"h.{self.layer_num_}.mlp.dense_4h_to_h.weight" in weights:
            n_embed = self.network_config_["n_embed"] * 4
            split_n_embed = n_embed // self.tp_world_size_
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

### (2) 添加实现推理类

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
        tp_vocab_size_ = network_config["vocab_size"] // self.tp_world_size_
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
        if self.tp_world_size_ > 1:
            dist.all_reduce(input_embdings, op=dist.ReduceOp.SUM, async_op=False)
        input_embdings = self._norm(input_embdings, infer_state, layer_weight)
        return input_embdings

    def token_forward(self, input_ids, infer_state: InferStateInfo, layer_weight: BloomPreAndPostLayerWeight):
        input_mask = torch.logical_or(self.vob_start_id_ > input_ids, input_ids >= self.vob_end_id_)
        tmp_input_ids = (input_ids - self.vob_start_id_)
        tmp_input_ids[input_mask] = 0
        input_embdings = torch.embedding(layer_weight.wte_weight_, tmp_input_ids, padding_idx=-1)
        input_embdings[input_mask] = 0.0
        if self.tp_world_size_ > 1:
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
        self.tp_q_head_num_ = network_config["num_attention_heads"] // self.tp_world_size_
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
                        beta=1.0 / self.tp_world_size_)
        return o
    
    def _ffn(self, input, infer_state:InferStateInfo, layer_weight: BloomTransformerLayerWeight)->torch.Tensor:
        ffn1_out = torch.addmm(layer_weight.ffn_1_bias_, input.view(-1, self.embed_dim_), layer_weight.ffn_1_weight_)
        input = None
        gelu_out = torch.nn.functional.gelu(ffn1_out, approximate='tanh')
        ffn1_out = None
        ffn2_out = torch.addmm(layer_weight.ffn_2_bias_, gelu_out, layer_weight.ffn_2_weight_, beta=1.0 / self.tp_world_size_)
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
        assert (network_config["vocab_size"] % self.tp_world_size_ == 0)
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
        if self.tp_world_size_ == 1:
            gather_data = logic_batch
        else:
            gather_data = torch.empty((self.vocab_size_, batch_size), device=logic_batch.device, dtype=torch.float16)
            split_size = self.vocab_size_ // self.tp_world_size_
            dist.all_gather([gather_data[i * split_size: (i + 1) * split_size, :]
                            for i in range(self.tp_world_size_)], logic_batch, group=None, async_op=False)
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

### （3） 实现模型的框架类

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

### (4) 注册模型

在 `lightllm/models/builtin.py` 中添加内置注册，使用 `module:Class` 路径，避免导入模型实现：

```python
ModelRegistry.register("bloom", "lightllm.models.bloom.model:BloomTpPartModel")
```

这里的 `model_type` 来自 checkpoint 的 `config.json`。同一个实现可注册多个类型；多模态模型使用 `is_multimodal=True`。当一个类型有条件变体时，在此处添加 `condition`，例如 reward 模型使用 `is_reward_model()`，嵌套语言模型使用 `llm_model_type_is(...)`。命中的条件变体优先于默认实现；多个特例条件同时命中会报错。

兜底实现也可以限定适用范围：同时设置 `condition` 和 `is_fallback=True`。例如 Llava 兜底排除 Tarsier 架构，但仍允许外部条件变体覆盖它。用真实 checkpoint 配置核对家族标识与嵌套字段；只有文本模型类型相同，不足以判断是同一种多模态架构。

`ModeBackend.init_model` 读取配置后调用 `get_model`，类查询使用 `get_model_class`，两者共享同一选择逻辑。选择成功后才导入实现。不需要在 router 中增加分支，也不需要在 `models/__init__.py` 或内置模型类上重复添加 import/decorator。

Draft 模型单独注册 checkpoint 类型和投机模式，例如：

```python
DraftModelRegistry.register(
    "qwen3", "dspark", "lightllm.models.qwen3_dspark.model:Qwen3DSparkModel"
)
```

注册成功仅表示可以选择模型类；视觉/音频 encoder、tokenizer 和相关配置适配仍需按模型需求实现。现有 `@ModelRegistry(...)` 和 `@DraftModelRegistry(...)` 接口保留给显式导入的外部扩展类。

使用有代表性的 checkpoint 配置，测试模型选择、条件优先级和实际类加载。运行：

```bash
exp -m "验证模型注册与兼容性" python -m pytest unit_tests/models/test_registry.py -q
```

### (5) 读取模型配置

统一调用 `lightllm.utils.model_config.load_model_config(model_dir, trust_remote_code=...)`，返回具体的 Transformers 配置类，例如 `LlavaConfig`、`Qwen3Config`。默认值、字段校验和标准组件转换由该类负责；不要重新实现 JSON 与 HF 字典的合并规则。

```python
from lightllm.utils.model_config import load_model_config, get_text_config, to_model_config_dict

hf_config = load_model_config(model_dir, trust_remote_code=args.trust_remote_code)
text_config = get_text_config(hf_config)
hidden_size = text_config.hidden_size
runtime_config = to_model_config_dict(text_config)  # 旧推理引擎需要字典时才转换
```

每次加载拥有独立的对象树，`get_text_config` 返回该树中的文本组件。`to_model_config_dict` 创建独立字典，并转换通用维度别名、旧引擎 RoPE 字段及周期 attention 布局。finetune、MTP、draft 等局部修改只作用于运行时字典。语言模型的 `hf_config` 保留加载对象；`config` 延续旧引擎的字典协议。

`load_model_config_dict` 和 `get_config_json` 是旧 registry、tokenizer、视觉/音频包装器的兼容边界；在原 JSON 明确提供身份时，它们保留原 `model_type` 和 `architectures`，其他值采用 Config 解释结果。`read_model_config` 仅供源元数据查询和确需原始字段优先级的 draft 覆盖路径使用；所有文件/版本解析仍交给 HF。

Transformers 5.8 已支持的家族使用其原生类；旧 Qwen、InternLM、MiniCPM、InternVL、Tarsier 等兼容定义位于 `lightllm/utils/hf_configs/`。新增兼容定义需给出来源和实际配置样例，未知家族不会静默退回裸字典。显式允许远程代码且 checkpoint 提供 `auto_map.AutoConfig` 时使用该自定义类；启动阶段必须传入 CLI 的 `trust_remote_code`，不能依赖尚未写入的启动环境。

配置对象加载不导入 LightLLM 模型实现，但 HF 自身可能导入 torch/Triton 辅助模块。相关无权重测试位于 `unit_tests/utils/test_model_config.py`、`unit_tests/models/test_config_init.py`，使用精简的内联配置覆盖关键回归。配置测试不等于推理精度或性能验证。
