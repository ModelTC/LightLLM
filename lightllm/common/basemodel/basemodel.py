import os

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import gc
import copy
import json
import torch
import torch.nn.functional as F
from typing import final, List
from tqdm import tqdm

from lightllm.common.basemodel.layer_weights.hf_load_utils import load_hf_weights
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.common.kv_cache_mem_manager import MemoryManager
from lightllm.common.kv_cache_mem_manager.mem_utils import select_mem_manager_class
from lightllm.common.req_manager import ReqManager
from lightllm.common.infer_utils import init_req_to_token_indexes
from lightllm.common.build_utils import repair_config
from lightllm.common.basemodel.triton_kernel.copy_kv_index_to_req import copy_kv_index_to_req
from lightllm.common.basemodel.layer_infer.cache_tensor_manager import g_cache_manager
from lightllm.common.basemodel.cuda_graph import CudaGraph
from lightllm.common.basemodel.prefill_cuda_graph import PrefillCudaGraph
from lightllm.common.quantization import Quantcfg
from lightllm.common.basemodel.triton_kernel.gather_token_id import gather_token
from lightllm.utils.log_utils import init_logger
from lightllm.utils.dist_utils import get_dp_world_size
from lightllm.utils.envs_utils import get_env_start_args, get_llm_data_type, get_added_mtp_kv_layer_num
from lightllm.distributed.communication_op import dist_group_manager
from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.common.triton_utils.autotuner import AutotuneLevel
from lightllm.utils.custom_kernel_utis import pad2dim_tensor_to_new_batch
from lightllm.utils.envs_utils import set_model_init_status, enable_diverse_mode_gqa_decode_fast_kernel
from lightllm.common.triton_utils.autotuner import Autotuner
from lightllm.utils.infer_utils import post_empty_cache
from .attention import get_prefill_att_backend_class, get_decode_att_backend_class
from .attention import BaseAttBackend

logger = init_logger(__name__)

torch.backends.cudnn.enabled = True


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

    def __init__(self, kvargs):
        self.args = get_env_start_args()
        self.run_mode = kvargs["run_mode"]
        self.weight_dir_ = kvargs["weight_dir"]
        self.max_total_token_num = kvargs["max_total_token_num"]
        self.batch_max_tokens = kvargs.get("batch_max_tokens", None)
        self.load_way = kvargs.get("load_way", "HF")
        self.weight_dict = kvargs.get("weight_dict", None)
        self.finetune_config = kvargs.get("finetune_config", None)
        self.max_req_num = kvargs.get("max_req_num", 1000)
        self.max_seq_length = kvargs.get("max_seq_length", 1024 * 5)
        # 用于等待外围的一些模块的初始化完成（如 CPU KV Cache 注册完成）
        self.wait_events = kvargs.get("wait_events", [])
        # is_token_healing 和 return_all_prompt_logics 是有排斥关系的两个模式，只能单独有一个生效
        # 主要是在prefill阶段返回多少个token的用于后续处理相关。
        self.is_token_healing = kvargs.get("is_token_healing", False)
        self.return_all_prompt_logics = kvargs.get("return_all_prompt_logics", False)
        assert not (self.is_token_healing and self.return_all_prompt_logics), "can not be true in same time"
        self.data_type = get_llm_data_type()
        mtp_step = get_env_start_args().mtp_step
        self.graph_max_batch_size = kvargs.get("graph_max_batch_size", 16)
        self.graph_max_batch_size = (
            self.graph_max_batch_size // 2
            if get_env_start_args().enable_decode_microbatch_overlap
            else self.graph_max_batch_size
        )
        # mtp 模式下需要修缮对应的最大batch size，为 （mtp_step + 1) 的倍数
        self.graph_max_batch_size = self.graph_max_batch_size * (mtp_step + 1)

        self.graph_max_len_in_batch = kvargs.get("graph_max_len_in_batch", 8192)
        self.disable_cudagraph = kvargs.get("disable_cudagraph", False)
        self.quant_type = kvargs.get("quant_type", "none")
        self.quant_cfg_path = kvargs.get("quant_cfg", None)
        self.mem_fraction = kvargs.get("mem_fraction", 0.9)
        self.tp_world_size_ = get_dp_world_size()
        self.enable_tpsp_mix_mode = get_env_start_args().enable_tpsp_mix_mode

        self.is_mtp_mode = self.args.mtp_mode in [
            "vanilla_with_att",
            "eagle_with_att",
            "vanilla_no_att",
            "eagle_no_att",
        ]
        self.prefill_graph: PrefillCudaGraph = None

        self._init_config()
        self._verify_must()
        self._verify_params()
        self._init_quant()

        # 更连续的显存分配可以有更好的性能
        if self.max_total_token_num is None:
            self._init_weights()
            self._init_mem_manager()
        else:
            self._init_mem_manager()
            self._init_weights()

        self._init_kv_move_buffer()
        self._check_mem_size()
        self._init_req_manager()
        self._init_infer_layer()
        self._init_some_value()
        self._init_custom()
        # wait必须在init cudagraph 之前，避免错误捕获
        self._wait_other_modules_ready()

        self._init_att_backend()
        self._init_att_backend1()

        logger.info(f"use prefill att backend: {self.prefill_att_backend.__class__.__name__}")
        logger.info(f"use decode att backend: {self.decode_att_backend.__class__.__name__}")
        if self.prefill_att_backend1 is not None:
            logger.info(f"use prefill att backend1: {self.prefill_att_backend1.__class__.__name__}")
            logger.info(f"use decode att backend1: {self.decode_att_backend1.__class__.__name__}")

        self._autotune_warmup()
        self._init_padded_req()
        self._init_cudagraph()
        self._init_prefill_cuda_graph()
        self._check_max_len_infer()
        torch.cuda.empty_cache()
        set_model_init_status(True)
        return

    def _wait_other_modules_ready(self):
        for event in self.wait_events:
            event.wait()
        return

    def _init_config(self):
        with open(os.path.join(self.weight_dir_, "config.json"), "r") as json_file:
            self.config = json.load(json_file)
        # rename keys
        repair_config(self.config, same_names=["num_attention_heads", "n_head"])
        repair_config(self.config, same_names=["hidden_size", "n_embd", "n_embed"])
        repair_config(self.config, same_names=["num_hidden_layers", "n_layer"])
        if self.finetune_config:
            self.config["vocab_size"] = self.finetune_config.vocab_size
        return

    @final
    def _verify_must(self):
        assert self.config["num_attention_heads"] % self.tp_world_size_ == 0
        return

    def _verify_params(self):
        assert self.load_way == "HF", "only support HF format weights"
        assert self.config["num_key_value_heads"] % self.tp_world_size_ == 0
        return

    def _init_quant(self):
        self.quant_cfg = Quantcfg(self.config, self.quant_type, self.quant_cfg_path)
        logger.info(f"Initial quantization. " f"The default quantization method is {self.quant_cfg.quant_type}")

    def _init_weights(self, start_layer_index=0):
        self.pre_post_weight = self.pre_and_post_weight_class(self.data_type, network_config=self.config)
        self.trans_layers_weight = [
            self.transformer_weight_class(
                i,
                self.data_type,
                network_config=self.config,
                quant_cfg=self.quant_cfg,
            )
            for i in range(start_layer_index, start_layer_index + self.config["n_layer"])
        ]
        load_hf_weights(
            self.data_type,
            weight_dir=self.weight_dir_,
            pre_post_layer=self.pre_post_weight,
            transformer_layer_list=self.trans_layers_weight,
            weight_dict=self.weight_dict,
        )
        self.pre_post_weight.verify_load()
        [weight.verify_load() for weight in self.trans_layers_weight]
        return

    def _init_mem_manager(self):
        assert self.config["num_attention_heads"] % self.tp_world_size_ == 0
        self.mem_manager: MemoryManager = select_mem_manager_class()(
            self.max_total_token_num,
            dtype=self.data_type,
            head_num=self.config["num_attention_heads"] // self.tp_world_size_,
            head_dim=self.config["n_embed"] // self.config["num_attention_heads"],
            layer_num=self.config["n_layer"] + get_added_mtp_kv_layer_num(),
            mem_fraction=self.mem_fraction,
        )
        return

    def _init_kv_move_buffer(self):
        # p d 分离的推理模式下才需要做这一步初始化
        if self.run_mode in ["prefill", "decode"]:
            self.mem_manager.alloc_kv_move_buffer(self.mem_manager.size)

    def _check_mem_size(self):
        self.max_total_token_num = self.mem_manager.size
        assert self.max_seq_length <= self.max_total_token_num
        return

    def _init_req_manager(self):
        create_max_seq_len = 0

        if self.batch_max_tokens is not None:
            create_max_seq_len = max(create_max_seq_len, self.batch_max_tokens)
        if self.max_seq_length is not None:
            create_max_seq_len = max(create_max_seq_len, self.max_seq_length)

        self.req_manager = ReqManager(self.max_req_num, create_max_seq_len, self.mem_manager)
        return

    def _init_infer_layer(self, start_layer_index=0):
        self.pre_infer = self.pre_layer_infer_class(network_config=self.config)
        self.post_infer = self.post_layer_infer_class(network_config=self.config)
        self.layers_infer = [
            self.transformer_layer_infer_class(i, network_config=self.config)
            for i in range(start_layer_index, start_layer_index + self.config["n_layer"])
        ]
        return

    def _init_some_value(self):
        # Dealing with head_dim_!=n_embed // num_attention_heads scenarios, such as mistral 13B
        head_dim_ = self.config["n_embed"] // self.config["num_attention_heads"]
        self.head_dim_ = self.config.get("head_dim", head_dim_)
        self.tp_k_head_num_ = self.config["num_key_value_heads"] // self.tp_world_size_
        self.tp_v_head_num_ = self.tp_k_head_num_
        self.layers_num = self.config["n_layer"]
        self.vocab_size = self.config["vocab_size"]
        return

    def _init_att_backend(self):
        self.prefill_att_backend: BaseAttBackend = get_prefill_att_backend_class(index=0)(model=self)
        self.decode_att_backend: BaseAttBackend = get_decode_att_backend_class(index=0)(model=self)
        return

    def _init_att_backend1(self):
        # self.prefill_att_backend1 是给后续有模型支持不同层用不同的att模块时，保留的扩展。
        self.prefill_att_backend1: BaseAttBackend = None
        self.decode_att_backend1: BaseAttBackend = None
        return

    def _init_cudagraph(self):
        self.graph = (
            None if self.disable_cudagraph else CudaGraph(self.graph_max_batch_size, self.graph_max_len_in_batch)
        )
        if self.graph is not None:
            if get_env_start_args().enable_decode_microbatch_overlap:
                self.graph.warmup_overlap(self)
            else:
                self.graph.warmup(self)

    def _init_prefill_cuda_graph(self):
        self.prefill_graph = (
            None
            if not get_env_start_args().enable_prefill_cudagraph
            else PrefillCudaGraph(decode_cuda_graph=self.graph)
        )
        if self.prefill_graph is not None:
            if get_env_start_args().enable_prefill_microbatch_overlap:
                self.prefill_graph.warmup_overlap(self)
            else:
                self.prefill_graph.warmup(self)

    def _init_custom(self):
        pass

    @torch.no_grad()
    def forward(self, model_input: ModelInput):
        model_input.to_cuda()
        assert model_input.mem_indexes.is_cuda

        if model_input.is_prefill:
            return self._prefill(model_input)
        else:
            return self._decode(model_input)

    def _create_inferstate(self, model_input: ModelInput, microbatch_index: int = 0):
        infer_state = self.infer_state_class()
        infer_state.input_ids = model_input.input_ids
        infer_state.is_prefill = model_input.is_prefill
        infer_state.is_token_healing = self.is_token_healing
        infer_state.return_all_prompt_logics = self.return_all_prompt_logics
        infer_state.batch_size = model_input.batch_size
        infer_state.total_token_num = model_input.total_token_num
        infer_state.max_q_seq_len = model_input.max_q_seq_len
        infer_state.max_kv_seq_len = model_input.max_kv_seq_len
        infer_state.max_cache_len = model_input.max_cache_len
        infer_state.prefix_total_token_num = model_input.prefix_total_token_num
        assert model_input.b_req_idx.shape[0] == model_input.b_seq_len.shape[0]
        infer_state.b_req_idx = model_input.b_req_idx
        infer_state.b_seq_len = model_input.b_seq_len
        if model_input.is_prefill:
            if model_input.b_ready_cache_len is not None:
                infer_state.b_ready_cache_len = model_input.b_ready_cache_len
            else:
                infer_state.b_ready_cache_len = torch.zeros_like(input=infer_state.b_seq_len)
        else:
            if enable_diverse_mode_gqa_decode_fast_kernel():
                infer_state.b_shared_seq_len = model_input.b_shared_seq_len
                infer_state.b_mark_shared_group = model_input.b_mark_shared_group

        infer_state.multimodal_params = model_input.multimodal_params

        infer_state.mem_manager = self.mem_manager
        infer_state.req_manager = self.req_manager

        infer_state.mem_index = model_input.mem_indexes
        infer_state.microbatch_index = microbatch_index
        infer_state.dist_group = dist_group_manager.get_group(microbatch_index)

        # 特殊模型，特殊模式的特定变量初始化操作。
        infer_state.mtp_draft_input_hiddens = model_input.mtp_draft_input_hiddens

        if infer_state.is_prefill:
            infer_state.prefill_att_state = self.prefill_att_backend.create_att_prefill_state(infer_state=infer_state)
            if self.prefill_att_backend1 is not None:
                infer_state.prefill_att_state1 = self.prefill_att_backend1.create_att_prefill_state(
                    infer_state=infer_state
                )
        else:
            infer_state.decode_att_state = self.decode_att_backend.create_att_decode_state(infer_state=infer_state)
            if self.decode_att_backend1 is not None:
                infer_state.decode_att_state1 = self.decode_att_backend1.create_att_decode_state(
                    infer_state=infer_state
                )

        return infer_state

    def _create_padded_decode_model_input(self, model_input: ModelInput, new_batch_size: int):
        if model_input.batch_size == new_batch_size:
            return model_input

        assert model_input.batch_size <= new_batch_size

        padded_batch_size = new_batch_size - model_input.batch_size
        new_model_input = copy.copy(model_input)
        new_model_input.batch_size = new_batch_size
        new_model_input.total_token_num += padded_batch_size * 2
        new_model_input.max_kv_seq_len = max(2, model_input.max_kv_seq_len)
        new_model_input.input_ids = F.pad(new_model_input.input_ids, (0, padded_batch_size), mode="constant", value=1)
        new_model_input.b_req_idx = F.pad(
            new_model_input.b_req_idx, (0, padded_batch_size), mode="constant", value=self.req_manager.HOLD_REQUEST_ID
        )
        new_model_input.b_seq_len = F.pad(new_model_input.b_seq_len, (0, padded_batch_size), mode="constant", value=2)
        new_model_input.mem_indexes = F.pad(
            new_model_input.mem_indexes,
            (0, padded_batch_size),
            mode="constant",
            value=self.mem_manager.HOLD_TOKEN_MEMINDEX,
        )
        new_model_input.multimodal_params = new_model_input.multimodal_params + [
            {"images": [], "audios": []} for _ in range(padded_batch_size)
        ]

        if enable_diverse_mode_gqa_decode_fast_kernel():
            if new_model_input.b_shared_seq_len is not None:
                new_model_input.b_shared_seq_len = F.pad(
                    new_model_input.b_shared_seq_len, (0, padded_batch_size), mode="constant", value=0
                )
            if new_model_input.b_mark_shared_group is not None:
                new_model_input.b_mark_shared_group = F.pad(
                    new_model_input.b_mark_shared_group, (0, padded_batch_size), mode="constant", value=1
                )

        # 特殊模型，特殊模式的特殊变量的特殊 padding
        if new_model_input.mtp_draft_input_hiddens is not None:
            new_model_input.mtp_draft_input_hiddens = pad2dim_tensor_to_new_batch(
                input=new_model_input.mtp_draft_input_hiddens,
                new_batch_size=new_batch_size,
            )

        new_model_input.check_input()
        return new_model_input

    def _create_padded_prefill_model_input(self, model_input: ModelInput, new_handle_token_num: int):
        assert model_input.total_token_num - model_input.prefix_total_token_num < new_handle_token_num

        padded_token_num = new_handle_token_num - (model_input.total_token_num - model_input.prefix_total_token_num)
        assert padded_token_num > 0
        new_model_input = copy.copy(model_input)
        new_model_input.batch_size = model_input.batch_size + 1
        new_model_input.total_token_num += padded_token_num
        new_model_input.max_q_seq_len = max(padded_token_num, model_input.max_q_seq_len)
        new_model_input.max_kv_seq_len = max(padded_token_num, model_input.max_kv_seq_len)
        new_model_input.max_cache_len = max(0, model_input.max_cache_len)
        new_model_input.input_ids = F.pad(new_model_input.input_ids, (0, padded_token_num), mode="constant", value=1)
        new_model_input.mem_indexes = F.pad(
            new_model_input.mem_indexes,
            (0, padded_token_num),
            mode="constant",
            value=self.mem_manager.HOLD_TOKEN_MEMINDEX,
        )
        new_model_input.b_req_idx = F.pad(
            new_model_input.b_req_idx, (0, 1), mode="constant", value=self.req_manager.HOLD_REQUEST_ID
        )
        new_model_input.b_mtp_index = F.pad(new_model_input.b_mtp_index, (0, 1), mode="constant", value=0)
        new_model_input.b_seq_len = F.pad(new_model_input.b_seq_len, (0, 1), mode="constant", value=padded_token_num)
        new_model_input.b_ready_cache_len = F.pad(new_model_input.b_ready_cache_len, (0, 1), mode="constant", value=0)
        b_q_seq_len = new_model_input.b_seq_len - new_model_input.b_ready_cache_len
        new_model_input.b_prefill_start_loc = b_q_seq_len.cumsum(dim=0, dtype=torch.int32) - b_q_seq_len
        # 构建新的list, 使用 append 可能会让外面使用的数组引用发生变化，导致错误。
        new_model_input.b_prefill_has_output_cpu = [e for e in new_model_input.b_prefill_has_output_cpu] + [False]
        new_model_input.prefix_total_token_num = model_input.prefix_total_token_num

        new_model_input.multimodal_params = [e for e in new_model_input.multimodal_params] + [
            {"images": [], "audios": []}
        ]

        # 特殊模型，特殊模式的特殊变量的特殊 padding
        if new_model_input.mtp_draft_input_hiddens is not None:
            new_model_input.mtp_draft_input_hiddens = pad2dim_tensor_to_new_batch(
                input=new_model_input.mtp_draft_input_hiddens,
                new_batch_size=new_handle_token_num,
            )

        new_model_input.check_input()
        return new_model_input

    def _create_unpad_decode_model_output(self, model_output: ModelOutput, origin_batch_size: int):
        padded_batch_size = model_output.logits.shape[0]
        if padded_batch_size == origin_batch_size:
            return model_output
        new_model_output = copy.copy(model_output)
        new_model_output.logits = new_model_output.logits[0:origin_batch_size]

        # 特殊模型，特殊模式的特殊变量的特殊 unpad
        if new_model_output.mtp_main_output_hiddens is not None:
            _hidden_states = new_model_output.mtp_main_output_hiddens
            new_model_output.mtp_main_output_hiddens = _hidden_states[0:origin_batch_size]

        return new_model_output

    def _create_unpad_prefill_model_output(self, padded_model_output: ModelOutput, origin_handle_token_num: int):
        if self.return_all_prompt_logics:
            new_model_output = copy.copy(padded_model_output)
            new_model_output.logits = new_model_output.logits[0:origin_handle_token_num]
        else:
            new_model_output = copy.copy(padded_model_output)
            # 移除多余的pad 的那个 req 对应的 logics
            new_model_output.logits = new_model_output.logits[0:-1]

        # 特殊模型，特殊模式的特殊变量的特殊 unpad
        if new_model_output.mtp_main_output_hiddens is not None:
            _hidden_states = new_model_output.mtp_main_output_hiddens
            new_model_output.mtp_main_output_hiddens = _hidden_states[0:origin_handle_token_num]

        return new_model_output

    def _prefill(
        self,
        model_input: ModelInput,
    ):
        origin_handle_token_num = model_input.total_token_num - model_input.prefix_total_token_num

        is_padded_model_input = False
        if self.prefill_graph is not None and self.prefill_graph.can_run(handle_token_num=origin_handle_token_num):
            finded_handle_token_num = self.prefill_graph.find_closest_graph_handle_token_num(
                handle_token_num=origin_handle_token_num
            )
            if finded_handle_token_num != origin_handle_token_num:
                is_padded_model_input = True
                model_input = self._create_padded_prefill_model_input(
                    model_input=model_input, new_handle_token_num=finded_handle_token_num
                )

        infer_state = self._create_inferstate(model_input)
        init_req_to_token_indexes(
            req_to_token_indexs=self.req_manager.req_to_token_indexs,
            b_req_idx=infer_state.b_req_idx,
            b_seq_len=infer_state.b_seq_len,
            b_ready_cache_len=infer_state.b_ready_cache_len,
            b_start_loc=model_input.b_prefill_start_loc,
            alloc_mem_index=infer_state.mem_index,
            max_q_seq_len=infer_state.max_q_seq_len,
        )
        prefill_mem_indexes_ready_event = torch.cuda.Event()
        prefill_mem_indexes_ready_event.record()

        infer_state.init_some_extra_state(self)
        infer_state.init_att_state()
        model_output = self._context_forward(infer_state)
        if is_padded_model_input:
            model_output = self._create_unpad_prefill_model_output(
                model_output, origin_handle_token_num=origin_handle_token_num
            )
        model_output.prefill_mem_indexes_ready_event = prefill_mem_indexes_ready_event
        return model_output

    def _decode(
        self,
        model_input: ModelInput,
    ) -> ModelOutput:
        # for overlap mode
        if model_input.input_ids is None:
            model_input.input_ids = gather_token(
                self.req_manager.req_sampling_params_manager.req_to_next_token_ids,
                model_input.b_req_idx,
                model_input.b_mtp_index,
            )

        if self.graph is not None and self.graph.can_run(model_input.batch_size, model_input.max_kv_seq_len):
            find_graph_batch_size = self.graph.find_closest_graph_batch_size(model_input.batch_size)
            padded_model_input = self._create_padded_decode_model_input(model_input, find_graph_batch_size)
            infer_state = self._create_inferstate(padded_model_input)
            copy_kv_index_to_req(
                self.req_manager.req_to_token_indexs,
                infer_state.b_req_idx,
                infer_state.b_seq_len,
                infer_state.mem_index,
            )
            infer_state.init_some_extra_state(self)
            infer_state.init_att_state()

            if self.graph.need_capture(find_graph_batch_size):
                infer_state.is_cuda_graph = True
                model_output: ModelOutput = self.graph.capture_decode(self._token_forward, infer_state)
            else:
                model_output: ModelOutput = self.graph.replay(infer_state)

            model_output = self._create_unpad_decode_model_output(
                model_output, origin_batch_size=model_input.batch_size
            )
        else:
            infer_state = self._create_inferstate(model_input)
            copy_kv_index_to_req(
                self.req_manager.req_to_token_indexs,
                infer_state.b_req_idx,
                infer_state.b_seq_len,
                infer_state.mem_index,
            )
            infer_state.init_some_extra_state(self)
            infer_state.init_att_state()
            model_output = self._token_forward(infer_state)

        return model_output

    @final
    def _context_forward(self, infer_state: InferStateInfo):
        run_mode_index = 1 if self.enable_tpsp_mix_mode else 0
        input_ids = infer_state.input_ids
        cuda_input_ids = input_ids

        pre_method = (self.pre_infer.context_forward, self.pre_infer.tpsp_context_forward)[run_mode_index]
        input_embs = pre_method(cuda_input_ids, infer_state, self.pre_post_weight)
        input_tensors = [input_embs]

        def prefill_func(input_tensors, infer_state):
            _input_embs = input_tensors[0]
            for i in range(self.layers_num):
                layer = self.layers_infer[i]
                layer_method = (layer.context_forward, layer.tpsp_context_forward)[run_mode_index]
                _input_embs = layer_method(_input_embs, infer_state, self.trans_layers_weight[i])
            return [_input_embs]

        handle_token_num = input_ids.shape[0]

        if self.prefill_graph is not None and self.prefill_graph.can_run(handle_token_num=handle_token_num):
            finded_handle_token_num = self.prefill_graph.find_closest_graph_handle_token_num(
                handle_token_num=handle_token_num
            )
            if self.prefill_graph.need_capture(handle_token_num=finded_handle_token_num):
                output_tensors: List[torch.Tensor] = self.prefill_graph.capture_prefill(
                    prefill_func=prefill_func,
                    input_tensors=input_tensors,
                    infer_state=infer_state,
                )
            else:
                output_tensors: List[torch.Tensor] = self.prefill_graph.replay(
                    input_tensors=input_tensors, infer_state=infer_state
                )

        else:
            g_cache_manager.cache_env_in()
            output_tensors: List[torch.Tensor] = prefill_func(input_tensors, infer_state)
            g_cache_manager.cache_env_out()

        input_embs = output_tensors[0]
        post_method = (self.post_infer.token_forward, self.post_infer.tpsp_token_forward)[run_mode_index]
        predict_logits = post_method(input_embs, infer_state, self.pre_post_weight)
        model_output = ModelOutput(logits=predict_logits)

        # 特殊模型特殊模式的额外输出
        if self.is_mtp_mode:
            model_output.mtp_main_output_hiddens = input_embs

        # 在开启使用deepep的时候，需要调用clear_deepep_buffer做资源清理，没有启用的时候
        # 该调用没有实际意义
        dist_group_manager.clear_deepep_buffer()
        return model_output

    @final
    def _token_forward(self, infer_state: InferStateInfo):
        run_mode_index = 1 if self.enable_tpsp_mix_mode else 0
        input_ids = infer_state.input_ids
        cuda_input_ids = input_ids
        pre_method = (self.pre_infer.token_forward, self.pre_infer.tpsp_token_forward)[run_mode_index]
        input_embs = pre_method(cuda_input_ids, infer_state, self.pre_post_weight)
        for i in range(self.layers_num):
            layer = self.layers_infer[i]
            layer_method = (layer.token_forward, layer.tpsp_token_forward)[run_mode_index]
            input_embs: torch.Tensor = layer_method(input_embs, infer_state, self.trans_layers_weight[i])

        post_method = (self.post_infer.token_forward, self.post_infer.tpsp_token_forward)[run_mode_index]
        predict_logits: torch.Tensor = post_method(input_embs, infer_state, self.pre_post_weight)

        if self.is_mtp_mode:
            graph_out_hiddens = input_embs.contiguous()

        model_output = ModelOutput(logits=predict_logits.contiguous())

        # 特殊模型特殊模式的额外输出
        if self.is_mtp_mode:
            model_output.mtp_main_output_hiddens = graph_out_hiddens

        # 在 cuda graph 模式下，输出需要转为 no ref tensor, 加强mem pool 的复用，降低显存的使用。
        if infer_state.is_cuda_graph:
            model_output.to_no_ref_tensor()

        return model_output

    @torch.no_grad()
    def microbatch_overlap_prefill(self, model_input0: ModelInput, model_input1: ModelInput):
        model_input0.to_cuda()
        model_input1.to_cuda()

        assert model_input0.mem_indexes.is_cuda
        assert model_input1.mem_indexes.is_cuda
        input_ids0, input_ids1 = model_input0.input_ids, model_input1.input_ids

        infer_state0 = self._create_inferstate(model_input0, 0)
        init_req_to_token_indexes(
            req_to_token_indexs=self.req_manager.req_to_token_indexs,
            b_req_idx=infer_state0.b_req_idx,
            b_seq_len=infer_state0.b_seq_len,
            b_ready_cache_len=infer_state0.b_ready_cache_len,
            b_start_loc=model_input0.b_prefill_start_loc,
            alloc_mem_index=infer_state0.mem_index,
            max_q_seq_len=infer_state0.max_q_seq_len,
        )
        infer_state0.init_some_extra_state(self)
        infer_state0.init_att_state()

        infer_state1 = self._create_inferstate(model_input1, 1)
        init_req_to_token_indexes(
            req_to_token_indexs=self.req_manager.req_to_token_indexs,
            b_req_idx=infer_state1.b_req_idx,
            b_seq_len=infer_state1.b_seq_len,
            b_ready_cache_len=infer_state1.b_ready_cache_len,
            b_start_loc=model_input1.b_prefill_start_loc,
            alloc_mem_index=infer_state1.mem_index,
            max_q_seq_len=infer_state1.max_q_seq_len,
        )
        infer_state1.init_some_extra_state(self)
        infer_state1.init_att_state()

        prefill_mem_indexes_ready_event = torch.cuda.Event()
        prefill_mem_indexes_ready_event.record()

        model_output0, model_output1 = self._overlap_tpsp_context_forward(
            input_ids0, infer_state0, input_ids1=input_ids1, infer_state1=infer_state1
        )

        # 在开启使用deepep的时候，需要调用clear_deepep_buffer做资源清理，没有启用的时候
        # 该调用没有实际意义
        dist_group_manager.clear_deepep_buffer()
        model_output0.prefill_mem_indexes_ready_event = prefill_mem_indexes_ready_event
        model_output1.prefill_mem_indexes_ready_event = prefill_mem_indexes_ready_event
        return model_output0, model_output1

    @torch.no_grad()
    def microbatch_overlap_decode(self, model_input0: ModelInput, model_input1: ModelInput):
        model_input0.to_cuda()
        model_input1.to_cuda()

        if model_input0.input_ids is None:
            model_input0.input_ids = gather_token(
                self.req_manager.req_sampling_params_manager.req_to_next_token_ids,
                model_input0.b_req_idx,
                model_input0.b_mtp_index,
            )
        if model_input1.input_ids is None:
            model_input1.input_ids = gather_token(
                self.req_manager.req_sampling_params_manager.req_to_next_token_ids,
                model_input1.b_req_idx,
                model_input1.b_mtp_index,
            )
        # TODO 动态 mtp fix
        assert model_input0.batch_size == model_input1.batch_size
        assert model_input0.mem_indexes.is_cuda
        assert model_input1.mem_indexes.is_cuda

        origin_batch_size = model_input0.batch_size
        max_len_in_batch = max(model_input0.max_kv_seq_len, model_input1.max_kv_seq_len)

        if self.graph is not None and self.graph.can_run(origin_batch_size, max_len_in_batch):
            find_graph_batch_size = self.graph.find_closest_graph_batch_size(origin_batch_size)
            # TODO 如果支持动态步数的 mtp，在不同的mtp步上，model_input0 和 model_input1 的内部batch size可能不
            # 一致，需要按照较高 batch size 进行graph的寻找，同时，进行有效的恢复。
            padded_model_input0 = self._create_padded_decode_model_input(model_input0, find_graph_batch_size)
            padded_model_input1 = self._create_padded_decode_model_input(model_input1, find_graph_batch_size)
            infer_state0 = self._create_inferstate(padded_model_input0, 0)
            copy_kv_index_to_req(
                self.req_manager.req_to_token_indexs,
                infer_state0.b_req_idx,
                infer_state0.b_seq_len,
                infer_state0.mem_index,
            )
            infer_state0.init_some_extra_state(self)
            infer_state0.init_att_state()

            infer_state1 = self._create_inferstate(padded_model_input1, 1)
            copy_kv_index_to_req(
                self.req_manager.req_to_token_indexs,
                infer_state1.b_req_idx,
                infer_state1.b_seq_len,
                infer_state1.mem_index,
            )
            infer_state1.init_some_extra_state(self)
            infer_state1.init_att_state()

            if self.graph.need_capture(find_graph_batch_size):
                infer_state0.is_cuda_graph = True
                infer_state1.is_cuda_graph = True

                model_output0, model_output1 = self.graph.capture_decode(
                    self._overlap_tpsp_token_forward,
                    infer_state0,
                    infer_state1=infer_state1,
                )
            else:
                model_output0, model_output1 = self.graph.replay(
                    infer_state0,
                    infer_state1=infer_state1,
                )

            # TODO 动态 mtp fix
            model_output0 = self._create_unpad_decode_model_output(model_output0, origin_batch_size=origin_batch_size)
            model_output1 = self._create_unpad_decode_model_output(model_output1, origin_batch_size=origin_batch_size)
        else:
            infer_state0 = self._create_inferstate(model_input0, 0)
            copy_kv_index_to_req(
                self.req_manager.req_to_token_indexs,
                infer_state0.b_req_idx,
                infer_state0.b_seq_len,
                infer_state0.mem_index,
            )
            infer_state0.init_some_extra_state(self)
            infer_state0.init_att_state()

            infer_state1 = self._create_inferstate(model_input1, 1)
            copy_kv_index_to_req(
                self.req_manager.req_to_token_indexs,
                infer_state1.b_req_idx,
                infer_state1.b_seq_len,
                infer_state1.mem_index,
            )
            infer_state1.init_some_extra_state(self)
            infer_state1.init_att_state()

            model_output0, model_output1 = self._overlap_tpsp_token_forward(infer_state0, infer_state1=infer_state1)
        return model_output0, model_output1

    @final
    def _overlap_tpsp_context_forward(self, infer_state: InferStateInfo, infer_state1: InferStateInfo):
        g_cache_manager.cache_env_in()
        input_embs, input_embs1 = self.pre_infer.overlap_tpsp_context_forward(
            infer_state.input_ids, infer_state1.input_ids, infer_state, infer_state1, self.pre_post_weight
        )
        for i in range(self.layers_num):
            input_embs, input_embs1 = self.layers_infer[i].overlap_tpsp_context_forward(
                input_embs, input_embs1, infer_state, infer_state1, self.trans_layers_weight[i]
            )
        predict_logits, predict_logits1 = self.post_infer.overlap_tpsp_token_forward(
            input_embs, input_embs1, infer_state, infer_state1, self.pre_post_weight
        )
        g_cache_manager.cache_env_out()

        model_output = ModelOutput(logits=predict_logits.contiguous())
        model_output1 = ModelOutput(logits=predict_logits1.contiguous())

        if self.is_mtp_mode:
            model_output.mtp_main_output_hiddens = input_embs.contiguous()
            model_output1.mtp_main_output_hiddens = input_embs1.contiguous()

        return model_output, model_output1

    @final
    def _overlap_tpsp_token_forward(self, infer_state: InferStateInfo, infer_state1: InferStateInfo):
        input_embs, input_embs1 = self.pre_infer.overlap_tpsp_token_forward(
            infer_state.input_ids, infer_state1.input_ids, infer_state, infer_state1, self.pre_post_weight
        )

        for i in range(self.layers_num):
            input_embs, input_embs1 = self.layers_infer[i].overlap_tpsp_token_forward(
                input_embs, input_embs1, infer_state, infer_state1, self.trans_layers_weight[i]
            )

        predict_logits, predict_logits1 = self.post_infer.overlap_tpsp_token_forward(
            input_embs, input_embs1, infer_state, infer_state1, self.pre_post_weight
        )

        if self.is_mtp_mode:
            graph_out_hiddens = input_embs.contiguous()
            graph_out_hiddens1 = input_embs1.contiguous()

        model_output = ModelOutput(logits=predict_logits.contiguous())
        model_output1 = ModelOutput(logits=predict_logits1.contiguous())

        if self.is_mtp_mode:
            model_output.mtp_main_output_hiddens = graph_out_hiddens
            model_output1.mtp_main_output_hiddens = graph_out_hiddens1

        if infer_state.is_cuda_graph:
            model_output.to_no_ref_tensor()
            model_output1.to_no_ref_tensor()

        return model_output, model_output1

    @final
    @torch.no_grad()
    def _check_max_len_infer(self):
        disable_check_max_len_infer = os.getenv("DISABLE_CHECK_MAX_LEN_INFER", None) is not None
        if disable_check_max_len_infer:
            logger.info("disable_check_max_len_infer is true")
            return

        # 做一次 同步
        torch.distributed.barrier()

        # 模拟最大长度进行 prefill，观察是否出现 OOM
        try:
            logger.info("begin check max_len infer")
            dummy_input_ids = torch.ones(self.batch_max_tokens, dtype=torch.int32, device="cuda")
            b_req_idx = torch.tensor([self.req_manager.alloc()], dtype=torch.int32, device="cuda")
            mem_indexes = self.mem_manager.alloc(len(dummy_input_ids)).cuda()
            b_seq_len = torch.ones(1, dtype=torch.int32, device="cuda")
            b_seq_len[:] = self.batch_max_tokens
            b_ready_cache_len = torch.zeros(1, dtype=torch.int32, device="cuda")
            b_prefill_start_loc = torch.zeros(1, dtype=torch.int32, device="cuda")
            total_token_num = self.batch_max_tokens
            b_mtp_index = torch.zeros(1, dtype=torch.int32, device="cuda")
            model_input = ModelInput(
                batch_size=1,
                total_token_num=total_token_num,
                max_q_seq_len=self.batch_max_tokens,
                max_kv_seq_len=self.batch_max_tokens,
                max_cache_len=0,
                prefix_total_token_num=0,
                input_ids=dummy_input_ids,
                mem_indexes=mem_indexes,
                b_req_idx=b_req_idx,
                b_seq_len=b_seq_len,
                b_mtp_index=b_mtp_index,
                is_prefill=True,
                b_ready_cache_len=b_ready_cache_len,
                b_prefill_start_loc=b_prefill_start_loc,
                multimodal_params=[{"images": [], "audios": []}],
            )
            model_output = self.forward(
                model_input,
            )
            prob_out = torch.softmax(model_output.logits, dim=-1)
            del model_output
            torch.argmax(prob_out, dim=1, keepdim=True)
            prob_out = None
            self.req_manager.free_all()
            self.mem_manager.free_all()
            logger.info(f"check max_len {self.batch_max_tokens} infer ok")
        except (RuntimeError, torch.OutOfMemoryError) as e:
            logger.exception(str(e))
            exception_str = (
                "check max len infer fail, you can try:"
                "1.Set the --mem_fraction or --max_total_token_num startup parameter to a smaller value."
                "2.Set the --max_req_total_len to a smaller value."
                "3.Set the --batch_max_tokens startup parameter to a smaller value."
            )
            logger.error(exception_str)
            raise Exception(exception_str)
        return

    def autotune_layers(self):
        # 控制autotune的层数，用于适配不同模型
        return self.config.get("first_k_dense_replace", 0) + 1

    @final
    @torch.no_grad()
    @post_empty_cache
    def _autotune_warmup(self):
        Autotuner.start_autotune_warmup()
        torch.distributed.barrier()

        warmup_lengths = [1, 8, 16, 32, 64, 100, 128, 256, 1024, 2048, 4096]

        if self.batch_max_tokens not in warmup_lengths:
            warmup_lengths.append(self.batch_max_tokens)

        warmup_lengths = [e for e in warmup_lengths if e <= self.batch_max_tokens]

        warmup_lengths.sort(reverse=True)

        layer_num_bak = self.layers_num
        self.layers_num = self.autotune_layers()
        for input_len in tqdm(warmup_lengths, desc="warming up"):
            try:
                rand_gen = torch.Generator(device="cuda")
                rand_gen.manual_seed(input_len)
                dummy_input_ids = torch.randint(
                    0, 10000, (input_len,), dtype=torch.int32, device="cuda", generator=rand_gen
                )
                b_req_idx = torch.tensor([self.req_manager.alloc()], dtype=torch.int32, device="cuda")
                mem_indexes = self.mem_manager.alloc(len(dummy_input_ids)).cuda()
                b_seq_len = torch.ones(1, dtype=torch.int32, device="cuda")
                b_seq_len[:] = input_len
                b_ready_cache_len = torch.zeros(1, dtype=torch.int32, device="cuda")
                b_prefill_start_loc = torch.zeros(1, dtype=torch.int32, device="cuda")
                total_token_num = input_len
                b_mtp_index = torch.zeros(1, dtype=torch.int32, device="cuda")
                model_input = ModelInput(
                    batch_size=1,
                    total_token_num=total_token_num,
                    max_q_seq_len=input_len,
                    max_kv_seq_len=input_len,
                    max_cache_len=0,
                    prefix_total_token_num=0,
                    input_ids=dummy_input_ids,
                    mem_indexes=mem_indexes,
                    b_req_idx=b_req_idx,
                    b_seq_len=b_seq_len,
                    b_mtp_index=b_mtp_index,
                    is_prefill=True,
                    b_ready_cache_len=b_ready_cache_len,
                    b_prefill_start_loc=b_prefill_start_loc,
                    multimodal_params=[{"images": [], "audios": []}],
                    **self._gen_special_model_input(total_token_num),
                )
                model_output = self.forward(
                    model_input,
                )
                del model_output
                self.req_manager.free_all()
                self.mem_manager.free_all()
                gc.collect()
                torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"autotune warmup for length {input_len} failed: {str(e)}")
                logger.exception(str(e))
                self.req_manager.free_all()
                self.mem_manager.free_all()
                gc.collect()
                torch.cuda.empty_cache()
        self.layers_num = layer_num_bak
        torch.distributed.barrier()
        Autotuner.end_autotune_warmup()

    @final
    @torch.no_grad()
    def _init_padded_req(self):
        """
        对 padded 所使用的req 进行初始化， 目前有非常多的地方需要使用，所以将其初始化固定为固定流程
        """
        # 做一次 同步
        torch.distributed.barrier()

        # prefill init padding req.
        prefill_input_len = 1
        batch_size = 1
        dummy_input_ids = torch.ones((batch_size,), dtype=torch.int32, device="cuda")
        b_req_idx = torch.tensor(
            [self.req_manager.HOLD_REQUEST_ID for _ in range(batch_size)], dtype=torch.int32, device="cuda"
        )
        mem_indexes = torch.tensor(
            [self.mem_manager.HOLD_TOKEN_MEMINDEX for _ in range(batch_size)], dtype=torch.int32, device="cuda"
        )
        b_seq_len = torch.ones(batch_size, dtype=torch.int32, device="cuda")
        b_ready_cache_len = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
        b_q_seq_len = b_seq_len - b_ready_cache_len
        b_prefill_start_loc = b_q_seq_len.cumsum(dim=0, dtype=torch.int32) - b_q_seq_len
        total_token_num = prefill_input_len * batch_size
        b_mtp_index = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
        model_input = ModelInput(
            batch_size=batch_size,
            total_token_num=total_token_num,
            max_q_seq_len=prefill_input_len,
            max_kv_seq_len=prefill_input_len,
            max_cache_len=0,
            prefix_total_token_num=0,
            input_ids=dummy_input_ids,
            mem_indexes=mem_indexes,
            b_req_idx=b_req_idx,
            b_mtp_index=b_mtp_index,
            b_seq_len=b_seq_len,
            b_ready_cache_len=b_ready_cache_len,
            b_prefill_start_loc=b_prefill_start_loc,
            is_prefill=True,
            multimodal_params=[{"images": [], "audios": []} for _ in range(batch_size)],
            **self._gen_special_model_input(total_token_num),
        )

        model_output: ModelOutput = self.forward(model_input)
        del model_input
        del dummy_input_ids
        del b_req_idx
        del mem_indexes
        del b_seq_len
        del b_ready_cache_len
        del model_output
        torch.cuda.empty_cache()
        return

    def _gen_special_model_input(self, token_num: int):
        special_model_input = {}

        is_mtp_draft_model = (
            "Deepseek3MTPModel" in str(self.__class__)
            or "Qwen3MOEMTPModel" in str(self.__class__)
            or "MistralMTPModel" in str(self.__class__)
        )
        if is_mtp_draft_model:
            special_model_input["mtp_draft_input_hiddens"] = torch.randn(
                token_num, self.config["hidden_size"], dtype=self.data_type, device="cuda"
            )
        else:
            special_model_input["mtp_draft_input_hiddens"] = None

        return special_model_input
