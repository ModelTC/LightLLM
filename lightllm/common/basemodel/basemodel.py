import os

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from contextlib import contextmanager
import math
import json
import torch
import triton
from typing import final, List, TYPE_CHECKING

from lightllm.common.basemodel.layer_weights.hf_load_utils import load_hf_weights
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.common.req_manager import ReqManager
from lightllm.common.attention_state import AttentionStateManager
from lightllm.common.basemodel.layer_infer.cache_tensor_manager import g_cache_manager
from lightllm.core.runner.decode_graph import CudaGraph
from lightllm.core.runner.prefill_graph import PrefillCudaGraph
from lightllm.core.runner.graph import GraphRunner
from lightllm.core.runner.state import BatchStateFactory, select_token_indices
from lightllm.core.runner.warmup import WarmupRunner
from lightllm.core.runner.batch import (
    BatchPolicy,
    BatchProgram,
    BatchRunner,
    pad_decode,
    pad_prefill,
    make_output,
    unpad_decode,
    unpad_prefill,
)
from lightllm.common.quantization import Quantcfg
from lightllm.utils.log_utils import init_logger
from lightllm.utils.dist_utils import get_dp_world_size, get_current_rank_in_dp
from lightllm.utils.envs_utils import (
    get_env_start_args,
    get_llm_data_type,
)
from lightllm.distributed.communication_op import dist_group_manager
from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput, PostLayerOutput
from lightllm.common.basemodel.hidden_collector import (
    NoopHiddenCollector,
)
from lightllm.common.basemodel.mtp_manager import MtpManager
from lightllm.common.triton_utils.autotuner import Autotuner
from lightllm.utils.torch_memory_saver_utils import (
    TorchMemorySaverWrapper,
    MemoryTag,
)
from .attention import get_prefill_att_backend_class, get_decode_att_backend_class
from .attention import BaseAttBackend

if TYPE_CHECKING:
    from easydict import EasyDict
    from lightllm.server.core.objs.start_args_type import StartArgs
    from lightllm.common.attention_state.full_att import FullAttentionState
    from lightllm.common.attention_state.bounded_att import BoundedAttentionState

logger = init_logger(__name__)

torch.backends.cudnn.enabled = True


class TpPartBaseModel:
    is_mtp_draft_model = False

    @property
    def graph_runner(self):
        if not hasattr(self, "_graph_runner"):
            self._graph_runner = GraphRunner()
        return self._graph_runner

    @property
    def batch_state_factory(self):
        if not hasattr(self, "_batch_state_factory"):
            prefill, decode = [], []
            for suffix in ("", "1"):
                backend = getattr(self, "prefill_att_backend" + suffix)
                if backend is not None:
                    prefill.append(("prefill_att_state" + suffix, backend.create_att_prefill_state))
                backend = getattr(self, "decode_att_backend" + suffix)
                if backend is not None:
                    decode.append(("decode_att_state" + suffix, backend.create_att_decode_state))
            select = self._select_mem_indexes
            if getattr(select, "__func__", None) is TpPartBaseModel._select_mem_indexes:
                select = None
            bind_inputs = getattr(self.pre_infer, "bind_inputs", None)
            prepare_inputs = (
                bind_inputs(self.pre_post_weight, self.multimodal_embedding_cache) if bind_inputs is not None else None
            )
            self._batch_state_factory = BatchStateFactory(
                state_type=self.infer_state_class,
                hidden_collector=self.hidden_collector_prototype,
                requests=self.req_manager,
                state_store=self.att_state_manager,
                role=self.att_state_role,
                is_draft=self.is_mtp_draft_model,
                return_all_prompt_logics=self.return_all_prompt_logics,
                group_for_lane=dist_group_manager.get_group,
                execution_runner=getattr(self, "execution_runner", None),
                prefill_states=tuple(prefill),
                decode_states=tuple(decode),
                select_indices=select,
                route_capture=getattr(self, "route_capture", None),
                prepare_inputs=prepare_inputs,
            )
        return self._batch_state_factory

    @property
    def batch_metadata(self):
        if not hasattr(self, "_batch_metadata"):
            self._batch_metadata = self.infer_state_class.bind_metadata(self)
        return self._batch_metadata

    @property
    def execution_torch_device(self):
        runtime = getattr(self, "execution_device", None)
        return runtime.device if runtime is not None else torch.device("cuda")

    @property
    def execution_runtime(self):
        from lightllm.core.device.cuda import CudaRuntime

        return getattr(self, "execution_device", None) or CudaRuntime

    @property
    def warmup_runner(self):
        if not hasattr(self, "_warmup_runner"):
            self._warmup_runner = WarmupRunner(
                self.req_manager,
                self.att_state_manager.full_att_allocator,
                self.forward,
                self._gen_special_model_input,
                self.vocab_size,
                overlap_prefill=self.microbatch_overlap_prefill,
                overlap_decode=self.microbatch_overlap_decode,
                runtime=self.execution_runtime,
            )
        return self._warmup_runner

    @property
    def batch_runner(self):
        if not hasattr(self, "_batch_runner"):
            policy = BatchPolicy(
                decode_alignment=self.mtp_manager.get_decode_batch_alignment(self.is_mtp_draft_model),
                sequence_parallel_alignment=self.tp_world_size_ if self.args.enable_tpsp_mix_mode else 1,
                mixed_prefill=self.args.enable_prefill_decode_mixed,
            )
            create_state = self._create_inferstate
            if getattr(create_state, "__func__", None) is TpPartBaseModel._create_inferstate:
                create_state = self.batch_state_factory.create
            initialize_state = self._initialize_batch_state
            if getattr(initialize_state, "__func__", None) is TpPartBaseModel._initialize_batch_state:
                initialize_state = self.batch_metadata
            program = BatchProgram(
                create_state=create_state,
                initialize_state=initialize_state,
                prefill=self._context_forward,
                decode=self._token_forward,
                overlap_prefill=self._overlap_tpsp_context_forward,
                overlap_decode=self._overlap_tpsp_token_forward,
            )
            padding = self._create_padded_decode_model_input
            if getattr(padding, "__func__", None) is TpPartBaseModel._create_padded_decode_model_input:
                padding = None
            self._batch_runner = BatchRunner(
                policy,
                program,
                self.req_manager,
                self.graph_runner,
                decode_padding=padding,
                clear_transfer=dist_group_manager.clear_deepep_buffer,
                runtime=self.execution_runtime,
            )
        return self._batch_runner

    def _initialize_batch_state(self, state):
        self.batch_metadata(state)

    @property
    def graph(self):
        return self.graph_runner.decode

    @graph.setter
    def graph(self, value):
        self.graph_runner.decode = value

    @property
    def prefill_graph(self):
        return self.graph_runner.prefill

    @prefill_graph.setter
    def prefill_graph(self, value):
        self.graph_runner.prefill = value

    # weight class
    pre_and_post_weight_class = None
    transformer_weight_class = None

    # infer class
    pre_layer_infer_class = None
    post_layer_infer_class = None
    transformer_layer_infer_class = None

    # infer state class
    infer_state_class = InferStateInfo
    is_hybrid_att_model = False

    def __init__(self, kvargs):
        self.args = get_env_start_args()
        self.run_mode = kvargs["run_mode"]
        self.weight_dir_ = kvargs["weight_dir"]
        self.max_total_token_num = kvargs["max_total_token_num"]
        self.batch_max_tokens = kvargs.get("batch_max_tokens", None)
        self.load_way = kvargs.get("load_way", "HF")
        self.weight_dict = kvargs.get("weight_dict", None)
        self.multimodal_embedding_cache = kvargs.get("multimodal_embedding_cache")
        self.execution_providers = kvargs.get("execution_providers")
        self.execution_device = kvargs.get("execution_device")
        self.execution_communication = kvargs.get("execution_communication", ())
        self.execution_formats = kvargs.get("execution_formats")
        self.finetune_config = kvargs.get("finetune_config", None)
        self.max_req_num = kvargs.get("max_req_num", 1000)
        self.max_seq_length = kvargs.get("max_seq_length", 1024 * 5)
        self.return_all_prompt_logics = kvargs.get("return_all_prompt_logics", False)
        self.data_type = get_llm_data_type()
        self.graph_max_batch_size = kvargs.get("graph_max_batch_size", 16)
        if self.args.enable_decode_microbatch_overlap:
            self.graph_max_batch_size //= 2
        self.mtp_manager = MtpManager.get_instance()
        self.graph_max_batch_size *= self.mtp_manager.get_decode_tokens_per_request(self.is_mtp_draft_model)

        self.graph_max_len_in_batch = kvargs.get("graph_max_len_in_batch", 8192)
        self.disable_cudagraph = kvargs.get("disable_cudagraph", False)
        self.quant_type = kvargs.get("quant_type", "none")
        self.quant_cfg_path = kvargs.get("quant_cfg", None)
        self.expert_dtype = kvargs.get("expert_dtype", None)
        self.mem_fraction = kvargs.get("mem_fraction", 0.9)
        self.tp_world_size_ = get_dp_world_size()
        self.tp_rank_ = get_current_rank_in_dp()
        self.enable_tpsp_mix_mode = get_env_start_args().enable_tpsp_mix_mode
        self.graph_max_batch_size = self._align_decode_batch_size(self.graph_max_batch_size)

        self.torch_memory_saver = TorchMemorySaverWrapper(self.args.enable_torch_memory_saver)
        self.prefill_graph: PrefillCudaGraph = None

        self._init_config()
        self._verify_must()
        self._verify_params()
        self._init_quant()
        self._init_model_adapter()

        enable_weight_cpu_backup = self.args.enable_weight_cpu_backup
        with self.torch_memory_saver.region(tag=MemoryTag.WEIGHT, enable_cpu_backup=enable_weight_cpu_backup):
            self._init_weights()
        self._init_infer_layer()
        for recipe in (self.pre_infer, self.post_infer, *self.layers_infer):
            recipe.execution_device = self.execution_device
        self._init_some_value()
        self._init_custom()
        self.load_weights(self.weight_dict)
        # The backend calls each model's runtime stages after all weights are loaded.
        self.graph = None
        return

    @property
    def att_state_role(self) -> str:
        """当前模型的资源前缀；每个 draft 独立拥有 State。"""
        from lightllm.common.attention_state.utils import MAX_DRAFT_MODELS

        if not self.is_mtp_draft_model:
            return "main"
        draft_index = len(self.mtp_previous_draft_models)
        assert draft_index < MAX_DRAFT_MODELS, "at most 36 draft models are supported"
        return f"draft{draft_index}"

    def get_req_manager_max_seq_len(self) -> int:
        """返回请求到 token 索引映射所需的长度容量，不表示模型支持的最大上下文长度。"""
        return max(0, self.batch_max_tokens or 0, self.max_seq_length or 0)

    def bind_req_manager(self, req_manager: ReqManager) -> None:
        """绑定启动流程创建的共享请求管理器；main/draft 共用请求槽位和 token 索引映射。"""
        self.req_manager = req_manager

    def prepare_attention_states(self) -> None:
        """main 创建共享管理器，draft 引用同一实例，再准备当前模型自己的 State。"""
        assert getattr(self, "att_state_manager", None) is None, "resources already initialized"
        if self.is_mtp_draft_model:
            self.att_state_manager = self.main_model.att_state_manager
            assert not self.att_state_manager._state_registration_finished, "State registration is already finished"
            assert (
                self.max_req_num + 1 == self.att_state_manager._request_capacity
            ), "models must share the request domain"
        with self.torch_memory_saver.region(tag=MemoryTag.KV_CACHE):
            if not self.is_mtp_draft_model:
                self.att_state_manager = AttentionStateManager(request_capacity=self.max_req_num + 1)
            # 无 attention 的 draft 只共享管理器，不创建自身 State。
            if self.is_mtp_draft_model and self.args.mtp_mode in ("vanilla_no_att", "eagle_no_att"):
                return

            from lightllm.models.base.build import prepare_runner

            self.execution_runner = prepare_runner(self)
            if self.execution_runner is not None:
                return

            # 当前模型拥有的全局逻辑层 ID，按模型内部层顺序排列；各工厂自行选择参与层及共享分组。
            logical_layer_ids = tuple(layer.layer_num_ for layer in self.layers_infer)
            full_states = self.create_full_att_states(
                model_config=self.config,
                args=self.args,
                model_logical_layer_ids=logical_layer_ids,
                tp_rank=self.tp_rank_,
                tp_world_size=self.tp_world_size_,
            )
            bounded_states = self.create_bounded_att_states(
                model_config=self.config,
                args=self.args,
                model_logical_layer_ids=logical_layer_ids,
                request_capacity=self.max_req_num + 1,
                tp_rank=self.tp_rank_,
                tp_world_size=self.tp_world_size_,
                device=self.execution_torch_device,
            )
            for state in (*full_states, *bounded_states):
                self.att_state_manager.register_state(self.att_state_role, state)

    def init_runtime(self) -> None:
        """Initialize this model's backends, warmup and graphs after shared allocation."""
        self.req_manager.bind_hold_page(self.att_state_manager.full_att_allocator.hold_token_indices)
        self._check_mem_size()
        if getattr(self, "execution_runner", None) is None:
            self._init_att_backend()
            self._init_att_backend1()
        else:
            self.prefill_att_backend = self.decode_att_backend = None
            self.prefill_att_backend1 = self.decode_att_backend1 = None
        self._init_execution_plan()
        self._init_hidden_collector()
        self._batch_metadata = self.infer_state_class.bind_metadata(self)
        role = self.att_state_role
        if self.execution_plan is None:
            logger.info(
                f"{role} prefill/decode attention: {type(self.prefill_att_backend).__name__} / "
                f"{type(self.decode_att_backend).__name__}"
            )
        if self.prefill_att_backend1 is not None:
            logger.info(
                f"{role} prefill/decode attention1: {type(self.prefill_att_backend1).__name__} / "
                f"{type(self.decode_att_backend1).__name__}"
            )

        self._autotune_warmup()
        self._init_padded_req()
        self._init_cudagraph()
        self._init_prefill_cuda_graph()
        if not self.is_mtp_draft_model:
            self._check_max_len_infer()

    def _init_execution_plan(self):
        if getattr(self, "execution_plan", None) is not None:
            raise RuntimeError("bound execution plans cannot be replaced while resources or Graphs may be in use")
        runner = getattr(self, "execution_runner", None)
        self.execution_plan = runner.bind(self.layers_infer) if runner is not None else None
        logger.info(f"{self.att_state_role} attention execution: {'bound' if self.execution_plan else 'legacy'}")
        if self.execution_plan is not None:
            backends = sorted({item.backend for item in self.execution_plan.bindings})
            logger.info(f"Bound attention: {backends}; workspace: {self.execution_plan.workspace_bytes} bytes")

    @classmethod
    def normalize_model_config(cls, model_config: dict[str, object]) -> dict[str, object]:
        """返回当前模型的有效配置，供模型初始化和 CPU cache 共用。

        不依赖模型实例、权重或 CUDA；不修改输入配置，重复调用应保持结果不变。
        子类负责模型专有的配置覆盖，通用工具展开 thinker_config、llm_config、text_config 并同步字段别名。
        """
        from lightllm.common.basemodel.basemodel_utils import normalize_model_config

        return normalize_model_config(model_config)

    def _init_config(self):
        with open(os.path.join(self.weight_dir_, "config.json"), "r") as json_file:
            self.config = self.normalize_model_config(json.load(json_file))
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
        self.quant_cfg = Quantcfg(self.config, self.quant_type, self.quant_cfg_path, self.expert_dtype)
        logger.info(f"Initial quantization. " f"The default quantization method is {self.quant_cfg.quant_type}")

    def _init_model_adapter(self):
        # Subclasses with different recipes opt in explicitly as they migrate.
        adapter_class = type(self).__dict__.get("model_adapter_class")
        self.model_adapter = None
        if adapter_class is not None:
            adapter = self._create_model_adapter(adapter_class)
            if self.args.mtp_mode is not None:
                from lightllm.core.compute.attention.dense.spec import DenseAttentionSpec
                from lightllm.core.compute.attention.gdn.spec import GatedDeltaNetSpec
                from lightllm.core.compute.attention.kda.spec import KimiDeltaAttentionSpec
                from lightllm.core.compute.attention.mla.spec import MLAAttentionSpec
                from lightllm.core.compute.attention.pooled_nope.spec import PooledNoPEAttentionSpec

                # Other recurrent/sparse families still need their own verify bindings.
                if not all(
                    isinstance(
                        spec, (DenseAttentionSpec, GatedDeltaNetSpec, KimiDeltaAttentionSpec, PooledNoPEAttentionSpec)
                    )
                    or type(spec) is MLAAttentionSpec
                    for spec in adapter.attention_specs().values()
                ):
                    return
            self.model_adapter = adapter

    def _create_model_adapter(self, adapter_class):
        return adapter_class(
            self.config, role=self.att_state_role, dtype=self.data_type, tp_world_size=self.tp_world_size_
        )

    def _init_weights(self, start_layer_index=0):
        if getattr(self, "model_adapter", None) is not None:
            assert start_layer_index == 0, "adapter declares its own layer ordering"
            self.pre_post_weight, self.trans_layers_weight = self.model_adapter.build_weights(self.quant_cfg)
            return
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
        return

    def load_weights(self, weight_dict: dict):
        if getattr(self, "model_adapter", None) is not None:
            self.model_adapter.load_weights(
                (self.pre_post_weight, self.trans_layers_weight), self.weight_dir_, weight_dict
            )
            from lightllm.common.basemodel.weight_binding import bind_linears, bind_moe, finalize_weights
            from lightllm.core.registry import default_providers
            from lightllm.common.basemodel.layer_infer.cache_tensor_manager import g_cache_manager

            if getattr(self, "execution_providers", None) is None:
                self.execution_providers = default_providers()
            finalize_weights(
                (self.pre_post_weight, *self.trans_layers_weight), self.execution_device, self.execution_providers
            )
            self.linear_plan = bind_linears(
                (self.pre_post_weight, *self.trans_layers_weight),
                g_cache_manager.empty,
                providers=self.execution_providers,
                runtime=self.execution_device,
            )
            if self.execution_providers.operations:
                for recipe, weight in zip(self.layers_infer, self.trans_layers_weight, strict=True):
                    bind = getattr(recipe, "bind_operations", None)
                    if bind is not None:
                        bind(self.execution_providers, weight, self.execution_device, self.execution_communication)
            self.moe_plan = bind_moe(
                self.model_adapter.moe_specs(),
                self.trans_layers_weight,
                self.layers_infer,
                providers=self.execution_providers,
                runtime=self.execution_device,
            )
            return
        assert weight_dict is None or isinstance(weight_dict, dict), "weight_dict must be a dict or None"
        load_hf_weights(
            data_type=self.data_type,
            weight_dir=self.weight_dir_,
            pre_post_layer=self.pre_post_weight,
            transformer_layer_list=self.trans_layers_weight,
            weight_dict=weight_dict,
        )
        self.pre_post_weight.verify_load()
        [weight.verify_load() for weight in self.trans_layers_weight]
        return

    @classmethod
    def create_full_att_states(
        cls,
        model_config: dict[str, object],
        args: "StartArgs | EasyDict",
        *,
        model_logical_layer_ids: tuple[int, ...],
        tp_rank: int = 0,
        tp_world_size: int = 1,
    ) -> list["FullAttentionState"]:
        """返回本模型的 full-attention 定义列表；默认没有此类资源，不分配 KV 容量。

        model_logical_layer_ids 是当前模型拥有的全局逻辑层 ID，按模型内部层顺序排列；
        具体模型自行选择参与层并定义 logical_layer_groups，不要求使用全部层。
        """
        return []

    @classmethod
    def create_bounded_att_states(
        cls,
        model_config: dict[str, object],
        args: "StartArgs | EasyDict",
        *,
        model_logical_layer_ids: tuple[int, ...],
        request_capacity: int = 1,
        tp_rank: int = 0,
        tp_world_size: int = 1,
        device: str = "meta",
    ) -> list["BoundedAttentionState"]:
        """返回本模型的 bounded 运行态列表；meta 用于布局查询，大小页后续分配。

        model_logical_layer_ids 描述当前模型拥有的全部全局逻辑层，具体模型自行选择参与层。
        """
        return []

    def _check_mem_size(self):
        self.max_total_token_num = self.att_state_manager.full_att_allocator.size

        assert (
            self.max_total_token_num > self.batch_max_tokens
        ), "max_total_token_num must be greater than batch_max_tokens"

        # 非个人性能模式下，需要保证 max_seq_length 小于等于 max_total_token_num,
        # 这样才能得到完整的上下文长度的支持。个人模式主要是私有化场景，显卡显存不是
        # 特别大，可能能分配的 kv 容量有限，无法支持 max_seq_length 的推理。所以个人模式下
        # 可以适当放宽这个限制，不做这个校验。
        if self.args.performance_mode != "personal":
            assert self.max_seq_length <= self.max_total_token_num, (
                f"max_total_token_num must be >= max_seq_length, "
                f"got max_total_token_num={self.max_total_token_num}, "
                f"max_seq_length={self.max_seq_length}. "
                f"Try set --max_req_total_len a smaller value < {self.max_total_token_num}."
            )

        return

    def _init_infer_layer(self, start_layer_index=0):
        if getattr(self, "model_adapter", None) is not None:
            assert start_layer_index == 0, "adapter declares its own layer ordering"
            self.pre_infer, self.post_infer, self.layers_infer = self.model_adapter.build_recipe()
            return
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
        role = self.att_state_role
        if not self.att_state_manager.get_full_attention_states(role):
            # FFN-only and linear-only models do not need a full-attention backend.
            self.prefill_att_backend = self.decode_att_backend = None
            return
        self.prefill_att_backend: BaseAttBackend = get_prefill_att_backend_class(index=0)(model=self)
        self.decode_att_backend: BaseAttBackend = get_decode_att_backend_class(index=0)(model=self)
        return

    def _init_att_backend1(self):
        # self.prefill_att_backend1 是给后续有模型支持不同层用不同的att模块时，保留的扩展。
        self.prefill_att_backend1: BaseAttBackend = None
        self.decode_att_backend1: BaseAttBackend = None
        return

    def _align_decode_batch_size(self, batch_size: int) -> int:
        alignment = self.mtp_manager.get_decode_batch_alignment(self.is_mtp_draft_model)
        if self.args.enable_tpsp_mix_mode:
            alignment = math.lcm(alignment, self.tp_world_size_)
        return triton.cdiv(batch_size, alignment) * alignment

    def _init_cudagraph(self):
        # When graph covers the configured request length, it must also cover MTP's internal token margin.
        if self.args.mtp_mode is not None and self.graph_max_len_in_batch >= self.args.max_req_total_len:
            self.graph_max_len_in_batch = max(self.graph_max_len_in_batch, self.max_seq_length)

        decode_tokens_per_request = self.mtp_manager.get_decode_tokens_per_request(self.is_mtp_draft_model)
        cuda_graph_grow_step_size = self.mtp_manager.get_decode_batch_alignment(self.is_mtp_draft_model)
        self.graph = (
            None
            if self.disable_cudagraph
            else CudaGraph(
                batch_step_size_before_split=cuda_graph_grow_step_size,
                split_batch_size=self.args.graph_split_batch_size * decode_tokens_per_request,
                batch_step_size_after_split=self.args.graph_grow_step_size * cuda_graph_grow_step_size,
                max_batch_size=self.graph_max_batch_size,
                max_len_in_batch=self.graph_max_len_in_batch,
                tp_world_size=self.tp_world_size_,
                capture_infer_cost=self.args.mtp_dynamic_verify,
                runtime=self.execution_runtime,
            )
        )
        if self.graph is not None:
            self.warmup_runner.capture_decode(self.graph, overlap=self.args.enable_decode_microbatch_overlap)

    def _init_prefill_cuda_graph(self):
        self.prefill_graph = (
            None
            if not get_env_start_args().enable_prefill_cudagraph
            else PrefillCudaGraph(
                decode_cuda_graph=self.graph, tp_world_size=self.tp_world_size_, runtime=self.execution_runtime
            )
        )
        if self.prefill_graph is not None:
            self.warmup_runner.capture_prefill(self.prefill_graph, overlap=self.args.enable_prefill_microbatch_overlap)

    def _init_custom(self):
        pass

    def _init_hidden_collector(self):
        self.hidden_collector_prototype = self.mtp_manager.create_hidden_collector(model=self)

    @torch.no_grad()
    def forward(self, model_input: ModelInput):
        model_input.to_device(self.req_manager.req_to_token_indexs.device)

        if model_input.is_prefill:
            return self._prefill(model_input=model_input)
        else:
            return self._decode(model_input)

    def _select_mem_indexes(self, model_input: ModelInput):
        return select_token_indices(self.req_manager.req_to_token_indexs, model_input)

    def _create_inferstate(self, model_input: ModelInput, microbatch_index: int = 0):
        return self.batch_state_factory.create(model_input, microbatch_index)

    def _create_padded_decode_model_input(self, model_input: ModelInput, new_batch_size: int):
        return pad_decode(model_input, new_batch_size, self.req_manager.HOLD_REQUEST_ID)

    def _create_padded_prefill_model_input(self, model_input: ModelInput, new_handle_token_num: int):
        return pad_prefill(model_input, new_handle_token_num, self.req_manager.HOLD_REQUEST_ID)

    def _create_model_output(self, post_output: PostLayerOutput, infer_state: InferStateInfo) -> ModelOutput:
        return make_output(post_output, infer_state)

    def _create_unpad_decode_model_output(self, model_output: ModelOutput, origin_batch_size: int):
        return unpad_decode(model_output, origin_batch_size)

    def _create_unpad_prefill_model_output(
        self, padded_model_output: ModelOutput, origin_handle_token_num: int, origin_batch_size: int
    ):
        return unpad_prefill(padded_model_output, origin_handle_token_num, origin_batch_size)

    def _prefill(self, model_input: ModelInput):
        return self.batch_runner.run_prefill(model_input)

    def _decode(self, model_input: ModelInput) -> ModelOutput:
        return self.batch_runner.run_decode(model_input)

    @final
    def _context_forward(self, infer_state: InferStateInfo):
        input_embs = self.pre_infer.context_forward(infer_state.input_ids, infer_state, self.pre_post_weight)
        if self.args.enable_dp_prefill_balance:
            assert not self.args.enable_prefill_cudagraph, "not support now"
            infer_state.prepare_prefill_dp_balance()
            input_embs = infer_state._all_to_all_balance_get(data=input_embs)

        input_embs = self.pre_infer._tpsp_sp_split(input=input_embs, infer_state=infer_state)
        input_tensors = [input_embs]
        if Autotuner.is_autotune_warmup():
            infer_state.hidden_collector = NoopHiddenCollector()

        def prefill_func(input_tensors, _infer_state):
            hidden_collector = _infer_state.hidden_collector
            _input_embs = input_tensors[0]
            for i in range(self.layers_num):
                layer = self.layers_infer[i]
                _input_embs = layer.context_forward(_input_embs, _infer_state, self.trans_layers_weight[i])
                hidden_collector.add(
                    layer_index=i,
                    hidden=_input_embs,
                )

            return [_input_embs]

        output_tensors = self.graph_runner.run_prefill(prefill_func, input_tensors, infer_state)

        input_embs = output_tensors[0]

        last_input_embs = self.post_infer._tpsp_allgather(input=input_embs, infer_state=infer_state)
        if infer_state.need_dp_prefill_balance:
            last_input_embs = infer_state._all_to_all_unbalance_get(data=last_input_embs)

        post_output = self.post_infer.token_forward(last_input_embs, infer_state, self.pre_post_weight)
        hidden_collector = infer_state.hidden_collector
        hidden_collector.add_final_hidden(last_input_embs)
        model_output = self._create_model_output(post_output, infer_state)
        del post_output

        # 在开启使用deepep的时候，需要调用clear_deepep_buffer做资源清理，没有启用的时候
        # 该调用没有实际意义
        dist_group_manager.clear_deepep_buffer()
        return model_output

    def _token_forward(self, infer_state: InferStateInfo):
        hidden_collector = infer_state.hidden_collector
        input_ids = infer_state.input_ids
        cuda_input_ids = input_ids
        input_embs = self.pre_infer.token_forward(cuda_input_ids, infer_state, self.pre_post_weight)
        input_embs = self.pre_infer._tpsp_sp_split(input=input_embs, infer_state=infer_state)

        for i in range(self.layers_num):
            layer = self.layers_infer[i]
            input_embs: torch.Tensor = layer.token_forward(input_embs, infer_state, self.trans_layers_weight[i])
            hidden_collector.add(layer_index=i, hidden=input_embs)

        last_input_embs = self.post_infer._tpsp_allgather(input=input_embs, infer_state=infer_state)
        post_output: PostLayerOutput = self.post_infer.token_forward(
            last_input_embs, infer_state=infer_state, layer_weight=self.pre_post_weight
        )

        hidden_collector.add_final_hidden(last_input_embs)
        model_output = self._create_model_output(post_output, infer_state)
        del post_output

        # 在 cuda graph 模式下，输出需要转为 no ref tensor, 加强mem pool 的复用，降低显存的使用。
        if infer_state.is_cuda_graph_capture:
            model_output.to_no_ref_tensor(self.execution_runtime.borrow_tensor)

        return model_output

    @torch.no_grad()
    def microbatch_overlap_prefill(self, model_input0: ModelInput, model_input1: ModelInput):
        for model_input in (model_input0, model_input1):
            self.batch_runner.prepare_input(model_input, move_to_device=True)
        return self._microbatch_overlap_prefill_cuda(model_input0, model_input1)

    def _microbatch_overlap_prefill_cuda(self, model_input0: ModelInput, model_input1: ModelInput):
        assert self.args.enable_tpsp_mix_mode
        return self.batch_runner.run_prefill(model_input0, model_input1, prepared=True)

    @torch.no_grad()
    def microbatch_overlap_decode(self, model_input0: ModelInput, model_input1: ModelInput):
        for model_input in (model_input0, model_input1):
            self.batch_runner.prepare_input(model_input, move_to_device=True)
        return self._microbatch_overlap_decode_cuda(model_input0, model_input1)

    def _microbatch_overlap_decode_cuda(self, model_input0: ModelInput, model_input1: ModelInput):
        assert self.args.enable_tpsp_mix_mode
        return self.batch_runner.run_decode(model_input0, model_input1, prepared=True)

    @final
    def _overlap_tpsp_context_forward(self, infer_state: InferStateInfo, infer_state1: InferStateInfo):
        g_cache_manager.cache_env_in()
        hidden_collector0 = infer_state.hidden_collector
        hidden_collector1 = infer_state1.hidden_collector

        input_embs, input_embs1 = self.pre_infer.overlap_tpsp_context_forward(
            infer_state.input_ids, infer_state1.input_ids, infer_state, infer_state1, self.pre_post_weight
        )

        # 决定是否进行 dp balance 优化，可以提升dp > 1 时的 prefill 效率。
        if get_env_start_args().enable_dp_prefill_balance:
            assert not self.args.enable_prefill_cudagraph, "not support now"
            infer_state.prepare_prefill_dp_balance()
            infer_state1.prepare_prefill_dp_balance()
            input_embs = infer_state._all_to_all_balance_get(data=input_embs)
            input_embs1 = infer_state1._all_to_all_balance_get(data=input_embs1)

        input_embs = self.pre_infer._tpsp_sp_split(input=input_embs, infer_state=infer_state)
        input_embs1 = self.pre_infer._tpsp_sp_split(input=input_embs1, infer_state=infer_state1)

        for i in range(self.layers_num):
            input_embs, input_embs1 = self.layers_infer[i].overlap_tpsp_context_forward(
                input_embs, input_embs1, infer_state, infer_state1, self.trans_layers_weight[i]
            )
            hidden_collector0.add(
                layer_index=i,
                hidden=input_embs,
            )
            hidden_collector1.add(
                layer_index=i,
                hidden=input_embs1,
            )

        # 折叠模式调用完infer_state 和 infer_state1 上的hook函数后，input_embs 和 input_embs1 才具备正确的运算数据。
        infer_state.call_overlap_hook()
        infer_state1.call_overlap_hook()

        last_input_embs = self.post_infer._tpsp_allgather(input=input_embs, infer_state=infer_state)
        last_input_embs1 = self.post_infer._tpsp_allgather(input=input_embs1, infer_state=infer_state1)
        if infer_state.need_dp_prefill_balance:
            last_input_embs = infer_state._all_to_all_unbalance_get(data=last_input_embs)
            last_input_embs1 = infer_state1._all_to_all_unbalance_get(data=last_input_embs1)

        post_output, post_output1 = self.post_infer.overlap_tpsp_token_forward(
            last_input_embs, last_input_embs1, infer_state, infer_state1, self.pre_post_weight
        )
        g_cache_manager.cache_env_out()

        hidden_collector0.add_final_hidden(last_input_embs)
        hidden_collector1.add_final_hidden(last_input_embs1)
        model_output = self._create_model_output(post_output, infer_state)
        model_output1 = self._create_model_output(post_output1, infer_state1)
        del post_output, post_output1

        return model_output, model_output1

    @final
    def _overlap_tpsp_token_forward(self, infer_state: InferStateInfo, infer_state1: InferStateInfo):
        hidden_collector0 = infer_state.hidden_collector
        hidden_collector1 = infer_state1.hidden_collector
        input_embs, input_embs1 = self.pre_infer.overlap_tpsp_token_forward(
            infer_state.input_ids, infer_state1.input_ids, infer_state, infer_state1, self.pre_post_weight
        )
        input_embs = self.pre_infer._tpsp_sp_split(input=input_embs, infer_state=infer_state)
        input_embs1 = self.pre_infer._tpsp_sp_split(input=input_embs1, infer_state=infer_state1)

        for i in range(self.layers_num):
            input_embs, input_embs1 = self.layers_infer[i].overlap_tpsp_token_forward(
                input_embs, input_embs1, infer_state, infer_state1, self.trans_layers_weight[i]
            )
            hidden_collector0.add(
                layer_index=i,
                hidden=input_embs,
            )
            hidden_collector1.add(
                layer_index=i,
                hidden=input_embs1,
            )

        # 折叠模式调用完infer_state 上的hook函数后，input_embs 和 input_embs 才具备正确的运算数据。
        infer_state.call_overlap_hook()
        infer_state1.call_overlap_hook()

        last_input_embs = self.post_infer._tpsp_allgather(input=input_embs, infer_state=infer_state)
        last_input_embs1 = self.post_infer._tpsp_allgather(input=input_embs1, infer_state=infer_state1)

        post_output, post_output1 = self.post_infer.overlap_tpsp_token_forward(
            last_input_embs, last_input_embs1, infer_state, infer_state1, self.pre_post_weight
        )

        hidden_collector0.add_final_hidden(last_input_embs)
        hidden_collector1.add_final_hidden(last_input_embs1)
        model_output = self._create_model_output(post_output, infer_state)
        model_output1 = self._create_model_output(post_output1, infer_state1)
        del post_output, post_output1

        if infer_state.is_cuda_graph_capture:
            model_output.to_no_ref_tensor(self.execution_runtime.borrow_tensor)
            model_output1.to_no_ref_tensor(self.execution_runtime.borrow_tensor)

        return model_output, model_output1

    @final
    def _check_max_len_infer(self):
        return self.warmup_runner.check_max_length(self.batch_max_tokens)

    def autotune_layers(self):
        # 控制autotune的层数，用于适配不同模型
        return self.config.get("first_k_dense_replace", 0) + 1

    @final
    def _autotune_warmup(self):
        return self.warmup_runner.autotune(self.batch_max_tokens, self._autotune_recipe)

    @contextmanager
    def _autotune_recipe(self):
        layers = self.layers_num
        self.layers_num = min(layers, self.autotune_layers())
        try:
            yield
        finally:
            self.layers_num = layers

    @final
    def _init_padded_req(self):
        return self.warmup_runner.initialize_hold()

    def _gen_special_model_input(self, token_num: int):
        special_model_input = {}

        if self.is_mtp_draft_model:
            special_model_input["mtp_draft_input_hiddens"] = torch.randn(
                token_num, self.config["hidden_size"], dtype=self.data_type, device=self.execution_torch_device
            )
        else:
            special_model_input["mtp_draft_input_hiddens"] = None

        return special_model_input
