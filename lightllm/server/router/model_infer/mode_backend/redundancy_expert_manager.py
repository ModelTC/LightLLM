# 对于 deepseekv3 模型在 ep 运行模式下，自动分析统计各个专家的出现频率，然后
# 自动更新当前的冗余专家为新的冗余专家。
import torch
import time
import enum
import lightllm.utils.petrel_helper as utils
import threading
from typing import List
from lightllm.common.basemodel.basemodel import TpPartBaseModel
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe_weight_ep_redundancy import (
    FusedMoeWeightEPAutoRedundancy,
)
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe_weight_ep import FusedMoeWeightEP
from lightllm.utils.envs_utils import get_env_start_args, get_redundancy_expert_update_interval
from lightllm.utils.envs_utils import get_redundancy_expert_update_max_load_count
from lightllm.utils.dist_utils import get_global_rank
from lightllm.common.basemodel.layer_weights.hf_load_utils import load_func
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class RedundancyExpertManager:
    def __init__(self, model: TpPartBaseModel):
        self.args = get_env_start_args()
        self.model = model
        self.ep_fused_moeweights: List[FusedMoeWeightEPAutoRedundancy] = []
        for layer in self.model.trans_layers_weight:
            ep_weights = self._find_members_of_class(layer, FusedMoeWeightEP)
            assert len(ep_weights) <= 1
            self.ep_fused_moeweights.extend([FusedMoeWeightEPAutoRedundancy(e) for e in ep_weights])

        # save load params
        self.use_safetensors = True
        files = utils.PetrelHelper.list(self.args.model_dir, extension="all")
        candidate_files = list(filter(lambda x: x.endswith(".safetensors"), files))
        if len(candidate_files) == 0:
            self.use_safetensors = False
            candidate_files = list(filter(lambda x: x.endswith(".bin"), files))
        assert len(candidate_files) != 0, "can only support pytorch tensor and safetensors format for weights."
        self.candidate_files = candidate_files

        # state 1. check_to_update 2. prepare_update 3. start_load_hf_weights 4. wait_load_ready, 5. commit
        self.state: _STATE = _STATE.CHECK_TO_UPDATE
        self.update_time = time.time()
        self.update_interval = get_redundancy_expert_update_interval()
        self.load_thread: threading.Thread = None
        self.global_rank = get_global_rank()
        # 冗余专家的最大加载次数
        self.load_count = 0
        self.max_load_count = get_redundancy_expert_update_max_load_count()

        # 清理counter
        self._clear_all_counter()

    def step(self):
        if self.load_count >= self.max_load_count:
            return

        if self.state == _STATE.CHECK_TO_UPDATE:
            cur_time = time.time()
            if cur_time - self.update_time > self.update_interval:
                self.update_time = cur_time
                self.state = _STATE.PREPARE_UPDATE
                logger.info(f"global_rank {self.global_rank} state to prepare update")
        elif self.state == _STATE.PREPARE_UPDATE:
            self._prepare_load_new_redundancy_expert()
            self.state = _STATE.START_LOAD_HF_WEIGHTS
            logger.info(f"global_rank {self.global_rank} state to start load hf weights")

        elif self.state == _STATE.START_LOAD_HF_WEIGHTS:
            self.load_thread = threading.Thread(target=self._load_hf_weights, daemon=True)
            self.load_thread.start()
            self.state = _STATE.WAIT_LOAD_READY
            logger.info(f"global_rank {self.global_rank} state to wait load ready")

        elif self.state == _STATE.WAIT_LOAD_READY:
            if not self.load_thread.is_alive():
                self.load_thread = None
                self.state = _STATE.COMMIT
                logger.info(f"global_rank {self.global_rank} state to commit")

        elif self.state == _STATE.COMMIT:
            self._commit()
            self.state = _STATE.CHECK_TO_UPDATE
            self.load_count += 1
            logger.info(f"global_rank {self.global_rank} state to check to update")
        return

    def _prepare_load_new_redundancy_expert(self):
        for w in self.ep_fused_moeweights:
            w.prepare_redundancy_experts()
        return

    def _load_hf_weights(self):
        start = time.time()
        try:
            for file in self.candidate_files:
                load_func(
                    file,
                    use_safetensors=self.use_safetensors,
                    pre_post_layer=None,
                    transformer_layer_list=self.ep_fused_moeweights,
                    weight_dir=self.args.model_dir,
                )
        except BaseException as e:
            logger.exception(str(e))
            raise e
        cost_time = time.time() - start
        logger.info(f"global rank {self.global_rank} load redundancy_expert cost time: {cost_time} s")
        return

    def _commit(self):
        for w in self.ep_fused_moeweights:
            w.commit()
        return

    def _find_members_of_class(self, obj, cls):
        members = []
        for attr in dir(obj):
            value = getattr(obj, attr)
            if isinstance(value, cls):
                members.append(value)
        return members

    def _clear_all_counter(self):
        for w in self.ep_fused_moeweights:
            w.clear_counter()
        return


class _STATE(enum.Enum):
    CHECK_TO_UPDATE = 0
    PREPARE_UPDATE = 1
    START_LOAD_HF_WEIGHTS = 2
    WAIT_LOAD_READY = 3
    COMMIT = 4
