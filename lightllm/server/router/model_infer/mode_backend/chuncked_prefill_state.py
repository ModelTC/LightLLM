import dataclasses
import numpy as np
from typing import List
from lightllm.utils.envs_utils import get_env_start_args
from ..infer_batch import InferReq
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


@dataclasses.dataclass
class ChunkedPrefillState:
    """
    用于保存和控制 chuncked prefill 推理调度的控制状态，因为不同的场景，对于首字和包间的
    诉求不是特别一致，所以需要一个状态来控制。主要通过 args.router_max_wait_tokens 参数
    可以调节 prefill 的激进程度和方式，来协调首字和包间的平衡。
    """

    prefill_wait_step: int = 0
    need_prefill_count: int = 0
    current_wait_step: int = 0

    # dp chuncked prefill 的等待步数参数
    dp_prefill_wait_step: int = 0
    dp_current_wait_step: int = 0

    # world_size
    _global_world_size: int = 0

    def __post_init__(self):
        args = get_env_start_args()
        self.prefill_wait_step = args.router_max_wait_tokens
        self.dp_prefill_wait_step = args.dp_prefill_wait_step
        self._global_world_size = args.tp
        return

    def need_prefill(self, prefill_reqs: List[InferReq], decode_reqs: List[InferReq]) -> bool:
        no_decode_reqs = len(decode_reqs) == 0
        step_ok = self.current_wait_step >= self.prefill_wait_step
        need_prefill = self.need_prefill_count > 0

        if no_decode_reqs or step_ok or need_prefill:
            if need_prefill:
                self.need_prefill_count -= 1

            self.current_wait_step = 0
            if prefill_reqs:
                return True
            else:
                return False
        else:
            if prefill_reqs:
                self.current_wait_step += 1
            return False

    def dp_need_prefill(
        self,
        prefill_reqs: List[InferReq],
        decode_reqs: List[InferReq],
        dp_prefill_req_nums: np.ndarray,
        dp_max_prefill_num: int,
    ) -> bool:
        """
        dp_need_prefill 接口用于控制 DP 模式下进行chuncked prefill时，需要考虑各个DP的真实运行请求数量：
        考虑 8 个 dp 的场景，如果每个 dp 执行 prefill 的请求的数量分别为: [1, 1, 0, 0, 0, 0, 0, 0], 则在运行
        的过程中，请求数量为0的dp会pad一个fake req来参与计算，但是这会导致这些dp因为一些通信同步的原因，造成大量
        算力浪费，实际有效率很低。
        解决方法：
        在判断是否可以进行 prefill 的时候，需要先考虑所有dp的请求数量是否均衡，浪费率是否在可以接受的范围，如果无法
        接受这么高的浪费率，则可以延迟 prefill 的执行时机，直到所有dp的浪费率较低时再进行prefill, 不过延迟执行的极限
        等待时间，受到 dp_prefill_wait_step 参数的控制。
        """
        assert dp_prefill_req_nums.shape[0] == self._global_world_size

        use_ratio = np.count_nonzero(dp_prefill_req_nums) / dp_prefill_req_nums.shape[0]
        step_ok = self.dp_current_wait_step >= self.dp_prefill_wait_step

        if dp_max_prefill_num > 0 and (use_ratio > 0.6 or step_ok):
            if use_ratio < 0.2:
                self.dp_current_wait_step = 0
                logger.info(f"dp chuncked prefill effective GPU Utilization Rate {use_ratio}")

            return True
        else:
            if dp_max_prefill_num > 0:
                self.dp_current_wait_step += 1
            else:
                self.dp_current_wait_step = 0
            return False
