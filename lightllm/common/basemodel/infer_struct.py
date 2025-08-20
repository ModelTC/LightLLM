import torch
from lightllm.common.mem_manager import MemoryManager
from lightllm.common.req_manager import ReqManager
from lightllm.distributed import CustomProcessGroup
from typing import Tuple, Any, Optional
from .triton_kernel.gen_prefill_params import gen_prefill_params
from .triton_kernel.gen_decode_params import gen_decode_params
from .triton_kernel.multimodal_emb import mark_multimodal_obj
from .batch_objs import ModelInput


class InferStateInfo:
    """
    推理时用的信息结构体
    """

    def __init__(self):
        self.batch_size: int = None
        self.total_token_num: int = None
        self.b_req_idx: torch.Tensor = None
        self.b_start_loc: torch.Tensor = None
        self.b_ready_cache_len: torch.Tensor = None  # only for prefill prompt cache used.
        self.b_seq_len: torch.Tensor = None
        # max_len_in_batch prefill 和 decode 阶段含义不同
        # prefill 阶段指每个req 输入token的长度（不包括已经cache的部分）最大值
        # decode 阶段指的是每个req的总长 最大值
        self.max_len_in_batch: int = None
        self.is_prefill: bool = None

        self.mem_manager: MemoryManager = None
        self.req_manager: ReqManager = None

        self.mem_index: torch.Tensor = None
        self.kv_buffer_shapedtype: Tuple[Any, Any] = None

        self.is_token_healing: bool = False
        self.return_all_prompt_logics: bool = False
        self.multimodal_params: dict = None
        self.is_cuda_graph: bool = False  # 标记是否是cuda graph的捕获推理
        self.dist_group: CustomProcessGroup = None

        # 在microbatch overlap的运行模式下，用于标记当前 microbatch 的 index 序号
        # 在一些细节场景下需要有该信息区分一些资源的申请和管理。
        self.microbatch_index: int = 0

        # 衍生使用的管理变量，为了方便扩展接入其他的高性能attention推理算子，在
        # inferstate 基类上添加下面的标记变量，用于扩展。
        # b 开头的tensor变量其shape为[batch_size,]
        # b1 开头的tensor变量其shape为[batch_size + 1,]
        self.b_q_seq_len: torch.Tensor = None
        self.b1_cu_q_seq_len: torch.Tensor = None
        self.b_kv_seq_len: torch.Tensor = None
        self.b1_cu_kv_seq_len: torch.Tensor = None
        self.position_ids: torch.Tensor = None
        self.max_q_seq_len: int = None
        self.max_kv_seq_len: int = None

        # 一些特殊模型，特殊模式使用的输入变量，本身这些变量不适合放在
        # inferstate的基类中，但是为了代码的简洁和方便，都放在基类中
        # 进行管理。注意这些成员变量只会在特定的模型和模式下才会生效。

        # deepseekv3 mtp draft model 使用的额外输入参数,
        # 在开启 mtp_mode == deepseekv3 时，mtp draft model
        # 的输入会用到，其他模型和场景都不会用到
        self.deepseekv3_mtp_draft_input_hiddens: Optional[torch.Tensor] = None

    def init_some_extra_state(self, model, input_ids: torch.Tensor):
        if self.is_prefill:
            (
                self.b_q_seq_len,
                self.b1_cu_q_seq_len,
                self.b_kv_seq_len,
                self.b1_cu_kv_seq_len,
                self.position_ids,
                self.max_q_seq_len,
                self.max_kv_seq_len,
            ) = gen_prefill_params(
                input_token_num=input_ids.shape[0],
                b_ready_cache_len=self.b_ready_cache_len,
                b_seq_len=self.b_seq_len,
            )
            self.b_start_loc = self.b1_cu_q_seq_len[0:-1]
        else:
            (
                self.b_q_seq_len,
                self.b1_cu_q_seq_len,
                self.b_kv_seq_len,
                self.b1_cu_kv_seq_len,
                self.position_ids,
            ) = gen_decode_params(self.b_seq_len)
            self.max_q_seq_len = 1
            # TODO: check the correctness
            self.max_kv_seq_len = self.max_len_in_batch
            self.b_start_loc = self.b1_cu_kv_seq_len[0:-1]

    def copy_for_cuda_graph(self, new_infer_state: "InferStateInfo"):
        for attr_name, attr_value in vars(new_infer_state).items():
            if isinstance(attr_value, torch.Tensor):
                attr_ = getattr(self, attr_name, None)
                if attr_ is not None and attr_.data_ptr() != attr_value.data_ptr():
                    attr_.copy_(attr_value, non_blocking=True)
        return

    def mark_multimodal_objs_for_prefill(self, input_ids: torch.Tensor):
        """
        功能函数，用于标记在chuncked prefill的过程中，到底哪些多模态对象对应的token是需要参与计算的。
        因为分chunck的原因，并不是所有的多模态对象对应的token都需要参与计算。
        """
        multi_objs = []
        for _, p in enumerate(self.multimodal_params):
            for obj in p["images"] + p["audios"]:
                multi_objs.append(obj)

        if multi_objs:
            obj_start_ids = torch.tensor([e["token_id"] for e in multi_objs], dtype=torch.int64, device="cuda")
            obj_token_lens = torch.tensor([e["token_num"] for e in multi_objs], dtype=torch.int64, device="cuda")
            marks = mark_multimodal_obj(
                obj_start_token_ids=obj_start_ids, obj_token_lens=obj_token_lens, input_ids=input_ids
            )
            marks_array = marks.detach().cpu().numpy()
            for mark, obj in zip(marks_array, multi_objs):
                obj["_prefill_"] = mark > 0
        return
