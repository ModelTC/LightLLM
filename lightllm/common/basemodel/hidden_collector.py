from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import List, Optional

import torch
from transformers.configuration_utils import PretrainedConfig

from lightllm.common.basemodel.batch_objs import ModelMtpOutputCollector
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.tensor_utils import tensor_to_no_ref_tensor


class HiddenCollector(ABC):
    """Hidden state 收集器的抽象基类。

    推理过程中，模型会在每一层计算完成后调用 :meth:`add`，并在一次 forward
    结束时调用 :meth:`finish_output` 生成统一的辅助输出。不同投机解码模式可以
    通过子类决定不收集 hidden、只返回最终层 hidden、收集若干中间层 hidden，
    或同时收集 MTP head 生成的 token 与置信度。

    BaseModel 持有一个不承载推理状态的 prototype，每个 InferStateInfo 通过
    :meth:`new_instance` 获得独立收集器，使不同请求和 microbatch 的临时 hidden
    state 自然隔离，具体实现无需感知 microbatch 编号。
    """

    @abstractmethod
    def new_instance(self) -> "HiddenCollector":
        """基于当前 prototype 创建一个不包含运行时 tensor 的新实例。

        新实例可以复用 model、目标层编号等只读配置，但不得共享 ``layer_hiddens``
        或 ``final_hidden`` 等运行时状态。该接口用于替代 ``deepcopy``，避免递归
        复制 model、CUDA tensor 和通信对象。

        Returns:
            与当前 collector 类型和配置一致、运行时状态为空的新实例。
        """
        raise NotImplementedError

    def restore_graph_state(self, graph_collector: "HiddenCollector") -> None:
        """从 Prefill CUDA Graph 的 capture collector 恢复本次 replay 所需状态。

        基类默认不恢复任何数据。需要中间层 hidden 的子类应只复制保存 tensor
        引用的容器，不复制 tensor 本身，使本次 infer state 可以读取 graph replay
        更新后的固定地址，同时在 :meth:`finish_output` 后独立清理自己的容器。

        Args:
            graph_collector: capture 阶段保存在 Prefill CUDA Graph 中的只读 collector。
        """
        return

    def release_graph_tensor_ownership(self) -> tuple[int, int]:
        """将 Prefill CUDA Graph capture 状态转换为不持有显存所有权的引用。

        普通 collector 不应调用该接口。完成 CUDA Graph capture 后，graph memory
        pool 会负责保证固定显存地址的生命周期；collector 只需保存指针视图供
        replay 后读取。基类没有需要转换的 tensor，因此返回零统计值。

        Returns:
            ``(tensor_count, total_nbytes)``，分别表示转换的 tensor 数量及其总字节数。
        """
        return 0, 0

    def add(self, layer_index: int, hidden: torch.Tensor) -> None:
        """接收一个 decoder layer 刚计算完成的本地 hidden state。

        BaseModel 会按照模型层顺序逐层调用该接口。基类默认不保存 tensor，适用
        于不需要中间层 hidden 的实现；需要收集中间层的子类应重写该方法，并在
        必要时 clone 会被后续层复用的输入缓冲区。

        Args:
            layer_index: 当前 decoder layer 的零基索引。
            hidden: 当前层输出的本地 hidden tensor，可能仍是 TP/SP 切分状态。
        """
        return

    def add_final_hidden(self, final_hidden: torch.Tensor) -> None:
        """接收完成模型输出侧 gather 后的最终层 hidden tensor。

        BaseModel 在 logits 计算完成后、调用 :meth:`finish_output` 前调用该接口。基类
        默认不保存 tensor；需要直接返回最终层 hidden 的子类应重写该方法并保存
        引用，随后在 :meth:`finish_output` 中消费和清理。

        Args:
            final_hidden: 已完成 TP/SP all-gather 和 DP unbalance 的最终层 hidden。
        """
        return

    def add_mtp_outputs(
        self,
        draft_token_ids: Optional[torch.Tensor],
        confidence_logits: Optional[torch.Tensor],
        draft_token_probs: Optional[torch.Tensor] = None,
    ) -> None:
        """Collect optional token/confidence outputs produced by an MTP head.

        Only draft models with a specialized output head should call this
        hook. Raising here makes an incorrect collector selection fail fast
        instead of silently dropping model outputs.
        """

        raise RuntimeError(f"{self.__class__.__name__} does not collect MTP head outputs")

    @abstractmethod
    def finish_output(self, infer_state) -> ModelMtpOutputCollector:
        """结束当前 microbatch 的收集并生成统一的 MTP 输出。

        子类应在此完成必要的拼接、TP/SP all-gather、DP unbalance 和 contiguous
        转换，将结果封装为 :class:`ModelMtpOutputCollector`，并在返回前清理当前
        实例的临时状态。不需要提供额外 MTP 输出的实现应返回一个空 collector。

        Args:
            infer_state: 当前 forward 的推理状态，包含通信拓扑、DP balance 等信息。

        Returns:
            本次 forward 的 MTP 辅助输出；未启用 MTP 时返回内容为空的 collector。
        """
        raise NotImplementedError


class NoopHiddenCollector(HiddenCollector):
    """Null object used by models that do not expose speculative features."""

    def new_instance(self) -> HiddenCollector:
        return NoopHiddenCollector()

    def finish_output(self, infer_state) -> ModelMtpOutputCollector:
        return ModelMtpOutputCollector()


class MtpHeadOutputCollector(NoopHiddenCollector):
    """Collect outputs from an MTP draft head that does not expose hidden state."""

    def __init__(self) -> None:
        self.draft_token_ids: Optional[torch.Tensor] = None
        self.draft_token_probs: Optional[torch.Tensor] = None
        self.confidence_logits: Optional[torch.Tensor] = None

    def new_instance(self) -> HiddenCollector:
        return MtpHeadOutputCollector()

    def add_mtp_outputs(
        self,
        draft_token_ids: Optional[torch.Tensor],
        confidence_logits: Optional[torch.Tensor],
        draft_token_probs: Optional[torch.Tensor] = None,
    ) -> None:
        self.draft_token_ids = draft_token_ids
        self.draft_token_probs = draft_token_probs
        self.confidence_logits = confidence_logits

    def finish_output(self, infer_state) -> ModelMtpOutputCollector:
        output = ModelMtpOutputCollector(
            draft_token_ids=self.draft_token_ids,
            draft_token_probs=self.draft_token_probs,
            confidence_logits=self.confidence_logits,
        )
        self.draft_token_ids = None
        self.draft_token_probs = None
        self.confidence_logits = None
        return output


class FinalHiddenCollector(HiddenCollector):
    """Returns the final decoder hidden state without per-layer bookkeeping."""

    def __init__(self) -> None:
        self.final_hidden: Optional[torch.Tensor] = None

    def new_instance(self) -> HiddenCollector:
        return FinalHiddenCollector()

    def add_final_hidden(self, final_hidden: torch.Tensor) -> None:
        self.final_hidden = final_hidden

    def finish_output(self, infer_state) -> ModelMtpOutputCollector:
        assert self.final_hidden is not None
        final_hidden = self.final_hidden
        self.final_hidden = None
        return ModelMtpOutputCollector(spec_hidden=final_hidden.contiguous())


class LayerHiddenCollector(HiddenCollector):
    """Collects selected decoder-layer outputs for an intermediate-hidden draft."""

    def __init__(self, model) -> None:
        self.model = model
        self.layer_num = model.layers_num
        self.layer_ids = self._load_layer_ids()
        self.layer_hiddens: List[torch.Tensor] = []

    def new_instance(self) -> HiddenCollector:
        collector = copy.copy(self)
        collector.layer_hiddens = []
        return collector

    def restore_graph_state(self, graph_collector: HiddenCollector) -> None:
        assert isinstance(graph_collector, LayerHiddenCollector)
        self.layer_hiddens = graph_collector.layer_hiddens.copy()

    def release_graph_tensor_ownership(self) -> tuple[int, int]:
        tensor_count = len(self.layer_hiddens)
        total_nbytes = sum(hidden.numel() * hidden.element_size() for hidden in self.layer_hiddens)
        self.layer_hiddens = [tensor_to_no_ref_tensor(hidden) for hidden in self.layer_hiddens]
        return tensor_count, total_nbytes

    def _load_layer_ids(self) -> frozenset[int]:
        draft_model_dirs = get_env_start_args().mtp_draft_model_dir
        assert draft_model_dirs
        draft_config, _ = PretrainedConfig.get_config_dict(draft_model_dirs[0])
        layer_ids = draft_config.get("target_layer_ids")
        if layer_ids is None:
            layer_ids = draft_config.get("dflash_config", {}).get("target_layer_ids")
        assert layer_ids is not None, f"target_layer_ids is required in draft config: {draft_model_dirs[0]}"

        resolved_layer_ids = frozenset(int(layer_id) for layer_id in layer_ids)
        assert resolved_layer_ids and all(
            0 <= layer_id < self.layer_num for layer_id in resolved_layer_ids
        ), f"invalid target_layer_ids={resolved_layer_ids} for target layer_num={self.layer_num}"
        return resolved_layer_ids

    def add(self, layer_index: int, hidden: torch.Tensor) -> None:
        if layer_index not in self.layer_ids:
            return
        # Most LightLLM layers reuse their input buffer. Preserve intermediate
        # layers while allowing the final layer output to remain zero-copy.
        self.layer_hiddens.append(hidden if layer_index == self.layer_num - 1 else hidden.clone())

    def _local_hidden(self) -> torch.Tensor:
        assert len(self.layer_hiddens) == len(
            self.layer_ids
        ), f"captured {len(self.layer_hiddens)} hidden layers, expected {len(self.layer_ids)}"
        if len(self.layer_hiddens) == 1:
            return self.layer_hiddens[0]
        return torch.cat(self.layer_hiddens, dim=-1)

    def finish_output(self, infer_state) -> ModelMtpOutputCollector:
        local_hidden = self._local_hidden()
        self.layer_hiddens.clear()
        hidden = self.model.pre_infer._tpsp_allgather(input=local_hidden, infer_state=infer_state)
        if infer_state.need_dp_prefill_balance:
            hidden = infer_state._all_to_all_unbalance_get(data=hidden)
        return ModelMtpOutputCollector(spec_hidden=hidden.contiguous())
