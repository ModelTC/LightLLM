import gc
from typing import List, Optional

import torch

from lightllm.utils.dist_utils import init_custom_process_group
from lightllm.utils.rl.serialization import LocalSerializedTensor, MultiprocessingSerializer
from lightllm.utils.rl.tensor_bucket import FlattenedTensorBucket, FlattenedTensorMetadata
from lightllm.utils.rl.torch_cuda_ipc import cuda_rebuild_device_fallback, monkey_patch_torch_reductions
from lightllm.utils.torch_memory_saver_utils import MemoryTag
from lightllm.server.io_struct import (
    FlushCacheReq,
    InitWeightsUpdateGroupReq,
    DestroyWeightsUpdateGroupReq,
    UpdateWeightsFromDistributedReq,
    UpdateWeightsFromIPCReq,
    UpdateWeightsFromTensorReq,
)


class RlBackendOps:
    MEMORY_TAG_ORDER = (MemoryTag.WEIGHT, MemoryTag.KV_CACHE, MemoryTag.GRAPH)

    SUPPORTED = frozenset(
        {
            "flush_cache",
            "release_memory_occupation",
            "resume_memory_occupation",
            "init_weights_update_group",
            "destroy_weights_update_group",
            "update_weights_from_distributed",
            "update_weights_from_tensor",
            "update_weights_from_ipc",
        }
    )

    def __init__(self, backend) -> None:
        self.backend = backend
        self._model_update_group = {}
        self._skip_tensor_updates_reason = None
        self.logger = backend.logger

    @classmethod
    def supports(cls, op_name: str) -> bool:
        return op_name in cls.SUPPORTED

    def dispatch(self, op_name: str, op_args):
        if not self.supports(op_name):
            raise ValueError(f"RlBackendOps does not support op {op_name}")
        return getattr(self, op_name)(op_args)

    def flush_cache(self, request: FlushCacheReq):
        if self.backend.radix_cache is not None:
            self.backend.radix_cache.flush_cache()
        return True, "Succeeded to flush cache."

    def _iter_memory_tags(self, tags: Optional[List[MemoryTag]]):
        return self.MEMORY_TAG_ORDER if tags is None else tags

    def _clear_cuda_cache(self):
        torch.cuda.empty_cache()
        gc.collect()

    def _pause_memory_tags(self, tags: Optional[List[MemoryTag]]):
        torch.cuda.synchronize()
        for tag in self._iter_memory_tags(tags):
            self.backend.model.torch_memory_saver.pause(tag=tag)
        self._clear_cuda_cache()

    def _resume_memory_tags(self, tags: Optional[List[MemoryTag]]):
        self._clear_cuda_cache()
        for tag in self._iter_memory_tags(tags):
            self.backend.model.torch_memory_saver.resume(tag=tag)

    def release_memory_occupation(self, tags: Optional[List[MemoryTag]]):
        try:
            self._pause_memory_tags(tags)
            self.flush_cache(request=None)
            return True, "Succeeded to release memory occupation."
        except Exception as e:
            self.logger.error(f"release memory occupation failed: {str(e)}")
            return False, f"release memory occupation failed: {str(e)}"

    def resume_memory_occupation(self, tags: Optional[List[MemoryTag]]):
        try:
            self._resume_memory_tags(tags)
            return True, "Succeeded to resume memory occupation."
        except Exception as e:
            self.logger.error(f"resume memory occupation failed: {str(e)}")
            return False, f"resume memory occupation failed: {str(e)}"

    def init_weights_update_group(self, request: InitWeightsUpdateGroupReq):
        assert torch.distributed.is_initialized(), "Default torch process group must be initialized"

        assert request.group_name != "", "Group name cannot be empty"
        rank_offset = request.rank_offset
        rank = rank_offset + self.backend.rank_in_dp
        world_size = request.world_size
        group_name = request.group_name
        self.logger.info(
            f"init custom process group: master_address={request.master_address}, master_port={request.master_port}, "
            f"rank_offset={rank_offset}, rank={rank}, world_size={world_size}, group_name={group_name}, "
            f" backend={request.backend}"
        )

        try:
            if group_name in self._model_update_group:
                raise ValueError(f"Process group with name {group_name} already exists.")

            self._model_update_group[group_name] = init_custom_process_group(
                backend=request.backend,
                init_method=f"tcp://{request.master_address}:{request.master_port}",
                world_size=world_size,
                rank=rank,
                group_name=group_name,
            )
            return True, "Succeeded to initialize custom process group."

        except Exception as e:
            message = f"Failed to initialize custom process group: {e}."
            self.logger.error(message)
            return False, message

    def destroy_weights_update_group(self, request: DestroyWeightsUpdateGroupReq):
        try:
            if request.group_name in self._model_update_group:
                pg = self._model_update_group.pop(request.group_name)
                torch.distributed.destroy_process_group(pg)
                return True, "Succeeded to destroy custom process group."
            else:
                return False, "The group to be destroyed does not exist."
        except Exception as e:
            message = f"Failed to destroy custom process group: {e}."
            self.logger.error(message)
            return False, message

    def update_weights_from_distributed(self, request: UpdateWeightsFromDistributedReq):
        """
        Update model weights online through the custom weight update process group.
        """

        assert request.group_name in self._model_update_group, (
            f"Group {request.group_name} not in {list(self._model_update_group.keys())}. "
            "Please call `init_weights_update_group` first."
        )

        try:
            weights = {}
            handles = []
            for name, dtype, shape in zip(request.names, request.dtypes, request.shapes):
                target_dtype = dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
                weight = torch.empty(shape, dtype=target_dtype, device="cuda")
                handles.append(
                    torch.distributed.broadcast(
                        weight,
                        src=0,
                        group=self._model_update_group[request.group_name],
                        async_op=True,
                    )
                )
                weights[name] = weight
            for handle in handles:
                handle.wait()

            self.backend.model.load_weights(weights)
            return True, "Succeeded to update parameter online from distributed."

        except Exception as e:
            error_msg = (
                f"Failed to update parameter online: {e}. "
                f"The full weights of the ModelRunner are partially updated. "
                f"Please discard the whole weights."
            )
            self.logger.error(error_msg)
            return False, error_msg

    def _update_weights_from_flattened_bucket(
        self,
        flattened_tensor_bucket_dict,
    ):
        flattened_tensor = flattened_tensor_bucket_dict["flattened_tensor"]
        metadata = flattened_tensor_bucket_dict["metadata"]

        converted_metadata = []
        for meta in metadata:
            if isinstance(meta, dict):
                converted_meta = FlattenedTensorMetadata(
                    name=meta["name"],
                    shape=meta["shape"],
                    dtype=meta["dtype"],
                    start_idx=meta["start_idx"],
                    end_idx=meta["end_idx"],
                    numel=meta["numel"],
                )
            else:
                converted_meta = FlattenedTensorMetadata(
                    name=meta.name,
                    shape=meta.shape,
                    dtype=meta.dtype,
                    start_idx=meta.start_idx,
                    end_idx=meta.end_idx,
                    numel=meta.numel,
                )
            converted_metadata.append(converted_meta)

        bucket = FlattenedTensorBucket(flattened_tensor=flattened_tensor, metadata=converted_metadata)
        reconstructed_tensors = bucket.reconstruct_tensors()

        named_tensors = {name: tensor for name, tensor in reconstructed_tensors}
        loaded, skipped = self._load_compatible_named_tensors(named_tensors)

        return (
            True,
            "Succeeded to update parameter online from flattened bucket tensor. "
            f"loaded={loaded}, skipped={skipped}.",
        )

    @staticmethod
    def _iter_named_tensors(named_tensors):
        if isinstance(named_tensors, dict):
            return named_tensors.items()
        return named_tensors

    def _get_tensor_update_skip_reason(self, items):
        if self._skip_tensor_updates_reason is not None:
            return self._skip_tensor_updates_reason

        target_config = getattr(self.backend.model, "config", {}) or {}
        target_is_moe = bool(target_config.get("num_experts") or target_config.get("moe_intermediate_size"))
        source_has_moe_experts = any(".mlp.experts." in name for name, _ in items)
        if source_has_moe_experts and not target_is_moe:
            self._skip_tensor_updates_reason = "received MoE expert weights for a non-MoE backend"
            self.logger.warning("skip tensor weight updates: %s", self._skip_tensor_updates_reason)
            return self._skip_tensor_updates_reason

        return None

    def _load_compatible_named_tensors(self, named_tensors):
        items = list(self._iter_named_tensors(named_tensors))
        skip_reason = self._get_tensor_update_skip_reason(items)
        if skip_reason is not None:
            return 0, len(items)

        def _load_range(weight_items):
            if not weight_items:
                return 0, 0
            weight_dict = dict(weight_items)
            try:
                self.backend.model.load_weights(weight_dict)
                return len(weight_items), 0
            except Exception as e:
                if len(weight_items) == 1:
                    name, tensor = weight_items[0]
                    self.logger.warning(
                        "skip incompatible tensor update %s shape=%s dtype=%s: %s",
                        name,
                        tuple(tensor.shape) if hasattr(tensor, "shape") else None,
                        getattr(tensor, "dtype", None),
                        e,
                    )
                    return 0, 1

                split_idx = len(weight_items) // 2
                left_loaded, left_skipped = _load_range(weight_items[:split_idx])
                right_loaded, right_skipped = _load_range(weight_items[split_idx:])
                return left_loaded + right_loaded, left_skipped + right_skipped

        return _load_range(items)

    def update_weights_from_tensor(self, request: UpdateWeightsFromTensorReq):
        try:
            monkey_patch_torch_reductions()
            device_module = torch.get_device_module("cuda")
            infered_device = device_module.current_device()

            if request.load_format == "flattened_bucket":
                with cuda_rebuild_device_fallback(infered_device):
                    serialized_named_tensors = MultiprocessingSerializer.deserialize(
                        request.serialized_named_tensors[self.backend.rank_in_dp]
                    )
                return self._update_weights_from_flattened_bucket(flattened_tensor_bucket_dict=serialized_named_tensors)

            def _unwrap_tensor(tensor, tp_rank, device):
                if isinstance(tensor, LocalSerializedTensor):
                    tensor = tensor.get(tp_rank)
                clone = tensor.to(device).clone()
                del tensor
                return clone

            with cuda_rebuild_device_fallback(infered_device):
                named_tensors = MultiprocessingSerializer.deserialize(
                    request.serialized_named_tensors[self.backend.rank_in_dp]
                )
                named_tensors = {
                    name: _unwrap_tensor(tensor, tp_rank=self.backend.rank_in_dp, device=infered_device)
                    for name, tensor in self._iter_named_tensors(named_tensors)
                }

            loaded, skipped = self._load_compatible_named_tensors(named_tensors)

            return True, f"Succeeded to update parameter online from tensor. loaded={loaded}, skipped={skipped}."

        except Exception as e:
            message = f"Failed to update parameter online from tensor. Reason: {e}."
            self.logger.error(message)

            return False, message

    def update_weights_from_ipc(self, request: UpdateWeightsFromIPCReq):
        try:
            from lightllm.utils.rl.bucketed_weight_transfer import BucketedWeightReceiver, get_zmq_handle

            zmq_handle = request.ipc_handle
            if isinstance(zmq_handle, dict):
                zmq_handle = zmq_handle.get(self.backend.rank_in_node, zmq_handle.get(str(self.backend.rank_in_node)))
                if zmq_handle is None:
                    raise ValueError(f"Missing ipc_handle for rank_in_node={self.backend.rank_in_node}")
            if zmq_handle in (None, "", "auto"):
                zmq_handle = get_zmq_handle()
            use_shm = request.use_shm
            recv_device = torch.device("cuda", self.backend.current_device_id)
            self.logger.debug(
                "[LightLLM] RlBackendOps.update_weights_from_ipc: request.ipc_handle=%r, "
                "resolved zmq_handle=%r, cuda_device_id=%s",
                request.ipc_handle,
                zmq_handle,
                self.backend.current_device_id,
            )

            bucketed_weight_receiver = BucketedWeightReceiver(
                zmq_handle=zmq_handle, device=recv_device, use_shm=use_shm
            )
            bucketed_weight_receiver.receive_weights(on_bucket_received=self.backend.model.load_weights)
            return True, "Succeeded to update parameter online from ipc."

        except Exception as e:
            import traceback

            traceback.print_exc()
            message = f"Failed to update parameter online from ipc. Reason: {e}."
            self.logger.error(message)

            return False, message
