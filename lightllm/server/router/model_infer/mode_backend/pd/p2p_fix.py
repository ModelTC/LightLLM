# mypy: allow-untyped-defs
import torch
import torch.utils.hooks
from torch._namedtensor_internals import check_serializing_named_tensor
from torch.multiprocessing.reductions import (
    StorageWeakRef,
    reduce_nested_tensor,
    reduce_sparse_tensor,
    rebuild_tensor,
    shared_cache,
    storage_from_cache,
)


def p2p_fix_rebuild_cuda_tensor(
    tensor_cls,
    tensor_size,
    tensor_stride,
    tensor_offset,
    storage_cls,
    dtype,
    storage_device,
    storage_handle,
    storage_size_bytes,
    storage_offset_bytes,
    requires_grad,
    ref_counter_handle,
    ref_counter_offset,
    event_handle,
    event_sync_required,
):
    storage_device = torch.cuda.current_device()
    if storage_handle is None or storage_size_bytes == 0:
        storage = storage_cls(0, dtype=dtype, device=storage_device, _internal=True)
    else:
        storage = storage_from_cache(storage_cls, (storage_handle, storage_offset_bytes))
        if storage is None:
            torch.cuda._lazy_init()
            storage = storage_cls._new_shared_cuda(
                storage_device,
                storage_handle,
                storage_size_bytes,
                storage_offset_bytes,
                ref_counter_handle,
                ref_counter_offset,
                event_handle,
                event_sync_required,
            )
            shared_cache[(storage_handle, storage_offset_bytes)] = StorageWeakRef(storage)
        else:
            storage_cls._release_ipc_counter(ref_counter_handle, ref_counter_offset, device=storage_device)

    _storage = storage if isinstance(storage, torch.UntypedStorage) else storage._untyped_storage
    tensor = torch._utils._rebuild_tensor(
        torch.storage.TypedStorage(wrap_storage=_storage, dtype=dtype, _internal=True),
        tensor_offset,
        tensor_size,
        tensor_stride,
    )

    if tensor_cls == torch.nn.parameter.Parameter:
        tensor = torch.nn.parameter.Parameter(tensor, requires_grad=requires_grad)
    else:
        tensor.requires_grad = requires_grad

    return tensor


def reduce_tensor(tensor):
    if tensor.requires_grad and not tensor.is_leaf:
        raise RuntimeError(
            "Cowardly refusing to serialize non-leaf tensor which requires_grad, "
            "since autograd does not support crossing process boundaries. "
            "If you just want to transfer the data, call detach() on the tensor "
            "before serializing."
        )

    check_serializing_named_tensor(tensor)
    torch.utils.hooks.warn_if_has_hooks(tensor)

    from torch.nested._internal.nested_tensor import NestedTensor

    if tensor.is_nested and not isinstance(tensor, NestedTensor):
        return reduce_nested_tensor(tensor)

    if tensor.layout in {
        torch.sparse_coo,
        torch.sparse_csr,
        torch.sparse_bsr,
        torch.sparse_csc,
        torch.sparse_bsc,
    }:
        return reduce_sparse_tensor(tensor)

    storage = tensor._typed_storage()

    if storage._untyped_storage.device.type == "cuda":
        from lightllm.server.router.model_infer.mode_backend.pd.p2p_fix import p2p_fix_rebuild_cuda_tensor

        (
            device,
            handle,
            storage_size_bytes,
            storage_offset_bytes,
            ref_counter_handle,
            ref_counter_offset,
            event_handle,
            event_sync_required,
        ) = storage._share_cuda_()
        tensor_offset = tensor.storage_offset()
        shared_cache[handle] = StorageWeakRef(storage)
        return (
            p2p_fix_rebuild_cuda_tensor,
            (
                type(tensor),
                tensor.size(),
                tensor.stride(),
                tensor_offset,
                type(storage),
                tensor.dtype,
                device,
                handle,
                storage_size_bytes,
                storage_offset_bytes,
                tensor.requires_grad,
                ref_counter_handle,
                ref_counter_offset,
                event_handle,
                event_sync_required,
            ),
        )

    metadata = (
        tensor.storage_offset(),
        tensor.size(),
        tensor.stride(),
        tensor.requires_grad,
    )
    return (rebuild_tensor, (type(tensor), storage, metadata))
