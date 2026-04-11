import pytest
import torch

from lightllm.common.mamba_cache_mem_manager.cpu_cache_manager import CpuMambaCacheManager

# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------
POOL_SIZE = 8
LAYER_NUM = 2
CONV_KERNEL_SIZE = 4
NUM_LINEAR_K_HEADS = 2
NUM_LINEAR_V_HEADS = 2
HEAD_LINEAR_K_DIM = 4
HEAD_LINEAR_V_DIM = 4


def _make_manager(size=POOL_SIZE):
    return CpuMambaCacheManager(
        size=size,
        layer_num=LAYER_NUM,
        conv_state_dtype=torch.float32,
        ssm_state_dtype=torch.float32,
        conv_kernel_size=CONV_KERNEL_SIZE,
        num_linear_k_heads=NUM_LINEAR_K_HEADS,
        num_linear_v_heads=NUM_LINEAR_V_HEADS,
        head_linear_k_dim=HEAD_LINEAR_K_DIM,
        head_linear_v_dim=HEAD_LINEAR_V_DIM,
    )


# ---------------------------------------------------------------------------
# TestCpuMambaCacheManagerAllocFree
# ---------------------------------------------------------------------------
class TestCpuMambaCacheManagerAllocFree:
    def test_alloc_returns_correct_count(self):
        mgr = _make_manager()
        slots = mgr.alloc(3)
        assert len(slots) == 3
        assert mgr.can_use_mem_size == POOL_SIZE - 3

    def test_free_restores_capacity(self):
        mgr = _make_manager()
        slots = mgr.alloc(3)
        mgr.free(slots.tolist())
        assert mgr.can_use_mem_size == POOL_SIZE

    def test_alloc_exhaustion_raises(self):
        mgr = _make_manager(size=4)
        mgr.alloc(4)
        with pytest.raises(AssertionError):
            mgr.alloc(1)

    def test_alloc_indices_are_unique(self):
        mgr = _make_manager()
        slots = mgr.alloc(POOL_SIZE)
        unique = set(slots.tolist())
        assert len(unique) == POOL_SIZE

    def test_alloc_free_alloc_reuse(self):
        mgr = _make_manager()
        slots1 = mgr.alloc(3)
        freed_indices = slots1.tolist()
        mgr.free(slots1)
        slots2 = mgr.alloc(3)
        # The reallocated indices should match the freed ones (stack LIFO order)
        assert sorted(slots2.tolist()) == sorted(freed_indices)

    def test_free_all(self):
        mgr = _make_manager()
        mgr.alloc(5)
        assert mgr.can_use_mem_size == POOL_SIZE - 5
        mgr.free_all()
        assert mgr.can_use_mem_size == POOL_SIZE
        # Should be able to alloc the full pool again
        slots = mgr.alloc(POOL_SIZE)
        assert len(slots) == POOL_SIZE

    def test_cpu_buffer_shape_is_slot_first(self):
        """CPU buffers must have shape (size, layer_num, *state_shape)."""
        mgr = _make_manager()
        assert mgr.conv_state_buffer.shape[0] == POOL_SIZE
        assert mgr.conv_state_buffer.shape[1] == LAYER_NUM
        assert mgr.ssm_state_buffer.shape[0] == POOL_SIZE
        assert mgr.ssm_state_buffer.shape[1] == LAYER_NUM


# ---------------------------------------------------------------------------
# TestCpuMambaCacheTransfer  (requires CUDA)
# ---------------------------------------------------------------------------
_has_cuda = torch.cuda.is_available()


@pytest.mark.skipif(not _has_cuda, reason="CUDA not available")
class TestCpuMambaCacheTransfer:
    def _setup_mgr(self, size=POOL_SIZE):
        """Create manager + GPU buffers and bind them via set_gpu_buffers."""
        mgr = _make_manager(size)
        conv_shape = (mgr.layer_num, mgr.size, *mgr.conv_state_shape)
        ssm_shape = (mgr.layer_num, mgr.size, *mgr.ssm_state_shape)
        gpu_conv = torch.randn(conv_shape, dtype=mgr.conv_state_dtype, device="cuda")
        gpu_ssm = torch.randn(ssm_shape, dtype=mgr.ssm_state_dtype, device="cuda")
        mgr.set_gpu_buffers(gpu_conv, gpu_ssm)
        return mgr, gpu_conv, gpu_ssm

    def test_offload_and_load_roundtrip(self):
        mgr, gpu_conv, gpu_ssm = self._setup_mgr()

        gpu_indices = [0, 3]
        cpu_slots = mgr.alloc(2)
        cpu_slot_list = cpu_slots.tolist()

        expected_conv = gpu_conv[:, gpu_indices, ...].clone()
        expected_ssm = gpu_ssm[:, gpu_indices, ...].clone()

        mgr.offload_to_cpu(gpu_indices, cpu_slot_list)
        mgr.sync_transfer()

        for g in gpu_indices:
            gpu_conv[:, g, ...] = 0
            gpu_ssm[:, g, ...] = 0

        mgr.load_to_gpu(cpu_slot_list, gpu_indices)

        for i, g in enumerate(gpu_indices):
            torch.testing.assert_close(gpu_conv[:, g, ...], expected_conv[:, i, ...])
            torch.testing.assert_close(gpu_ssm[:, g, ...], expected_ssm[:, i, ...])

    def test_batched_offload_and_load_multiple_slots(self):
        """Verify transfer works with multiple non-contiguous GPU indices."""
        mgr, gpu_conv, gpu_ssm = self._setup_mgr()

        gpu_indices = [1, 3, 5, 7]
        cpu_slots = mgr.alloc(4)
        cpu_slot_list = cpu_slots.tolist()

        expected_conv = gpu_conv[:, gpu_indices, ...].clone()
        expected_ssm = gpu_ssm[:, gpu_indices, ...].clone()

        mgr.offload_to_cpu(gpu_indices, cpu_slot_list)
        mgr.sync_transfer()

        for i, c in enumerate(cpu_slot_list):
            torch.testing.assert_close(mgr.conv_state_buffer[c], expected_conv[:, i, ...].cpu())
            torch.testing.assert_close(mgr.ssm_state_buffer[c], expected_ssm[:, i, ...].cpu())

        for g in gpu_indices:
            gpu_conv[:, g, ...] = 0
            gpu_ssm[:, g, ...] = 0

        mgr.load_to_gpu(cpu_slot_list, gpu_indices)

        for i, g in enumerate(gpu_indices):
            torch.testing.assert_close(gpu_conv[:, g, ...], expected_conv[:, i, ...])
            torch.testing.assert_close(gpu_ssm[:, g, ...], expected_ssm[:, i, ...])

    def test_direct_offload_roundtrip(self):
        """Verify the direct=True path (no snapshot) produces correct data."""
        mgr, gpu_conv, gpu_ssm = self._setup_mgr()

        gpu_indices = [0, 2, 4, 6]
        cpu_slots = mgr.alloc(4)
        cpu_slot_list = cpu_slots.tolist()

        expected_conv = gpu_conv[:, gpu_indices, ...].clone()
        expected_ssm = gpu_ssm[:, gpu_indices, ...].clone()

        mgr.offload_to_cpu(gpu_indices, cpu_slot_list, direct=True)
        mgr.sync_transfer()

        for i, c in enumerate(cpu_slot_list):
            torch.testing.assert_close(mgr.conv_state_buffer[c], expected_conv[:, i, ...].cpu())
            torch.testing.assert_close(mgr.ssm_state_buffer[c], expected_ssm[:, i, ...].cpu())

        for g in gpu_indices:
            gpu_conv[:, g, ...] = 0
            gpu_ssm[:, g, ...] = 0

        mgr.load_to_gpu(cpu_slot_list, gpu_indices)

        for i, g in enumerate(gpu_indices):
            torch.testing.assert_close(gpu_conv[:, g, ...], expected_conv[:, i, ...])
            torch.testing.assert_close(gpu_ssm[:, g, ...], expected_ssm[:, i, ...])

    def test_offload_does_not_corrupt_other_slots(self):
        mgr, gpu_conv, gpu_ssm = self._setup_mgr()

        orig_conv_5 = gpu_conv[:, 5, ...].clone()
        orig_ssm_5 = gpu_ssm[:, 5, ...].clone()

        cpu_slots = mgr.alloc(1)
        mgr.offload_to_cpu([2], cpu_slots.tolist())
        mgr.sync_transfer()

        torch.testing.assert_close(gpu_conv[:, 5, ...], orig_conv_5)
        torch.testing.assert_close(gpu_ssm[:, 5, ...], orig_ssm_5)
