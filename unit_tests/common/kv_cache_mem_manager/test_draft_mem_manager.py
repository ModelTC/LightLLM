import pytest
import torch

from lightllm.common.kv_cache_mem_manager.draft_mem_manager import DraftMemoryManagerView


class _RecordingOperator:
    def __init__(self):
        self.calls = []

    def copy_kv_to_mem_manager(self, layer_index, mem_index, kv):
        self.calls.append((layer_index, mem_index, kv.clone()))


class _Owner:
    def __init__(self):
        self.head_num = 1
        self.head_dim = 256
        self.layer_num = 2
        self.kv_buffer = torch.arange(2 * 3 * 2 * 256, dtype=torch.float32).view(2, 3, 2, 256)
        self.scales = torch.tensor([[1.0, 10.0], [2.0, 20.0]])
        self.q_scales = torch.tensor([[3.0], [4.0]])
        self.operator = _RecordingOperator()
        self.att_layers = []
        self.allocator = object()
        self.calibration_owner_marker = "owner"
        self.q_observations = []

    def get_att_input_params(self, layer_index):
        self.att_layers.append(layer_index)
        return self.kv_buffer[layer_index, :, :1, :], self.kv_buffer[layer_index, :, 1:, :]

    def update_q_calibration_data(self, q, layer_index, valid_rows=None):
        # Match the owner collector's grouping rule: one target KV head owns
        # all narrower draft heads at the same cache width.
        if valid_rows is not None:
            q = q[valid_rows]
        self.q_observations.append((layer_index, q.reshape(q.shape[0], self.head_num, -1).abs().amax(dim=(0, 2))))


def test_draft_view_reinterprets_equal_width_cache_and_preserves_owner_collection_geometry():
    owner = _Owner()
    view = DraftMemoryManagerView(owner, head_num=2, head_dim=128)

    assert view.kv_buffer.data_ptr() == owner.kv_buffer.data_ptr()
    assert view.kv_buffer.shape == (2, 3, 4, 128)
    assert view.allocator is owner.allocator
    assert view.calibration_owner_marker == "owner"
    assert view.scales.tolist() == [[1.0, 1.0, 10.0, 10.0], [2.0, 2.0, 20.0, 20.0]]
    assert view.q_scales.tolist() == [[3.0, 3.0], [4.0, 4.0]]

    k, v = view.get_att_input_params(1)
    assert owner.att_layers == [1]
    assert k.shape == v.shape == (3, 2, 128)
    assert torch.equal(k.flatten(1), owner.kv_buffer[1, :, :1, :].flatten(1))
    assert torch.equal(v.flatten(1), owner.kv_buffer[1, :, 1:, :].flatten(1))

    draft_kv = torch.arange(2 * 4 * 128, dtype=torch.float32).view(2, 4, 128)
    mem_index = torch.tensor([0, 2])
    view.operator.copy_kv_to_mem_manager(1, mem_index, draft_kv)
    layer, actual_index, canonical = owner.operator.calls.pop()
    assert layer == 1
    assert actual_index is mem_index
    assert canonical.shape == (2, 2, 256)
    assert torch.equal(canonical.flatten(1), draft_kv.flatten(1))

    q = torch.tensor([[[1.0] * 128, [7.0] * 128], [[2.0] * 128, [3.0] * 128]])
    view.update_q_calibration_data(q, 1, valid_rows=torch.tensor([True, False]))
    assert owner.q_observations[0][0] == 1
    assert owner.q_observations[0][1].tolist() == [7.0]


@pytest.mark.parametrize("heads,dim,match", [(3, 128, "width"), (1, 512, "width")])
def test_draft_view_rejects_incompatible_width(heads, dim, match):
    with pytest.raises(ValueError, match=match):
        DraftMemoryManagerView(_Owner(), head_num=heads, head_dim=dim)


def test_draft_view_rejects_non_integral_head_multiple_at_equal_width():
    owner = _Owner()
    owner.head_num = 2
    owner.head_dim = 3
    owner.kv_buffer = torch.empty((2, 3, 4, 3))
    owner.scales = torch.ones((2, 4))
    owner.q_scales = torch.ones((2, 2))
    with pytest.raises(ValueError, match="integer multiple"):
        DraftMemoryManagerView(owner, head_num=3, head_dim=2)


def test_draft_view_rejects_bad_write_shape():
    view = DraftMemoryManagerView(_Owner(), head_num=2, head_dim=128)
    with pytest.raises(ValueError, match="draft KV shape"):
        view.operator.copy_kv_to_mem_manager(0, torch.tensor([0]), torch.empty(1, 2, 128))
