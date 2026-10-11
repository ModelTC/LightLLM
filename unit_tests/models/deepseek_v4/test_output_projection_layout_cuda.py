import unittest

import torch


@unittest.skipUnless(torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9, "requires CUDA SM90+")
class TestOutputProjectionLayoutCuda(unittest.TestCase):
    def test_strided_bmm_matches_parent_and_cuda_graph_replay(self):
        # Small BF16 tensors; this validates the exact layout used by _get_o.
        torch.manual_seed(23)
        tokens, groups, in_dim, out_dim = 3, 2, 16, 12
        value = torch.randn((tokens, groups, in_dim), device="cuda", dtype=torch.bfloat16)
        weight = torch.randn((groups, in_dim, out_dim), device="cuda", dtype=torch.bfloat16)

        parent = torch.bmm(value.transpose(0, 1).contiguous(), weight).transpose(0, 1).reshape(tokens, -1)
        out = torch.empty((tokens, groups, out_dim), device="cuda", dtype=torch.bfloat16)
        result = torch.bmm(value.transpose(0, 1), weight, out=out.transpose(0, 1))
        torch.cuda.synchronize()
        self.assertEqual(result.data_ptr(), out.transpose(0, 1).data_ptr())
        self.assertTrue(torch.equal(out.reshape(tokens, -1), parent))

        # Replay a static-shape capture using the same strided out view.
        static_value = value.clone()
        static_out = torch.empty_like(out)
        torch.bmm(static_value.transpose(0, 1), weight, out=static_out.transpose(0, 1))
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            torch.bmm(static_value.transpose(0, 1), weight, out=static_out.transpose(0, 1))
        static_value.copy_(value)
        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(static_out.reshape(tokens, -1), parent))
