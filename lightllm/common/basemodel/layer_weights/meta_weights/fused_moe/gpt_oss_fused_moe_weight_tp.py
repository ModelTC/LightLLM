import threading

import torch

from lightllm.common.basemodel.layer_weights.meta_weights.base_weight import BaseWeightTpl


class GPTOSSFusedMoeWeightTP(BaseWeightTpl):
    """Load GPT-OSS expert tensors; execution is bound separately by the Planner."""

    def __init__(self, weight_prefix, n_routed_experts, hidden_size, moe_intermediate_size, data_type):
        super().__init__(data_type=data_type)
        if moe_intermediate_size % self.tp_world_size_:
            raise ValueError("GPT-OSS expert width requires complete TP shards")
        self.weight_prefix = weight_prefix
        self.n_routed_experts = n_routed_experts
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.split_inter_size = moe_intermediate_size // self.tp_world_size_
        self.enable_ep_moe = False
        self.lock = threading.Lock()
        self.pending = {}
        self._create_weight()

    def _create_weight(self):
        count, width, hidden = self.n_routed_experts, self.split_inter_size, self.hidden_size
        self.fields = {
            name: torch.empty(shape, dtype=self.data_type_, device=self.device_id_)
            for name, shape in (
                ("gate_up", (count, width * 2, hidden)),
                ("down", (count, hidden, width)),
                ("gate_up_bias", (count, width * 2)),
                ("down_bias", (count, hidden)),
            )
        }
        self.loaded = set()

    @staticmethod
    def _unpack(blocks, scales, dtype):
        if blocks.dtype != torch.uint8 or scales.dtype != torch.uint8:
            raise ValueError("MXFP4 blocks and E8M0 scales must be uint8")
        if blocks.shape[:-1] != scales.shape or blocks.shape[-1] != 16:
            raise ValueError("MXFP4 requires one scale per 32 packed values")
        lut = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
            dtype=torch.float32,
            device=blocks.device,
        )
        shape = (*blocks.shape[:-2], blocks.shape[-2] * 32)
        blocks, scales = blocks.reshape(-1, 16), scales.reshape(-1, 1)
        output = torch.empty((blocks.shape[0], 32), dtype=dtype, device=blocks.device)
        for start in range(0, blocks.shape[0], 32768):
            chunk = blocks[start : start + 32768]
            values = torch.stack((lut[(chunk & 15).long()], lut[(chunk >> 4).long()]), dim=-1).flatten(-2)
            output[start : start + 32768].copy_(torch.ldexp(values, scales[start : start + 32768].int() - 127))
        return output.reshape(shape)

    def load_hf_weights(self, weights):
        with self.lock:
            start = self.tp_rank_ * self.split_inter_size
            stop = start + self.split_inter_size
            for field, checkpoint in (("gate_up", "gate_up_proj"), ("down", "down_proj")):
                name = f"{self.weight_prefix}.{checkpoint}"
                tensor = weights.get(name)
                if tensor is not None:
                    shape = (
                        (self.n_routed_experts, self.hidden_size, 2 * self.moe_intermediate_size)
                        if field == "gate_up"
                        else (self.n_routed_experts, self.moe_intermediate_size, self.hidden_size)
                    )
                    if tensor.shape != shape:
                        raise ValueError(f"GPT-OSS checkpoint shape mismatch: {name}")
                    tensor = tensor[:, :, start * 2 : stop * 2] if field == "gate_up" else tensor[:, start:stop]
                else:
                    for suffix in ("_blocks", "_scales"):
                        if name + suffix in weights:
                            self.pending[name + suffix] = weights[name + suffix]
                    if name + "_blocks" in self.pending and name + "_scales" in self.pending:
                        blocks, scales = (self.pending.pop(name + suffix) for suffix in ("_blocks", "_scales"))
                        shape = (
                            (self.n_routed_experts, self.hidden_size, self.moe_intermediate_size // 16, 16)
                            if field == "gate_up"
                            else (self.n_routed_experts, self.moe_intermediate_size, self.hidden_size // 32, 16)
                        )
                        if blocks.shape != shape or scales.shape != blocks.shape[:-1]:
                            raise ValueError(f"GPT-OSS MXFP4 checkpoint shape mismatch: {name}")
                        if field == "gate_up":
                            if start % 16 or stop % 16:
                                raise ValueError("MXFP4 gate/up TP shards must align to 32 packed values")
                            blocks, scales = (
                                blocks[:, :, start // 16 : stop // 16],
                                scales[:, :, start // 16 : stop // 16],
                            )
                        else:
                            blocks, scales = blocks[:, start:stop], scales[:, start:stop]
                        device = self.fields[field].device
                        tensor = self._unpack(blocks.to(device), scales.to(device), self.data_type_)
                if tensor is not None:
                    self.fields[field].copy_(tensor.transpose(1, 2))
                    self.loaded.add(field)
                bias = weights.get(name + "_bias")
                if bias is not None:
                    size = 2 * self.moe_intermediate_size if field == "gate_up" else self.hidden_size
                    if bias.shape != (self.n_routed_experts, size):
                        raise ValueError(f"GPT-OSS bias shape mismatch: {name}")
                    bias = bias[:, start * 2 : stop * 2] if field == "gate_up" else bias / self.tp_world_size_
                    self.fields[field + "_bias"].copy_(bias)
                    self.loaded.add(field + "_bias")

    def verify_load(self):
        return self.loaded == set(self.fields) and not self.pending

    def native_moe_view(self):
        from lightllm.common.execution.specs import FormatSpec

        if not self.verify_load():
            raise ValueError("GPT-OSS expert weights and biases must be fully loaded before binding")
        return (
            FormatSpec(
                "plain",
                self.data_type_,
                "expert_output_input_interleaved",
                tuple(self.fields),
                1,
                (self.tp_rank_, self.tp_world_size_),
            ),
            self.fields.copy(),
        )
