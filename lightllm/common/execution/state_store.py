"""Allocate declared attention states within shared token and request domains."""

from collections import defaultdict

import torch

from .specs import StateView


class DenseStateStore:
    def __init__(self, declarations, format, manager):
        from lightllm.common.attention_state.full_att import GQAAttentionConfig, create_gqa_state

        encodings = {
            "plain": "None",
            "int4": "int4kv",
            "int8": "int8kv",
            "fp8_per_head": "fp8kv_sph",
            "fp8_per_tensor": "fp8kv_spt",
        }
        if format.encoding not in encodings:
            raise ValueError(f"unsupported Dense storage encoding: {format.encoding}")

        groups = defaultdict(list)
        self.resources = {}
        for spec in declarations:
            if spec.retention != "history" or len(spec.shape) != 2:
                raise ValueError("Dense storage requires two-dimensional full-history states")
            if spec.key in self.resources:
                raise ValueError(f"duplicate resource: {spec.key}")
            self.resources[spec.key] = None
            groups[spec.role, spec.shape].append(spec)
        for (_, shape), specs in groups.items():
            state = create_gqa_state(
                config=GQAAttentionConfig(
                    tuple(s.consumers for s in specs),
                    *shape,
                    format.scale_dtype if format.encoding != "plain" else format.dtype,
                    tp_rank=format.shard[0],
                    tp_world_size=format.shard[1],
                ),
                kv_type=encodings[format.encoding],
                group_size=format.group_size or 8,
                page_size=format.page_size,
                device="meta",
            )
            # Preserve existing CPU/PD identities for a uniform model. Different
            # shapes in one role need separate storage, even with the same dtype.
            count = sum(role == specs[0].role for role, _ in groups)
            resource_id = f"h{shape[0]}_d{shape[1]}" if count > 1 else None
            manager.register_state(specs[0].role, state, resource_id=resource_id)
            for spec in specs:
                self.resources[spec.key] = (spec, state)
        self.format = format

    def views(self):
        views = {}
        for key, (spec, state) in self.resources.items():
            fields = state.get_layer_view(spec.producer)
            if self.format.encoding in ("int4", "int8"):
                (k, k_scale), (v, v_scale) = fields
                fields = (k, v, k_scale, v_scale)
            elif self.format.encoding in ("fp8_per_head", "fp8_per_tensor"):
                index = state.get_storage_layer_index(spec.producer)
                k, v = (tensor.view(self.format.dtype) for tensor in fields)
                q_scale = None if state.q_scales is None else state.q_scales[index]
                fields = (k, v, state.scales[index], q_scale)
            views[key] = StateView(spec, self.format, fields)
        return views


class MLAStateStore:
    def __init__(self, declarations, format, manager):
        from lightllm.common.attention_state.full_att import MLAConfig, create_mla_state

        if format.encoding != "plain" or format.layout != "token_latent" or format.fields != ("latent",):
            raise ValueError("MLA storage requires a plain latent format")
        groups, self.resources = defaultdict(list), {}
        for spec in declarations:
            if spec.retention != "history" or len(spec.shape) != 2:
                raise ValueError("MLA requires full-history latent and rotary dimensions")
            if spec.key in self.resources:
                raise ValueError(f"duplicate resource: {spec.key}")
            self.resources[spec.key] = None
            groups[spec.role, spec.shape].append(spec)
        for (role, shape), specs in groups.items():
            state = create_mla_state(
                MLAConfig(tuple(s.consumers for s in specs), sum(shape), format.dtype, *format.shard),
                page_size=format.page_size,
                device="meta",
            )
            resource_id = "_".join(map(str, shape)) if sum(r == role for r, _ in groups) > 1 else None
            manager.register_state(role, state, resource_id=resource_id)
            for spec in specs:
                self.resources[spec.key] = (spec, state)
        self.format = format

    def views(self):
        return {
            key: StateView(spec, self.format, (state.get_layer_view(spec.producer),))
            for key, (spec, state) in self.resources.items()
        }


class PackedMLAStateStore:
    def __init__(self, declarations, format, manager):
        from lightllm.common.attention_state.full_att import PackedMLALogicalConfig, create_packed_mla_state

        import torch

        encodings = {
            "plain": "packed_fp16" if format.dtype == torch.float16 else "packed_bf16",
            "fp8_group128": "packed_group128_fp8",
            "fp8_group64_e8m0": "packed_group64_e8m0",
        }
        pooled = format.layout == "token_pooled_nope"
        if format.encoding not in encodings or format.layout not in ("token_packed_mla", "token_pooled_nope"):
            raise ValueError("packed MLA requires an explicit latent and indexer format")
        groups, self.resources = defaultdict(list), {}
        for spec in declarations:
            if spec.retention != "history" or len(spec.shape) != (2 if pooled else 3) or spec.key in self.resources:
                raise ValueError("packed MLA requires exclusive latent, rotary and indexer declarations")
            self.resources[spec.key] = None
            groups[spec.role, spec.shape].append(spec)
        for (role, shape), specs in groups.items():
            latent, rope, indexer = (shape[0], 0, shape[1]) if pooled else shape
            state = create_packed_mla_state(
                PackedMLALogicalConfig(
                    tuple(s.consumers for s in specs),
                    latent + rope,
                    encodings[format.encoding],
                    indexer,
                    rope,
                    *format.shard,
                ),
                page_size=format.page_size,
                device="meta",
            )
            resource_id = "_".join(map(str, shape)) if sum(r == role for r, _ in groups) > 1 else None
            manager.register_state(role, state, resource_id=resource_id)
            for spec in specs:
                self.resources[spec.key] = (spec, state)
        self.format = format

    def views(self):
        views = {}
        for key, (spec, state) in self.resources.items():
            native = state.get_layer_view(spec.producer)
            views[key] = StateView(spec, self.format, (native.kv, native.indexer_k))
        return views


class ConvSSMStateStore:
    def __init__(self, declarations, format, manager, request_capacity, device, kind="gdn"):
        from lightllm.common.attention_state.bounded_att import ConvSSMStateConfig, create_gdn_state, create_kda_state

        create_state = {"gdn": create_gdn_state, "kda": create_kda_state}[kind]

        speculative = format.candidate_slots > 1
        expected_fields = ("conv", "ssm", "accepted_index") if speculative else ("conv", "ssm")
        if format.encoding != "conv_ssm" or format.fields != expected_fields:
            raise ValueError("recurrent storage requires explicit Conv, SSM and candidate formats")
        if len(format.field_dtypes) != len(expected_fields) or (speculative and format.field_dtypes[-1] != torch.int32):
            raise ValueError("unsupported recurrent candidate storage")
        groups, self.resources = defaultdict(list), {}
        for spec in declarations:
            if spec.retention != "recurrent" or len(spec.shape) != 5 or spec.consumers != (spec.producer,):
                raise ValueError("Conv/SSM storage requires an exclusive recurrent state for each producer")
            if spec.key in self.resources:
                raise ValueError(f"duplicate resource: {spec.key}")
            self.resources[spec.key] = None
            groups[spec.role, spec.shape].append(spec)
        for (role, shape), specs in groups.items():
            state = create_state(
                ConvSSMStateConfig(
                    tuple(s.producer for s in specs),
                    *shape,
                    *format.field_dtypes[:2],
                    tp_rank=format.shard[0],
                    tp_world_size=format.shard[1],
                ),
                request_capacity=request_capacity,
                mtp_step=format.candidate_slots - 1,
                device=device,
            )
            resource_id = "_".join(map(str, shape)) if sum(r == role for r, _ in groups) > 1 else None
            manager.register_state(role, state, resource_id=resource_id)
            for spec in specs:
                self.resources[spec.key] = (spec, state)
        self.format = format

    def views(self):
        return {
            key: StateView(
                spec,
                self.format,
                state.get_layer_view(spec.producer)
                + ((state.mtp_state_index,) if self.format.candidate_slots > 1 else ()),
            )
            for key, (spec, state) in self.resources.items()
        }


class IndexerWindowStateStore:
    def __init__(self, declarations, format, manager, request_capacity, device):
        from lightllm.common.attention_state.bounded_att import (
            WindowCompressorStateConfig,
            create_window_compressor_state,
        )

        if format.encoding != "plain" or format.layout != "request_indexer_window" or format.fields != ("tail",):
            raise ValueError("indexer window requires a separate candidate/gate tail format")
        groups, self.resources = defaultdict(list), {}
        for spec in declarations:
            if spec.retention != "window" or len(spec.shape) != 3 or spec.consumers != (spec.producer,):
                raise ValueError("indexer window must have an exclusive producer and explicit window/stride/dimension")
            if spec.key in self.resources:
                raise ValueError(f"duplicate resource: {spec.key}")
            self.resources[spec.key] = None
            groups[spec.role, spec.shape].append(spec)
        for (role, shape), specs in groups.items():
            state = create_window_compressor_state(
                WindowCompressorStateConfig(
                    tuple(s.producer for s in specs),
                    *shape,
                    target="indexer",
                    dtype=format.dtype,
                    tp_rank=format.shard[0],
                    tp_world_size=format.shard[1],
                ),
                request_capacity=request_capacity,
                mtp_step=format.candidate_slots - 1,
                device=device,
            )
            manager.register_state(role, state)
            for spec in specs:
                self.resources[spec.key] = (spec, state)
        self.format = format

    def views(self):
        return {
            key: StateView(spec, self.format, (state.get_layer_view(spec.producer),))
            for key, (spec, state) in self.resources.items()
        }
