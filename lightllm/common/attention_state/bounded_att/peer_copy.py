from dataclasses import asdict, dataclass

from .copy_kernel import copy_bounded_att_tensors


@dataclass(frozen=True)
class BoundedPeerCopy:
    source: object
    destination: object
    runtime_fields: tuple
    snapshot_fields: tuple

    @classmethod
    def bind(cls, source, destination, source_row=0, destination_row=0, layers=None):
        src_config, dst_config = asdict(source.config), asdict(destination.config)
        src_layers, dst_layers = src_config.pop("logical_layer_ids"), dst_config.pop("logical_layer_ids")
        if layers is None:
            layers = len(dst_layers)
        if (
            source.resource_kind != destination.resource_kind
            or source.get_state_type() != destination.get_state_type()
            or src_config != dst_config
            or min(source_row, destination_row) < 0
            or layers <= 0
            or source_row + layers > len(src_layers)
            or destination_row + layers > len(dst_layers)
            or src_layers[source_row : source_row + layers] != dst_layers[destination_row : destination_row + layers]
        ):
            raise ValueError("incompatible DP bounded resource, format or shard")

        def bind_fields(src, dst):
            if tuple(name for name, _ in src.fields) != tuple(name for name, _ in dst.fields):
                raise ValueError("DP bounded fields differ")
            pairs = []
            for (name, field), (_, other) in zip(src.fields, dst.fields):
                if (
                    field.axes != other.axes
                    or field.axes[1] != "layer"
                    or field.tensor.dtype != other.tensor.dtype
                    or field.tensor.shape[1:] != other.tensor.shape[1:]
                ):
                    raise ValueError(f"DP bounded field geometry differs: {name}")
                pairs.append((name, field.tensor, other.tensor))
            return tuple(pairs)

        src = source.get_runtime_transfer_view().narrow_layers(source_row, layers)
        dst = destination.get_runtime_transfer_view().narrow_layers(destination_row, layers)
        runtime = bind_fields(src, dst)
        snapshots = bind_fields(
            source.get_snapshot_transfer_view("big").narrow_layers(source_row, layers),
            destination.get_snapshot_transfer_view("big").narrow_layers(destination_row, layers),
        )
        return cls(src, dst, runtime, snapshots)

    def _copy(self, fields, source_indices, destination_indices, offsets=None):
        for _, source, destination in fields:
            if any(not 0 <= i < len(source) for i in source_indices) or any(
                not 0 <= i < len(destination) for i in destination_indices
            ):
                raise ValueError("DP state index lies outside its allocation")
        for name, source, destination in fields:
            copies = [(source[src], destination[dst]) for src, dst in zip(source_indices, destination_indices)]
            copy_bounded_att_tensors(copies, source_byte_offsets=None if offsets is None else offsets.get(name))

    def copy_runtime(self, source_requests, destination_requests):
        if len(source_requests) != len(destination_requests):
            raise ValueError("source and destination request counts differ")
        if not source_requests:
            return
        device = self.runtime_fields[0][2].device
        offsets = self.source.offsets(source_requests, device)
        self._copy(self.runtime_fields, source_requests, destination_requests, offsets)
        self.destination.finish_restore(destination_requests)

    def copy_pages(self, source_pages, destination_pages):
        if len(source_pages) != len(destination_pages):
            raise ValueError("source and destination page counts differ")
        if not source_pages or not self.snapshot_fields:
            return
        self._copy(self.snapshot_fields, source_pages, destination_pages)
