from lightllm.common.basemodel.graph.base.decode_graph import DecodeGraph, register_decode_graph


@register_decode_graph("cuda", "musa", "maca")
class CudaGraph(DecodeGraph):
    pass


__all__ = ["CudaGraph"]
