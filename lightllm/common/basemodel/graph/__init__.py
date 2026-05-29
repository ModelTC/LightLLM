from lightllm.common.basemodel.graph.acl_graph import AclGraph
from lightllm.common.basemodel.graph.base.decode_graph import DecodeGraph
from lightllm.common.basemodel.graph.cuda_graph import CudaGraph

DECODE_GRAPH_MAP = {
    "cuda": CudaGraph,
    "musa": CudaGraph,
    "ascend": AclGraph,
}

DecodeGraph.PLATFORM_CLASS_MAP = DECODE_GRAPH_MAP

__all__ = ["DecodeGraph"]
