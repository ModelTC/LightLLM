import pickle
import io

class RestrictedUnpickler(pickle.Unpickler):
    """
    A restricted unpickler that only allows a whitelist of classes to be deserialized.
    This mitigates the Remote Code Execution (RCE) risk associated with pickle.loads().
    """
    ALLOWED_MODULES = {
        "lightllm.server.config_server.api_http",
        "lightllm.server.router.model_infer.mode_backend.continues_batch.pd_mode.decode_node_impl.up_status",
        "lightllm.server.router.model_infer.mode_backend.pd_nixl.decode_node_impl.decode_trans_process",
        "lightllm.server.router.model_infer.mode_backend.pd_nixl.prefill_node_impl.prefill_trans_process",
        "lightllm.server.httpserver.manager",
        "lightllm.server.httpserver_for_pd_master.manager",
        "lightllm.server.router.model_infer.infer_batch",
        "lightllm.server.router.model_infer.mode_backend.pd_nixl.nixl_kv_transporter",
        "lightllm.server.router.model_infer.mode_backend.pd_nixl.decode_node_impl.up_status",
        "enum",
        "builtins",
        "collections",
        "numpy",
        "torch",
    }

    ALLOWED_CLASSES = {
        ("builtins", "list"),
        ("builtins", "dict"),
        ("builtins", "tuple"),
        ("builtins", "set"),
        ("builtins", "int"),
        ("builtins", "float"),
        ("builtins", "str"),
        ("builtins", "bool"),
        ("builtins", "NoneType"),
        ("builtins", "getattr"),
        ("collections", "deque"),
        ("enum", "Enum"),
    }

    def find_class(self, module, name):
        # Only allow specific modules or classes
        if module in self.ALLOWED_MODULES or (module, name) in self.ALLOWED_CLASSES:
            return super().find_class(module, name)
        
        # Also allow classes starting with lightllm
        if module.startswith("lightllm."):
             return super().find_class(module, name)
             
        raise pickle.UnpicklingError(f"Global '{module}.{name}' is forbidden")

def safe_pickle_loads(data):
    """Safely loads a pickled object using a restricted unpickler."""
    if data is None:
        return None
    return RestrictedUnpickler(io.BytesIO(data)).load()

def safe_pickle_dumps(obj):
    """Dumps an object using pickle.HIGHEST_PROTOCOL."""
    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
