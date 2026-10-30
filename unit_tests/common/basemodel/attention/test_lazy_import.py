"""Check import isolation and selection without loading optional GPU dependencies."""

import importlib
import importlib.abc
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[4]
ATTENTION = "lightllm.common.basemodel.attention"


class BackendImports(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Reject unselected implementations, and supply lightweight selected backends."""

    def __init__(self, package=ATTENTION):
        self.package = package
        self.allowed = {}
        self.imported = []

    def allow(self, suffix, class_name, **attributes):
        module = ModuleType(f"{self.package}.{suffix}")
        setattr(module, class_name, type(class_name, (), {}))
        module.__dict__.update(attributes)
        self.allowed[module.__name__] = module
        return getattr(module, class_name)

    def find_spec(self, fullname, path=None, target=None):
        suffix = fullname.removeprefix(f"{self.package}.")
        if not fullname.startswith(f"{self.package}.") or "." not in suffix:
            return None
        if suffix.split(".")[0] not in {"triton", "fa3", "flashinfer", "nsa", "linear", "sdpa", "xformers"}:
            return None
        self.imported.append(fullname)
        if fullname not in self.allowed:
            raise AssertionError(f"Unexpected attention backend import: {fullname}")
        return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec):
        return self.allowed[spec.name]

    def exec_module(self, module):
        pass


class TestLazyAttentionImports(unittest.TestCase):
    def setUp(self):
        # Isolate attention from basemodel's unrelated GPU initialization.
        modules_patch = patch.dict(sys.modules)
        modules_patch.start()
        self.addCleanup(modules_patch.stop)
        for name in list(sys.modules):
            if name == ATTENTION or name.startswith(f"{ATTENTION}."):
                del sys.modules[name]
        for name in ("lightllm", "lightllm.common", "lightllm.common.basemodel", "lightllm.utils"):
            module = ModuleType(name)
            module.__path__ = [str(ROOT.joinpath(*name.split(".")))]
            sys.modules[name] = module

        self.args = SimpleNamespace(
            llm_kv_type="None",
            llm_prefill_att_backend=["triton"],
            llm_decode_att_backend=["triton"],
            enable_ep_moe=False,
            mtp_step=0,
        )
        self.validate = Mock(return_value=True)
        dependencies = {
            f"{ATTENTION}.base_att": {
                name: type(name, (), {})
                for name in ("BaseAttBackend", "BasePrefillAttState", "BaseDecodeAttState", "AttControl")
            },
            "lightllm.utils.envs_utils": {"get_env_start_args": lambda: self.args},
            "lightllm.utils.backend_validator": {"validate": self.validate},
            "lightllm.utils.log_utils": {"init_logger": lambda name: Mock()},
        }
        for name, attributes in dependencies.items():
            module = ModuleType(name)
            module.__dict__.update(attributes)
            sys.modules[name] = module

        self.imports = BackendImports()
        sys.meta_path.insert(0, self.imports)
        self.addCleanup(sys.meta_path.remove, self.imports)
        self.attention = importlib.import_module(ATTENTION)

    def test_importing_selection_apis_loads_no_implementations(self):
        importlib.import_module(f"{ATTENTION}.create_linear_utils")
        importlib.import_module(f"{ATTENTION}.linear")
        importlib.import_module(f"{ATTENTION}.nsa")
        self.assertEqual(self.imports.imported, [])

    def test_explicit_regular_selection_skips_neo(self):
        backend = self.imports.allow("fa3.fp", "Fa3AttBackend")
        self.args.llm_prefill_att_backend = ["fa3"]

        self.assertIs(self.attention.get_prefill_att_backend_class(), backend)
        self.assertEqual(self.imports.imported, [f"{ATTENTION}.fa3.fp"])
        self.validate.assert_not_called()

    def test_auto_selection_does_not_import_failed_candidates(self):
        backend = self.imports.allow("flashinfer.fp", "FlashInferAttBackend")
        self.args.llm_prefill_att_backend = ["auto"]
        self.validate.side_effect = lambda name: name == "flashinfer"

        self.assertIs(self.attention.get_prefill_att_backend_class(), backend)
        self.assertEqual([call.args[0] for call in self.validate.call_args_list], ["fa3", "flashinfer"])
        self.assertEqual(self.imports.imported, [f"{ATTENTION}.flashinfer.fp"])

    def test_mtp_decode_preserves_fa3_priority(self):
        backend = self.imports.allow("fa3.fp", "Fa3AttBackend")
        self.args.llm_decode_att_backend = ["auto"]
        self.args.mtp_step = 2

        self.assertIs(self.attention.get_decode_att_backend_class(), backend)
        self.validate.assert_called_once_with("fa3")

    def test_expert_parallelism_still_excludes_flashinfer(self):
        backend = self.imports.allow("triton.fp", "TritonAttBackend")
        self.args.llm_decode_att_backend = ["auto"]
        self.args.enable_ep_moe = True
        self.validate.side_effect = lambda name: name == "triton"

        self.assertIs(self.attention.get_decode_att_backend_class(), backend)
        self.assertEqual([call.args[0] for call in self.validate.call_args_list], ["fa3", "triton"])

    def test_auto_selection_falls_back_to_triton(self):
        backend = self.imports.allow("triton.mla", "MlaTritonAttBackend")
        self.args.llm_prefill_att_backend = ["auto"]
        self.validate.return_value = False

        self.assertIs(self.attention.get_mla_prefill_att_backend_class(), backend)
        self.assertEqual(self.imports.imported, [f"{ATTENTION}.triton.mla"])

    def test_nsa_selection_does_not_import_other_kv_types(self):
        backend = self.imports.allow("nsa.fp8_flashmla_sparse", "NsaFlashMlaFp8SparseAttBackend")
        self.args.llm_kv_type = "fp8kv_dsa"
        self.args.llm_decode_att_backend = ["flashmla_sparse"]

        self.assertIs(self.attention.get_nsa_decode_att_backend_class(), backend)
        self.assertEqual(self.imports.imported, [f"{ATTENTION}.nsa.fp8_flashmla_sparse"])

    def test_custom_backend_map_uses_import_paths(self):
        backend = self.imports.allow("nsa.custom", "CustomBackend")
        backend_map = {"None": {"custom": "nsa.custom:CustomBackend"}}
        self.args.llm_prefill_att_backend = ["custom"]
        self.args.llm_decode_att_backend = ["auto"]

        self.assertIs(self.attention.get_nsa_prefill_att_backend_class(backend_map=backend_map), backend)
        self.assertIs(
            self.attention.get_nsa_decode_att_backend_class(priority_list=["custom"], backend_map=backend_map),
            backend,
        )
        self.assertEqual(self.imports.imported, [f"{ATTENTION}.nsa.custom"])

    def test_neo_triton_selection_does_not_probe_fa3(self):
        backend = self.imports.allow("triton.neo", "NeoTritonAttBackend")

        self.assertIs(self.attention.get_neo_prefill_att_backend_class(), backend)
        self.args.llm_prefill_att_backend = ["auto"]
        self.assertIs(self.attention.get_neo_prefill_att_backend_class(priority_list=["triton"]), backend)
        self.assertEqual(self.imports.imported, [f"{ATTENTION}.triton.neo"])

    def test_neo_auto_selection_still_falls_back_without_image_token_interface(self):
        self.imports.allow("fa3.neo", "NeoFa3AttBackend", HAS_FLASH_ATTN_INTERFACE=False)
        backend = self.imports.allow("triton.neo", "NeoTritonAttBackend")
        self.args.llm_prefill_att_backend = ["auto"]

        self.assertIs(self.attention.get_neo_prefill_att_backend_class(), backend)
        self.validate.assert_called_once_with("triton")
        self.args.llm_prefill_att_backend = ["fa3"]
        with self.assertRaises(KeyError):
            self.attention.get_neo_prefill_att_backend_class()

    def test_neo_auto_selection_uses_supported_fa3(self):
        backend = self.imports.allow("fa3.neo", "NeoFa3AttBackend", HAS_FLASH_ATTN_INTERFACE=True)
        self.args.llm_prefill_att_backend = ["auto"]

        self.assertIs(self.attention.get_neo_prefill_att_backend_class(), backend)
        self.assertEqual(self.imports.imported, [f"{ATTENTION}.fa3.neo"])
        self.validate.assert_called_once_with("fa3")

    def test_linear_selection_imports_only_validated_backend(self):
        backend = self.imports.allow("linear.flashqla", "FlashQlaLinearAttBackend")
        selection = importlib.import_module(f"{ATTENTION}.create_linear_utils")

        self.assertIs(selection.get_qwen35_linear_prefill_att_backend_class(), backend)
        self.assertEqual(self.imports.imported, [f"{ATTENTION}.linear.flashqla"])
        self.validate.assert_called_once_with("flashqla")

    def test_linear_selection_falls_back_without_importing_flashqla(self):
        backend = self.imports.allow("linear.triton", "TritonLinearAttBackend")
        selection = importlib.import_module(f"{ATTENTION}.create_linear_utils")
        self.validate.return_value = False

        self.assertIs(selection.get_qwen35_linear_prefill_att_backend_class(), backend)
        self.assertIs(selection.get_qwen35_linear_decode_att_backend_class(), backend)
        self.args.llm_prefill_att_backend.append("triton")
        self.assertIs(selection.get_qwen35_linear_prefill_att_backend_class(), backend)
        self.assertEqual(self.imports.imported, [f"{ATTENTION}.linear.triton"])

    def test_vit_selection_loads_only_requested_implementation(self):
        package = "lightllm.common.basemodel.attention_vit"
        for name in list(sys.modules):
            if name == package or name.startswith(f"{package}."):
                del sys.modules[name]
        base = ModuleType(f"{package}.base_att")
        base.BaseVitAttBackend = type("BaseVitAttBackend", (), {})
        sys.modules[base.__name__] = base
        sys.modules["lightllm.utils.backend_validator"]._validate = self.validate
        imports = BackendImports(package)
        sys.meta_path.insert(0, imports)
        self.addCleanup(sys.meta_path.remove, imports)
        selection = importlib.import_module(f"{package}.create_utils")
        self.assertEqual(imports.imported, [])

        for name, class_name in (
            ("fa3", "Fa3VitAttBackend"),
            ("triton", "TritonVitAttBackend"),
            ("sdpa", "SdpaVitAttBackend"),
            ("xformers", "XformersVitAttBackend"),
        ):
            backend = imports.allow(f"{name}.fp", class_name)
            self.args.vit_att_backend = [name]
            self.assertEqual(selection.init_vit_att_backend(), name)
            self.assertIs(selection.get_vit_att_backend_class(name), backend)
        self.validate.assert_not_called()

        self.args.vit_att_backend = ["auto"]
        self.validate.side_effect = lambda name: name == "sdpa"
        self.assertEqual(selection.init_vit_att_backend(), "sdpa")
        self.assertEqual([call.args[0] for call in self.validate.call_args_list], ["fa3", "xformers", "sdpa"])


if __name__ == "__main__":
    unittest.main()
