"""Neo-specific image generation fed by LightLLM Prefix KV."""

from __future__ import annotations

import sys

from lightllm.models.registry import GenerationModelRegistry, llm_model_type_is
from lightllm.server.generationserver.prefix_attention import (
    DensePrefixAttentionProvider,
)


@GenerationModelRegistry(
    "neo_chat",
    output_modality="image",
    condition=llm_model_type_is("qwen3"),
)
class NeoImageGenerationModel:
    """Model-owned LightX2V adapter used by the generic image worker."""

    def __init__(
        self,
        args,
        *,
        prefix_layout: tuple[int, int, int],
        device_id: int,
    ) -> None:
        self.args = args
        self.prefix_layout = tuple(int(value) for value in prefix_layout)
        self.device_id = int(device_id)

        backend_path = args.generation_backend_path
        if backend_path is not None and backend_path not in sys.path:
            sys.path.insert(0, backend_path)

        self.pipe = self._load_pipeline()
        self._validate_layout()

    def _load_pipeline(self):
        try:
            from lightx2v import LightX2VPipeline
        except ImportError as exc:
            raise ImportError(
                "X2V generation backend is unavailable; install LightX2V or set "
                "--generation_backend_path"
            ) from exc

        pipe = LightX2VPipeline(
            model_path=self.args.generation_model_dir,
            model_cls="neopp",
            support_tasks=["t2i"],
        )
        pipe.create_generator(config_json=self.args.generation_config)
        pipe.modify_config(
            {
                "load_kv_cache_in_pipeline_for_debug": False,
                "save_result_for_debug": False,
            }
        )
        if not callable(getattr(pipe.runner, "set_prefix_provider", None)):
            raise RuntimeError(
                "the installed LightX2V does not expose NeoPP prefix providers; "
                "use the matching LightX2V checkout"
            )
        return pipe

    def _validate_layout(self) -> None:
        transformer = self.pipe.runner.model.transformer_infer
        expected = (
            int(transformer.num_layers),
            int(transformer.num_kv_heads),
            int(transformer.head_dim),
        )
        actual = self.prefix_layout
        if actual != expected:
            raise ValueError(
                "LightLLM/X2V prefix layout mismatch: "
                f"target={actual}, generation={expected}"
            )

    def generate(self, *, activations: dict, generation) -> list:
        prefixes = activations
        providers = {
            role: DensePrefixAttentionProvider(prefix)
            for role, prefix in prefixes.items()
        }
        runner = self.pipe.runner
        runner.scheduler.infer_steps = generation.steps
        runner.set_inference_params(
            index_offset_cond=prefixes["cond"].position_offset,
            index_offset_uncond=prefixes["uncond"].position_offset,
            cfg_interval=generation.cfg_interval,
            cfg_scale=generation.guidance_scale,
            cfg_norm=generation.cfg_norm,
            timestep_shift=generation.timestep_shift,
        )

        outputs = []
        for output_index in range(generation.num_outputs):
            # NeoPP clears non-owning inputs after every run, so rebind the
            # immutable Prefix KV providers for each requested sample.
            runner.set_prefix_provider(providers["cond"], providers["uncond"])
            outputs.append(
                self.pipe.generate(
                    seed=generation.seed + output_index,
                    save_result_path="",
                    target_shape=[generation.height, generation.width],
                )
            )
        return outputs

    def finish_task(self) -> None:
        self.pipe.runner.clear_kvcache()
