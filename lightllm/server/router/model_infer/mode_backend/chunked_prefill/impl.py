import torch
import os
from typing import List
from transformers.configuration_utils import PretrainedConfig
from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend
from lightllm.server.router.model_infer.mode_backend.overlap_events import OverlapEventPack
from lightllm.server.router.model_infer.infer_batch import InferReq
from lightllm.models.deepseek_mtp.model import Deepseek3MTPModel
from lightllm.server.router.model_infer.mode_backend.pre import (
    prepare_prefill_inputs,
    prepare_decode_inputs,
)
from lightllm.server.router.model_infer.mode_backend.mtp_pre_process import (
    prepare_mtp_prefill_inputs,
)
from lightllm.server.router.model_infer.mode_backend.generic_post_process import sample
from lightllm.server.router.model_infer.infer_batch import g_infer_context
from lightllm.common.basemodel.infer_lock import g_infer_state_lock
from lightllm.common.basemodel.batch_objs import ModelOutput
from lightllm.utils.log_utils import init_logger
from lightllm.utils.dist_utils import get_current_device_id
from lightllm.utils.envs_utils import get_env_start_args

logger = init_logger(__name__)


class ChunkedPrefillBackend(ModeBackend):
    def __init__(self) -> None:
        super().__init__()

        # 在 mtp 模式下切换绑定的prefill 和 decode 函数
        if get_env_start_args().mtp_mode:
            self.prefill = self.prefill_mtp
            self.decode = self.decode_mtp
        else:
            self.prefill = self.prefill_normal
            self.decode = self.decode_normal
        return

    def init_mtp_draft_model(self, main_kvargs: dict):
        self.mtp_step = self.args.mtp_step
        self.draft_models: List[Deepseek3MTPModel] = []

        os.environ["DISABLE_CHECK_MAX_LEN_INFER"] = "1"
        for i in range(self.mtp_step):
            mtp_model_cfg, _ = PretrainedConfig.get_config_dict(self.args.mtp_draft_model_dir)
            mtp_model_kvargs = {
                "weight_dir": self.args.mtp_draft_model_dir,
                "max_total_token_num": self.model.mem_manager.size,
                "load_way": main_kvargs["load_way"],
                "mode": main_kvargs["mode"],
                "max_req_num": main_kvargs.get("max_req_num", 1000),
                "max_seq_length": main_kvargs.get("max_seq_length", 1024 * 5),
                "is_token_healing": False,
                "return_all_prompt_logics": False,
                "use_dynamic_prompt_cache": self.use_dynamic_prompt_cache,
                "disable_chunked_prefill": self.disable_chunked_prefill,
                "data_type": main_kvargs.get("data_type", "float16"),
                "graph_max_batch_size": main_kvargs.get("graph_max_batch_size", 16),
                "graph_max_len_in_batch": main_kvargs.get("graph_max_len_in_batch", 8196),
                "disable_cudagraph": main_kvargs.get("disable_cudagraph", False),
                "mem_fraction": main_kvargs["mem_fraction"],
                "batch_max_tokens": main_kvargs.get("batch_max_tokens", None),
                "quant_type": main_kvargs.get("quant_type", None),
                "quant_cfg": main_kvargs.get("quant_cfg", None),
                "run_mode": "normal",
                "main_model": self.model,
                "mem_layer_start": self.model.config["num_hidden_layers"] + i * mtp_model_cfg["num_hidden_layers"],
            }

            mtp_model_cfg, _ = PretrainedConfig.get_config_dict(self.args.mtp_draft_model_dir)
            assert mtp_model_cfg["model_type"] == "deepseek_v3"
            assert mtp_model_cfg["architectures"][0] == "DeepseekV3ForCausalLMNextN"
            self.draft_models.append(Deepseek3MTPModel(mtp_model_kvargs))

            self.logger.info(f"loaded mtp model class {self.draft_models[i].__class__}")
        return

    def infer_loop(self):
        torch.cuda.set_device(get_current_device_id())
        try:
            while True:
                event_pack = self.overlap_event_manager.get_overlap_event_pack()
                event_pack.wait_to_forward()

                self._try_read_new_reqs()

                prefill_reqs, decode_reqs = self._get_classed_reqs()
                if prefill_reqs:
                    self.prefill(
                        event_pack=event_pack,
                        prefill_reqs=prefill_reqs,
                    )
                    continue

                if decode_reqs:
                    self.decode(
                        event_pack=event_pack,
                        decode_reqs=decode_reqs,
                    )
                    continue

                event_pack.notify_post_handle_and_wait_pre_post_handle()
                event_pack.notify_forward_and_wait_post_handle()
                event_pack.notify_pre_post_handle()
                continue
        except BaseException as e:
            self.logger.exception(str(e))
            raise e

    def prefill_normal(
        self,
        event_pack: OverlapEventPack,
        prefill_reqs: List[InferReq],
    ):
        # 第一阶段
        model_input, run_reqs = prepare_prefill_inputs(
            prefill_reqs, is_chuncked_mode=not self.disable_chunked_prefill, is_multimodal=self.is_multimodal
        )
        model_output = self.model.forward(model_input)
        logits = model_output.logits

        if self.prefill_mask_func is not None:
            self.prefill_mask_func(run_reqs, logits)

        next_token_ids, next_token_probs = sample(logits, run_reqs, self.eos_id)
        next_token_logprobs = torch.log(next_token_probs)
        sync_event = torch.cuda.Event()
        sync_event.record()

        # 第二阶段
        event_pack.notify_post_handle_and_wait_pre_post_handle()
        update_packs = self._pre_post_handle(run_reqs, is_chuncked_mode=not self.disable_chunked_prefill)

        # 第三阶段
        event_pack.notify_forward_and_wait_post_handle()
        sync_event.synchronize()
        next_token_ids = next_token_ids.detach().cpu().numpy()
        next_token_logprobs = next_token_logprobs.detach().cpu().numpy()
        self._post_handle(
            run_reqs=run_reqs,
            next_token_ids=next_token_ids,
            next_token_logprobs=next_token_logprobs,
            run_reqs_update_packs=update_packs,
            extra_post_req_handle_func=self.extra_post_req_handle_func,
        )
        # 第四阶段
        event_pack.notify_pre_post_handle()
        return

    def decode_normal(
        self,
        event_pack: OverlapEventPack,
        decode_reqs: List[InferReq],
    ):
        model_input, run_reqs = prepare_decode_inputs(decode_reqs)
        model_output = self.model.forward(model_input)
        logits = model_output.logits

        if self.decode_mask_func is not None:
            self.decode_mask_func(run_reqs, logits)

        next_token_ids, next_token_probs = sample(logits, run_reqs, self.eos_id)
        next_token_logprobs = torch.log(next_token_probs)
        sync_event = torch.cuda.Event()
        sync_event.record()

        # 第二阶段
        event_pack.notify_post_handle_and_wait_pre_post_handle()
        update_packs = self._pre_post_handle(run_reqs, is_chuncked_mode=False)

        # 第三阶段
        event_pack.notify_forward_and_wait_post_handle()
        sync_event.synchronize()
        next_token_ids = next_token_ids.detach().cpu().numpy()
        next_token_logprobs = next_token_logprobs.detach().cpu().numpy()
        self._post_handle(
            run_reqs=run_reqs,
            next_token_ids=next_token_ids,
            next_token_logprobs=next_token_logprobs,
            run_reqs_update_packs=update_packs,
            extra_post_req_handle_func=self.extra_post_req_handle_func,
        )

        # 第四阶段
        event_pack.notify_pre_post_handle()
        return

    def prefill_mtp(
        self,
        event_pack: OverlapEventPack,
        prefill_reqs: List[InferReq],
    ):
        model_input, run_reqs = prepare_prefill_inputs(
            prefill_reqs, is_chuncked_mode=not self.disable_chunked_prefill, is_multimodal=self.is_multimodal
        )
        model_output = self.model.forward(model_input)

        next_token_ids_gpu, next_token_probs = sample(model_output.logits, run_reqs, self.eos_id)
        next_token_ids_cpu = next_token_ids_gpu.detach().cpu().numpy()
        next_token_logprobs_cpu = torch.log(next_token_probs).detach().cpu().numpy()

        update_packs = self._pre_post_handle(run_reqs, is_chuncked_mode=not self.disable_chunked_prefill)
        self._post_handle(
            run_reqs=run_reqs,
            next_token_ids=next_token_ids_cpu,
            next_token_logprobs=next_token_logprobs_cpu,
            run_reqs_update_packs=update_packs,
            extra_post_req_handle_func=self.extra_post_req_handle_func,
        )

        # mtp kv fill
        draft_next_token_ids_gpu = next_token_ids_gpu
        draft_model_output = model_output
        draft_model_input = model_input
        # spec prefill: MTP, 这个地方只是为了填充draft model的 kv， 并不会使用生成的token_id。
        for draft_model_idx in range(self.mtp_step):
            draft_model_input = prepare_mtp_prefill_inputs(
                model_input=draft_model_input,
                b_next_token_ids=draft_next_token_ids_gpu,
                deepseekv3_mtp_draft_input_hiddens=draft_model_output.deepseekv3_mtp_main_output_hiddens,
            )

            draft_model_output = self.draft_models[draft_model_idx].forward(draft_model_input)
            draft_next_token_ids_gpu, draft_next_token_ids_cpu = self._gen_argmax_token_ids(draft_model_output)
        return

    def decode_mtp(
        self,
        event_pack: OverlapEventPack,
        decode_reqs: List[InferReq],
    ):
        model_input, run_reqs = prepare_decode_inputs(decode_reqs)
        model_output = self.model.forward(model_input)

        next_token_ids_gpu, next_token_probs = sample(model_output.logits, run_reqs, self.eos_id)
        next_token_ids_cpu = next_token_ids_gpu.detach().cpu().numpy()
        next_token_logprobs_cpu = torch.log(next_token_probs).detach().cpu().numpy()

        # verify
        mem_indexes_cpu = model_input.mem_indexes.detach().cpu().numpy()
        verify_ok_reqs, verify_ok_req_indexes, verify_ok_req_last_indexes, need_free_mem_indexes = self._verify_mtp(
            run_reqs, next_token_ids_cpu, mem_indexes_cpu
        )

        update_packs = self._pre_post_handle(verify_ok_reqs, is_chuncked_mode=False)
        self._post_handle(
            run_reqs=verify_ok_reqs,
            next_token_ids=next_token_ids_cpu[verify_ok_req_indexes],
            next_token_logprobs=next_token_logprobs_cpu[verify_ok_req_indexes],
            run_reqs_update_packs=update_packs,
            extra_post_req_handle_func=self.extra_post_req_handle_func,
        )

        # share some inference info with the main model
        draft_model_input = model_input
        draft_model_output = model_output
        draft_next_token_ids = next_token_ids_gpu
        # process the draft model output
        for draft_model_idx in range(self.mtp_step):

            draft_model_input.input_ids = draft_next_token_ids
            draft_model_input.deepseekv3_mtp_draft_input_hiddens = draft_model_output.deepseekv3_mtp_main_output_hiddens
            # spec decode: MTP
            draft_model_output: ModelOutput = self.draft_models[draft_model_idx].forward(draft_model_input)
            draft_next_token_ids, draft_next_token_ids_cpu = self._gen_argmax_token_ids(draft_model_output)

            unique_reqs = [run_reqs[index] for index in verify_ok_req_last_indexes]
            self._update_reqs_mtp_gen_token_ids(
                reqs=unique_reqs, mtp_draft_next_token_ids=draft_next_token_ids_cpu[verify_ok_req_last_indexes]
            )

        if need_free_mem_indexes:
            g_infer_state_lock.acquire()
            g_infer_context.req_manager.mem_manager.free(need_free_mem_indexes)
            g_infer_state_lock.release()
        return
