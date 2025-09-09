import os
import shutil
import torch
import functools
import time

from typing import List, Tuple
from outlines.models.transformers import TransformerTokenizer
from .impl import ChunkedPrefillBackend
from lightllm.server.core.objs import FinishStatus
from lightllm.server.router.model_infer.infer_batch import g_infer_context, InferReq
from lightllm.server.router.model_infer.mode_backend.generic_pre_process import (
    prepare_prefill_inputs,
    prepare_decode_inputs,
)
from lightllm.server.router.model_infer.mode_backend.generic_post_process import sample
from lightllm.server.tokenizer import get_tokenizer

from lightllm.utils.log_utils import init_logger

from lightllm.server.router.model_infer.mode_backend.chunked_prefill.pre3_core.core import (
    compute_first,
    compute_graph,
    Graph,
    NT,
    T,
)
from lightllm.server.router.model_infer.mode_backend.chunked_prefill.pre3_core.dpda import LRGraph, DPDA

try:
    from lightllm_constraint_decode_kernel import check_dpda, batched_check_dpda
except ImportError:
    assert False, "lightllm_constraint_decode_kernel is not installed. Please install it to use this pre3 mode."

logger = init_logger(__name__)


class DPDAStructure:
    def __init__(self):
        # Preprocessed Vocabulary List
        self.input_sequences = None
        self.sequence_len = None
        self.check_str = None  # Now keep the original string for check, can be removed later

        # Python DPDA structure
        self.lr1_graph = None
        self.dpda = None
        self.graph = None

        # Torch DPDA structure
        self.shift_table = None
        self.edge_num_table = None
        self.push_table = None
        self.pop_table = None
        self.dest_table = None
        self.symbol_to_id = None
        return


class Pre3JsonModeBackend(ChunkedPrefillBackend):
    def __init__(self) -> None:
        super().__init__()
        self.json_ebnf_path = (
            "lightllm/server/router/model_infer/mode_backend/chunked_prefill/pre3_core/json_grammar.ebnf"
        )
        self.format = "ebnf"  # "ebnf" or "python"
        self.batched_mask = True

    def init_custom(self):
        # Using ebnf format grammar
        if self.format == "ebnf":
            logger.warning("Parsing EBNF grammar is an experimental feature, and may not work as expected.")
            with open(self.json_ebnf_path, "r") as f:
                from lightllm.server.router.model_infer.mode_backend.chunked_prefill.pre3_core.grammar_parser import (
                    fix_grammar,
                    parse_ebnf,
                )

                input_text = f.read()
                parsed_grammar = parse_ebnf(input_text)
                grammar = parsed_grammar.get_grammar()
                json_grammar = fix_grammar(grammar)
                start_symbol = "root"
        elif self.format == "python":
            from lightllm.server.router.model_infer.mode_backend.chunked_prefill.pre3_core.example_grammar import (
                json_grammar,
            )

            start_symbol = "JSON"

        self.tokenizer = TransformerTokenizer(
            get_tokenizer(self.args.model_dir, self.args.tokenizer_mode, trust_remote_code=self.args.trust_remote_code)
        )
        self.eos_token_ids = []
        self.eos_token_ids.append(self.tokenizer.eos_token_id)
        self.eos_token_ids.extend(self.args.eos_id)
        self.tokenizer.eos_token_ids = self.eos_token_ids

        dpda_struct = DPDAStructure()
        start_time = time.time()
        dpda_struct.graph = compute_graph(json_grammar, start_symbol=start_symbol)
        # print(dpda_struct.graph)
        # graph.check_lr1()
        dpda_struct.lr1_graph = LRGraph(dpda_struct.graph)
        dpda_struct.dpda = DPDA(lr_graph=dpda_struct.lr1_graph)
        dpda_struct.dpda.remove_no_input_node_to_edges()
        (
            dpda_struct.shift_table,
            dpda_struct.edge_num_table,
            dpda_struct.push_table,
            dpda_struct.pop_table,
            dpda_struct.dest_table,
            dpda_struct.symbol_to_id,
        ) = dpda_struct.dpda.dump_to_tensor()
        logger.info(f"preprocess JSON dpda cost: {time.time() - start_time}")

        start_time = time.time()
        vocab = self.tokenizer.tokenizer.get_vocab()
        sorted_vocab = dict(sorted(vocab.items(), key=lambda item: item[1]))
        dpda_struct.check_str = list(sorted_vocab.keys())
        other_token_id = len(dpda_struct.symbol_to_id)
        # print(dpda_struct.symbol_to_id)
        _input_sequences = []
        _sequence_len = []
        for s in dpda_struct.check_str:
            _input_sequences.append(
                [dpda_struct.symbol_to_id[c] if c in dpda_struct.symbol_to_id else other_token_id for c in s]
            )
            _sequence_len.append(len(s))
        dpda_struct.sequence_len = torch.tensor(_sequence_len, dtype=torch.int32, device="cuda")
        dpda_struct.input_sequences = torch.empty(
            (len(_input_sequences), torch.max(dpda_struct.sequence_len)), dtype=torch.int32, device="cuda"
        )
        for i, s in enumerate(_input_sequences):
            dpda_struct.input_sequences[i, : len(s)] = torch.tensor(s, dtype=torch.int32, device="cuda")
        logger.info(f"preprocess LLM vocabulary cost: {time.time() - start_time}")

        self.dpda_struct = dpda_struct

    def decode(self):
        uninit_reqs, aborted_reqs, ok_finished_reqs, prefill_reqs, decode_reqs = self._get_classed_reqs(
            g_infer_context.infer_req_ids
        )

        if aborted_reqs:
            g_infer_context.filter_reqs(aborted_reqs)

        # 先 decode
        if decode_reqs:
            kwargs, run_reqs = prepare_decode_inputs(decode_reqs)
            logits = self.model.forward(**kwargs)
            self._overlap_req_init_and_filter(
                uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs, clear_list=True
            )

            mask = torch.ones_like(logits, dtype=torch.bool)
            if not self.batched_mask:
                for i, run_obj in enumerate(run_reqs):
                    self._batched_mask_req_out_token([i], [run_obj], mask)
            else:
                self._batched_mask_req_out_token(
                    [i for i in range(len(run_reqs))],
                    run_reqs,
                    mask,
                )
            logits[mask] = -1000000.0

            next_token_ids, next_token_probs = sample(logits, run_reqs, self.eos_id)
            next_token_ids = next_token_ids.detach().cpu().numpy()
            next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()
            self._post_handle(
                run_reqs,
                next_token_ids,
                next_token_logprobs,
                is_chuncked_mode=False,
                do_filter_finished_reqs=False,
                extra_post_req_handle_func=self._update_state_fsm,
            )
            logits = None

        # 再 prefill
        if len(decode_reqs) == 0 or (self.forward_step % self.max_wait_step == 0) or (self.need_prefill_count > 0):
            if prefill_reqs:
                self.need_prefill_count -= 1
                kwargs, run_reqs = prepare_prefill_inputs(
                    prefill_reqs, is_chuncked_mode=True, is_multimodal=self.is_multimodal
                )
                logits = self.model.forward(**kwargs)
                self._overlap_req_init_and_filter(
                    uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs, clear_list=True
                )
                # 对于不能满足前缀匹配的logic位置，将其logics设置为一个较大负值，将其概率掩盖为 0
                mask = torch.ones_like(logits, dtype=torch.bool)
                if not self.batched_mask:
                    for i, run_obj in enumerate(run_reqs):
                        self._batched_mask_req_out_token([i], [run_obj], mask)
                else:
                    self._batched_mask_req_out_token(
                        [i for i in range(len(run_reqs))],
                        run_reqs,
                        mask,
                    )
                logits[mask] = -1000000.0

                next_token_ids, next_token_probs = sample(logits, run_reqs, self.eos_id)
                next_token_ids = next_token_ids.detach().cpu().numpy()
                next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()
                self._post_handle(
                    run_reqs,
                    next_token_ids,
                    next_token_logprobs,
                    is_chuncked_mode=True,
                    do_filter_finished_reqs=False,
                    extra_post_req_handle_func=self._update_state_fsm,
                )
                logits = None

        self._overlap_req_init_and_filter(uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs, clear_list=True)
        self.forward_step += 1
        return

    def _update_state_fsm(self, req_obj: InferReq, next_token_id, next_token_logprob):
        next_token_id = int(next_token_id)
        next_token = self.tokenizer.tokenizer.convert_ids_to_tokens([next_token_id])[0]
        if next_token_id not in self.eos_token_ids:
            (
                ok,
                req_obj.sampling_param.lr1_stack,
                req_obj.sampling_param.lr1_current_node_id,
            ) = self.dpda_struct.dpda.try_shift(
                input_str=next_token,
                current_stack=req_obj.sampling_param.lr1_stack,
                current_node_id=req_obj.sampling_param.lr1_current_node_id,
            )
        if not ok:
            req_obj.finish_status.set_status(FinishStatus.FINISHED_STOP)
        return

    def _batched_mask_req_out_token(
        self, i_list, run_obj_list, mask, lr1_stack=None, lr1_stack_size=None, prefill=False
    ):
        batch_size = len(i_list)
        sample_params_list = [e.sampling_param for e in run_obj_list]
        vocab_size = self.dpda_struct.input_sequences.shape[0]

        current_state_list = []
        for sample_params in sample_params_list:
            if sample_params.lr1_current_node_id is not None:
                current_state_list.append(sample_params.lr1_current_node_id)
            else:
                current_state_list.append(0)
        output = torch.empty((batch_size * vocab_size), dtype=torch.int32, device="cuda")
        current_state = torch.tensor(current_state_list, dtype=torch.int32, device=output.device)

        if lr1_stack_size is not None:
            current_stack_top = lr1_stack_size.to(output.device)
        else:
            current_stack_top = torch.tensor(
                [len(sample_params.lr1_stack) for sample_params in sample_params_list],
                dtype=torch.int32,
                device=output.device,
            )
            max_stack_depth = torch.max(current_stack_top).item()

        if lr1_stack is not None:
            current_stack = lr1_stack.to(output.device)
            max_stack_depth = current_stack.shape[1]
        else:
            current_stack = torch.zeros((batch_size, max_stack_depth), dtype=torch.int32, device=output.device)
            for i, sample_params in enumerate(sample_params_list):
                current_stack[i, : len(sample_params.lr1_stack)] = torch.tensor(
                    sample_params.lr1_stack, dtype=torch.int32, device=output.device
                )

        batched_check_dpda(
            self.dpda_struct.input_sequences.to(output.device),
            self.dpda_struct.sequence_len.to(output.device),
            self.dpda_struct.shift_table.to(output.device),
            self.dpda_struct.edge_num_table.to(output.device),
            self.dpda_struct.push_table.to(output.device),
            self.dpda_struct.pop_table.to(output.device),
            self.dpda_struct.dest_table.to(output.device),
            current_stack,
            current_stack_top,
            current_state,
            output,
            max_stack_depth,
        )

        # print(current_state, current_state.shape)
        # current_state = current_state.reshape(batch_size, -1)
        # for j in range(len(current_state[0])):
        #     if current_state[0][j] != -1:
        #         print(f"accepted: {j} : {sample_params.dpda.check_str[j]}")
        for idx, i in enumerate(i_list):
            mask[i][:vocab_size] = output[idx * vocab_size : (idx + 1) * vocab_size] == -1
            # if self.eos_token_ids is not None:
            #     mask[i][self.eos_token_ids] = False

    def _mask_req_out_token(self, i, run_obj: InferReq, mask, prefill=False):
        sample_params = run_obj.sampling_param
        vocab_size = self.dpda_struct.input_sequences.shape[0]
        current_state = torch.tensor([sample_params.lr1_current_node_id] * vocab_size, dtype=torch.int32, device="cuda")
        current_stack = torch.tensor(sample_params.lr1_stack, dtype=torch.int32, device="cuda")
        check_dpda(
            self.dpda_struct.input_sequences,
            self.dpda_struct.sequence_len,
            self.dpda_struct.shift_table,
            self.dpda_struct.edge_num_table,
            self.dpda_struct.push_table,
            self.dpda_struct.pop_table,
            self.dpda_struct.dest_table,
            current_stack,
            current_state,
            64,
        )
        # if torch.sum(current_state != -1) <= 500:
        #     for j in range(len(current_state)):
        #         if current_state[j] != -1:
        #             print(f"accepted: {j} : {sample_params.dpda.check_str[j]}")
        mask[i][:vocab_size] = current_state == -1
        # mask[i][self.eos_token_ids] = False
