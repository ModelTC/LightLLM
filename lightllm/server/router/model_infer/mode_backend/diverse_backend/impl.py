import torch
from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend
from lightllm.server.router.model_infer.infer_batch import (
    g_infer_context,
    InferReq,
    InferReqUpdatePack,
)
from typing import List, Tuple
from lightllm.server.router.model_infer.mode_backend.pre import (
    prepare_prefill_inputs,
)
from lightllm.server.router.model_infer.mode_backend.generic_post_process import sample
from lightllm.server.router.model_infer.mode_backend.overlap_events import OverlapEventPack
from lightllm.common.basemodel.triton_kernel.gather_token_id import scatter_token
from lightllm.server.router.model_infer.pin_mem_manager import g_pin_mem_manager
from ..chunked_prefill.impl import ChunkedPrefillBackend
from lightllm.common.basemodel.infer_lock import g_infer_state_lock
from lightllm.utils.envs_utils import get_env_start_args


class DiversehBackend(ChunkedPrefillBackend):
    def __init__(self) -> None:
        super().__init__()

        if get_env_start_args().mtp_mode:
            # 当前只有 mistral mtp 可以使用 diverse mode 的 mtp 功能。
            self.prefill = self.beam_prefill
            assert get_env_start_args().mtp_mode in ["vanilla_no_att", "eagle_no_att"]
        else:
            self.prefill = self.beam_prefill

        self.classed_req_strict_prefill = True

    def beam_prefill(self, event_pack: OverlapEventPack, prefill_reqs: List[InferReq]):
        # 第一阶段
        group_reqs = [g_infer_context.requests_mapping[req.req_id] for req in prefill_reqs if req.is_master_req()]

        model_input, group_run_reqs = prepare_prefill_inputs(
            group_reqs, is_chuncked_mode=not self.disable_chunked_prefill
        )

        with torch.cuda.stream(g_infer_context.get_overlap_stream()):

            model_output = self.model.forward(model_input)
            logits = model_output.logits

            batch_idx, run_reqs = self._diverse_copy(
                master_reqs=group_reqs, b_prefill_has_out=model_input.b_prefill_has_output_cpu
            )
            b_req_idx = [req.req_idx for req in run_reqs]
            b_has_out = [model_input.b_prefill_has_output_cpu[i] for i in batch_idx]

            batch_idx = g_pin_mem_manager.gen_from_list(key="batch_idx_", data=batch_idx, dtype=torch.int64).cuda(
                non_blocking=True
            )
            b_req_idx = g_pin_mem_manager.gen_from_list(key="b_req_idx_", data=b_req_idx, dtype=torch.int32).cuda(
                non_blocking=True
            )
            b_has_out = g_pin_mem_manager.gen_from_list(key="b_has_out_", data=b_has_out, dtype=torch.bool).cuda(
                non_blocking=True
            )

            logits = logits[batch_idx]
            b_mtp_index = model_input.b_mtp_index[batch_idx]

            next_token_ids, next_token_logprobs = sample(logits, run_reqs, self.eos_id)

            scatter_token(
                next_token_ids=next_token_ids,
                req_to_next_token_ids=self.model.req_manager.req_sampling_params_manager.req_to_next_token_ids,
                b_req_idx=b_req_idx,
                b_mtp_index=b_mtp_index,
                b_has_out=b_has_out,
            )

            next_token_ids_cpu, next_token_logprobs_cpu = self._async_copy_next_token_infos_to_pin_mem(
                next_token_ids=next_token_ids, next_token_logprobs=next_token_logprobs
            )

            sync_event = torch.cuda.Event()
            sync_event.record()

        # 第二阶段
        event_pack.notify_post_handle_and_wait_pre_post_handle()
        model_output.prefill_mem_indexes_ready_event.synchronize()
        update_packs = self._diverse_pre_post_handle(run_reqs, is_chuncked_mode=not self.disable_chunked_prefill)
        # 第三阶段
        event_pack.notify_forward_and_wait_post_handle()
        sync_event.synchronize()
        self._post_handle(
            run_reqs=run_reqs,
            next_token_ids=next_token_ids_cpu,
            next_token_logprobs=next_token_logprobs_cpu,
            run_reqs_update_packs=update_packs,
            extra_post_req_handle_func=self.extra_post_req_handle_func,
        )
        # 第四阶段
        event_pack.notify_pre_post_handle()
        return

    def _diverse_copy(
        self, master_reqs: List[InferReq], b_prefill_has_out: List[bool]
    ) -> Tuple[List[int], List[InferReq]]:
        batch_idx = []
        run_reqs = []
        for i in range(len(master_reqs)):
            master_req = master_reqs[i]
            slave_reqs = master_req.slave_reqs
            slave_num = len(slave_reqs)
            batch_idx.append(i)
            run_reqs.append(master_req)

            if slave_num > 0 and b_prefill_has_out[i]:
                batch_idx.extend([i for _ in range(slave_num)])
                run_reqs.extend(slave_reqs)

        return batch_idx, run_reqs

        # 一些可以复用的通用功能函数

    def _diverse_pre_post_handle(self, run_reqs: List[InferReq], is_chuncked_mode: bool) -> List[InferReqUpdatePack]:
        update_func_objs: List[InferReqUpdatePack] = []
        # 通用状态预先填充
        is_master_in_dp = self.is_master_in_dp
        pre_master_req_pack = None
        for req_obj in run_reqs:
            req_obj: InferReq = req_obj
            if req_obj.is_master_req():
                if is_chuncked_mode:
                    new_kv_len = req_obj.get_chuncked_input_token_len()
                else:
                    new_kv_len = req_obj.get_cur_total_len()
                req_obj.cur_kv_len = new_kv_len
                if is_master_in_dp:
                    req_obj.shm_req.shm_cur_kv_len = req_obj.cur_kv_len

                # 对于没有到达需要输出 token 阶段的请求，直接略过, 说明还
                # 处于chuncked prefill kv 填充的阶段。
                if req_obj.cur_kv_len < req_obj.get_cur_total_len():
                    pack = InferReqUpdatePack(req_obj=req_obj, output_len=0)
                    update_func_objs.append(pack)
                    pre_master_req_pack = pack
                    # TODO 如果 diverse mode 需要支持 nixl pd 分离，则应该每个分块prefill后都进行相关的复制，
                    # 暂时不支持 diverse mode 和 pd 模式的混合
                    continue

                # 将生成的下一个token的信息写入到管理对象中。
                req_obj.cur_output_len += 1
                pack = InferReqUpdatePack(req_obj=req_obj, output_len=req_obj.cur_output_len)
                update_func_objs.append(pack)
                pre_master_req_pack = pack
                if req_obj.slave_reqs:
                    # 存在 slave reqs 的 master req 需要将自己的 kv 信息写入到 radix cache 中
                    # 方便 slave req 进行 kv 的复用
                    self._master_req_to_radix_cache(master_req=req_obj)
            else:
                # slave req 直接复用 master req 的更新包。
                assert pre_master_req_pack is not None
                assert pre_master_req_pack.req_obj.shm_req.group_req_id == req_obj.shm_req.group_req_id
                self._copy_master_req_to_slave_req(slave_req=req_obj)
                # 在拷贝后，请求独立了，与 master_req 的关系解除
                req_obj.remove_master_req()
                pack = InferReqUpdatePack(req_obj=req_obj, output_len=pre_master_req_pack.output_len)
                update_func_objs.append(pack)

        torch.cuda.current_stream().synchronize()
        return update_func_objs

    def _master_req_to_radix_cache(self, master_req: InferReq):
        g_infer_state_lock.acquire()
        key = master_req.get_input_token_ids()[0 : master_req.cur_kv_len]
        key = torch.tensor(key, dtype=torch.int64, device="cpu")
        value = self.model.req_manager.req_to_token_indexs[master_req.req_idx][: master_req.cur_kv_len].detach().cpu()
        prefix_len, new_shared_kv_node = self.radix_cache.insert(key, value)
        old_prefix_len = 0 if master_req.shared_kv_node is None else master_req.shared_kv_node.node_prefix_total_len
        assert old_prefix_len <= master_req.cur_kv_len
        self.model.mem_manager.free(
            self.model.req_manager.req_to_token_indexs[master_req.req_idx][old_prefix_len:prefix_len]
        )

        # 将原有共享节点替换为新共享节点，新共享节点对应的长度为当前的cur_kv_len
        self.radix_cache.dec_node_ref_counter(master_req.shared_kv_node)
        self.radix_cache.add_node_ref_counter(new_shared_kv_node)
        master_req.shared_kv_node = new_shared_kv_node
        assert (
            new_shared_kv_node.node_prefix_total_len == master_req.cur_kv_len
        ), f"shared len: {new_shared_kv_node.node_prefix_total_len} cur_kv_len {master_req.cur_kv_len}"

        share_node, kv_len, value = self.radix_cache.match_prefix(key, update_refs=False)
        assert share_node == new_shared_kv_node and kv_len == master_req.cur_kv_len
        self.model.req_manager.req_to_token_indexs[master_req.req_idx][0 : master_req.cur_kv_len] = value
        g_infer_state_lock.release()
        return

    def _copy_master_req_to_slave_req(self, slave_req: InferReq):
        g_infer_state_lock.acquire()
        master_req = slave_req.related_master_req
        assert master_req is not None

        self.radix_cache.dec_node_ref_counter(slave_req.shared_kv_node)
        self.radix_cache.add_node_ref_counter(master_req.shared_kv_node)

        kv_len = master_req.cur_kv_len

        self.model.req_manager.req_to_token_indexs[slave_req.req_idx][
            0:kv_len
        ] = self.model.req_manager.req_to_token_indexs[master_req.req_idx][0:kv_len]
        # torch.cuda.current_stream().synchronize()
        slave_req.shared_kv_node = master_req.shared_kv_node
        slave_req.cur_kv_len = kv_len
        slave_req.cur_output_len = master_req.cur_output_len
        if self.is_master_in_dp:
            slave_req.shm_req.shm_cur_kv_len = slave_req.cur_kv_len

        assert kv_len <= slave_req.shm_req.input_len

        g_infer_state_lock.release()
        return
