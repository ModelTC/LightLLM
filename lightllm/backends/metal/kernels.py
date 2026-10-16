"""PyTorch MPS operators consumed by the regular LightLLM Qwen3.5 path.

These functions replace device kernels only. Model orchestration, weight
loading, request indexing, KV allocation and hybrid state stay in their
existing LightLLM classes. CUDA/Triton implementations are imported separately.
"""

from typing import Optional, Tuple

import torch

from lightllm.utils.config_utils import ffn_use_tanh_approximate_gelu


def _expand_mps_prefill_metadata(b_req_idx, b_seq_len, b_ready_cache_len, b_start_loc, token_num):
    lengths = (b_seq_len - b_ready_cache_len).to(torch.long)
    batch_token_starts = torch.cumsum(lengths, dim=0) - lengths
    batch_metadata = torch.stack((b_req_idx, b_ready_cache_len, b_start_loc, batch_token_starts), dim=1).to(torch.long)
    token_metadata = torch.repeat_interleave(
        batch_metadata,
        lengths,
        dim=0,
        output_size=token_num,
    )
    token_offsets = torch.arange(token_num, dtype=torch.long, device=b_req_idx.device)
    local_offsets = token_offsets - token_metadata[:, 3]
    req_rows = token_metadata[:, 0]
    req_positions = token_metadata[:, 1] + local_offsets
    flat_positions = token_metadata[:, 2] + local_offsets
    return req_rows, req_positions, flat_positions


def _causal_conv1d_mps_grouped(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    starts,
    cache_ids,
    has_state,
    conv_states: Optional[torch.Tensor],
    activation: Optional[str],
    pad_slot_id: int,
) -> None:
    channels, kernel_width = weight.shape
    state_width = kernel_width - 1
    groups = {}
    for row, (start, end) in enumerate(zip(starts, starts[1:])):
        if end <= start or cache_ids[row] == pad_slot_id:
            continue
        groups.setdefault(end - start, []).append((row, start, end))

    for rows_info in groups.values():
        rows = torch.tensor([row for row, _, _ in rows_info], dtype=torch.long, device=x.device)
        if x.ndim == 3:
            active_x = x.index_select(0, rows)
        else:
            active_x = torch.stack([x[:, start:end] for _, start, end in rows_info])

        history = torch.zeros((len(rows_info), channels, state_width), dtype=torch.float32, device=x.device)
        if conv_states is not None and state_width:
            history_rows = [
                (group_row, cache_ids[row])
                for group_row, (row, _, _) in enumerate(rows_info)
                if cache_ids[row] != pad_slot_id and has_state[row]
            ]
            if history_rows:
                group_rows = torch.tensor(
                    [group_row for group_row, _ in history_rows], dtype=torch.long, device=x.device
                )
                state_indexes = torch.tensor(
                    [state_index for _, state_index in history_rows],
                    dtype=torch.long,
                    device=conv_states.device,
                )
                history.index_copy_(
                    0,
                    group_rows,
                    conv_states.index_select(0, state_indexes)[:, :, -state_width:].float(),
                )

        raw_x = active_x.float()
        conv_input = torch.cat((history, raw_x), dim=-1)
        output = torch.nn.functional.conv1d(
            conv_input,
            weight.float().unsqueeze(1),
            bias.float() if bias is not None else None,
            groups=channels,
        )
        if activation in ("silu", "swish"):
            output = torch.nn.functional.silu(output)

        if x.ndim == 3:
            x.index_copy_(0, rows, output.to(dtype=x.dtype))
        else:
            for group_row, (_, start, end) in enumerate(rows_info):
                x[:, start:end].copy_(output[group_row].to(dtype=x.dtype))

        if conv_states is not None and state_width:
            state_rows = [
                (group_row, cache_ids[row])
                for group_row, (row, _, _) in enumerate(rows_info)
                if cache_ids[row] != pad_slot_id
            ]
            if state_rows:
                group_rows = torch.tensor([group_row for group_row, _ in state_rows], dtype=torch.long, device=x.device)
                state_indexes = torch.tensor(
                    [state_index for _, state_index in state_rows],
                    dtype=torch.long,
                    device=conv_states.device,
                )
                updated_state = torch.cat(
                    (history.index_select(0, group_rows), raw_x.index_select(0, group_rows)), dim=-1
                )[:, :, -state_width:]
                conv_states.index_copy_(0, state_indexes, updated_state.to(dtype=conv_states.dtype))


def _causal_conv1d_mps(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    query_start_loc: Optional[torch.Tensor],
    cache_indices: Optional[torch.Tensor],
    has_initial_state: Optional[torch.Tensor],
    conv_states: Optional[torch.Tensor],
    activation: Optional[str],
    pad_slot_id: int,
) -> torch.Tensor:
    from .convolution import prefill

    if prefill(
        x, weight, bias, query_start_loc, cache_indices, has_initial_state, conv_states, activation, pad_slot_id
    ):
        return x
    if x.ndim == 2:
        if query_start_loc is None:
            starts = [0, x.shape[-1]]
        else:
            starts = query_start_loc.detach().to(device="cpu", dtype=torch.long).tolist()
    elif x.ndim == 3 and query_start_loc is None:
        starts = list(range(x.shape[0] + 1))
    else:
        raise NotImplementedError("MPS causal convolution expects [channels, tokens] or [batch, channels, tokens].")

    batch_size = len(starts) - 1
    cache_ids = (
        cache_indices.detach().to(device="cpu", dtype=torch.long).tolist()
        if cache_indices is not None
        else [None] * batch_size
    )
    has_state = (
        has_initial_state.detach().to(device="cpu", dtype=torch.bool).tolist()
        if has_initial_state is not None
        else [False] * batch_size
    )
    channels, kernel_width = weight.shape
    state_width = kernel_width - 1
    conv_weight = weight.float().unsqueeze(1)
    conv_bias = bias.float() if bias is not None else None

    active_state_ids = [state_id for state_id in cache_ids if state_id is not None and state_id != pad_slot_id]
    if len(active_state_ids) == len(set(active_state_ids)):
        _causal_conv1d_mps_grouped(
            x,
            weight,
            bias,
            starts,
            cache_ids,
            has_state,
            conv_states,
            activation,
            pad_slot_id,
        )
        return x

    for batch, (start, end) in enumerate(zip(starts, starts[1:])):
        if cache_ids[batch] == pad_slot_id:
            continue
        if x.ndim == 3:
            segment = x[batch]
        else:
            segment = x[:, start:end]
        if segment.shape[-1] == 0:
            continue

        slot = cache_ids[batch]
        valid_slot = slot is not None and slot != pad_slot_id and conv_states is not None
        if valid_slot and has_state[batch]:
            if state_width:
                history = conv_states[slot].float()[..., -state_width:]
            else:
                history = conv_states[slot][..., :0].float()
        else:
            history = torch.zeros((channels, state_width), dtype=torch.float32, device=x.device)

        raw_segment = segment.float()
        conv_input = torch.cat((history, raw_segment), dim=-1).unsqueeze(0)
        output = torch.nn.functional.conv1d(conv_input, conv_weight, conv_bias, groups=channels).squeeze(0)
        if activation in ("silu", "swish"):
            output = torch.nn.functional.silu(output)
        segment.copy_(output.to(segment.dtype))

        if valid_slot and state_width:
            updated_state = torch.cat((history, raw_segment), dim=-1)[..., -state_width:]
            conv_states[slot].copy_(updated_state.to(conv_states.dtype))
    return x


def _fused_recurrent_gated_delta_rule_mps(
    q,
    k,
    v,
    g,
    beta,
    scale,
    initial_state,
    inplace_final_state,
    cu_seqlens,
    ssm_state_indices,
    ssm_state_write_indices,
    num_accepted_tokens,
    use_qk_l2norm_in_kernel,
    A_log,
    dt_bias,
    a_raw,
    b_raw,
    out,
):
    batch, steps, query_heads, key_dim = q.shape
    value_heads, value_dim = v.shape[2:]
    if value_heads % query_heads:
        raise ValueError("The number of value heads must be divisible by the number of query heads.")
    if scale is None:
        scale = key_dim ** -0.5

    q_flat = q.reshape(-1, query_heads, key_dim)
    k_flat = k.reshape(-1, query_heads, key_dim)
    v_flat = v.reshape(-1, value_heads, value_dim)
    g_flat = g.reshape(-1, *g.shape[2:]) if g is not None else None
    beta_flat = beta.reshape(-1, *beta.shape[2:]) if beta is not None else None
    a_flat = a_raw.reshape(-1, a_raw.shape[-1]) if a_raw is not None else None
    b_flat = b_raw.reshape(-1, b_raw.shape[-1]) if b_raw is not None else None

    if out is None:
        # Prefill V is a transposed convolution view. The model flattens the
        # returned heads with view(), so preserve the CUDA kernel's contiguous
        # output contract instead of inheriting V's strides.
        output = torch.empty(v_flat.shape, dtype=v.dtype, device=v.device)
    else:
        output = out.reshape_as(v_flat)

    if (
        cu_seqlens is not None
        and not inplace_final_state
        and ssm_state_indices is None
        and ssm_state_write_indices is None
        and num_accepted_tokens is None
        and A_log is None
    ):
        from .recurrent import prefill

        result = prefill(
            q_flat,
            k_flat,
            v_flat,
            g_flat,
            beta_flat,
            initial_state,
            cu_seqlens,
            output,
            use_qk_l2norm_in_kernel,
            scale,
        )
        if result is not None:
            return result[0].view_as(v), result[1]

    if cu_seqlens is None:
        sequence_ranges = [(index * steps, (index + 1) * steps) for index in range(batch)]
    else:
        boundaries = cu_seqlens.detach().to(device="cpu", dtype=torch.long).tolist()
        sequence_ranges = list(zip(boundaries, boundaries[1:]))

    if ssm_state_indices is not None:
        read_indices = ssm_state_indices.detach().to(device="cpu", dtype=torch.long)
    else:
        read_indices = None
    if ssm_state_write_indices is not None:
        write_indices = ssm_state_write_indices.detach().to(device="cpu", dtype=torch.long)
    else:
        write_indices = read_indices
    accepted = (
        num_accepted_tokens.detach().to(device="cpu", dtype=torch.long).tolist()
        if num_accepted_tokens is not None
        else None
    )
    if inplace_final_state:
        if initial_state is None:
            raise NotImplementedError("In-place MPS recurrent gated delta rule requires an initial state tensor.")
        final_state = initial_state
    else:
        final_state = torch.empty(
            (len(sequence_ranges), value_heads, key_dim, value_dim),
            dtype=initial_state.dtype if initial_state is not None else v.dtype,
            device=v.device,
        )

    # Fuse ordinary recurrence over all tokens into one Metal dispatch. Keep
    # the reference implementation for speculative/state-aliasing layouts.
    if accepted is None and (not inplace_final_state or steps == 1):
        from .recurrent import recurrent

        if recurrent(
            q_flat,
            k_flat,
            v_flat,
            g_flat,
            beta_flat,
            initial_state,
            final_state,
            output,
            sequence_ranges,
            read_indices,
            write_indices,
            use_qk_l2norm_in_kernel,
            scale,
            A_log,
            dt_bias,
            a_flat,
            b_flat,
            inplace_final_state,
        ):
            return output.view_as(v), final_state

    head_map = torch.arange(value_heads, device=q.device) // (value_heads // query_heads)
    log_decay = A_log.reshape(-1).float() if A_log is not None else None
    dt_bias_fp32 = dt_bias.reshape(-1).float() if dt_bias is not None else None

    # Ordinary decode has one token per request and 1-D recurrent-state
    # indices. Process the whole batch together to avoid dispatching the
    # recurrent tensor operations once per request. Keep the sequential path
    # below for varlen/prefill, speculative layouts, and aliased state indices,
    # where write ordering is observable.
    if (
        inplace_final_state
        and cu_seqlens is None
        and steps == 1
        and accepted is None
        and (read_indices is None or read_indices.ndim == 1)
        and (write_indices is None or write_indices.ndim == 1)
    ):
        read_index_list = read_indices.tolist() if read_indices is not None else list(range(batch))
        write_index_list = write_indices.tolist() if write_indices is not None else read_index_list
        if batch == 0:
            return output.view_as(v), final_state
        if initial_state is None:
            raise NotImplementedError("In-place MPS recurrent gated delta rule requires an initial state tensor.")
        read_index_to_row = {index: row for row, index in enumerate(read_index_list)}
        writes_do_not_clobber_other_reads = all(
            write_index not in read_index_to_row or read_index_to_row[write_index] == row
            for row, write_index in enumerate(write_index_list)
        )
        if (
            len(set(read_index_list)) == batch
            and len(set(write_index_list)) == batch
            and writes_do_not_clobber_other_reads
        ):
            state_indices = (
                ssm_state_indices.to(device=v.device, dtype=torch.long)
                if ssm_state_indices is not None
                else torch.arange(batch, dtype=torch.long, device=v.device)
            )
            state = initial_state.index_select(0, state_indices).float()
            # Flattening already removed the single-token axis. Preserve all
            # heads here; selecting [:, 0] would discard the head dimension.
            q_token = q_flat.float().index_select(1, head_map)
            k_token = k_flat.float().index_select(1, head_map)
            v_token = v_flat.float()
            if use_qk_l2norm_in_kernel:
                q_token = q_token * torch.rsqrt(q_token.square().sum(dim=-1, keepdim=True) + 1e-6)
                k_token = k_token * torch.rsqrt(k_token.square().sum(dim=-1, keepdim=True) + 1e-6)
            q_token = q_token * scale

            if log_decay is not None:
                alpha = a_flat.float() + dt_bias_fp32.unsqueeze(0)
                decay_gate = -log_decay.exp().unsqueeze(0) * torch.nn.functional.softplus(alpha)
                beta_value = b_flat.float().sigmoid()
            else:
                decay_gate = g_flat.float()
                beta_value = beta_flat.float()
            if decay_gate.ndim == 1:
                decay_gate = decay_gate.unsqueeze(0).expand(batch, -1)
            state.mul_(decay_gate.exp()[:, :, None, None])

            predicted = (state * k_token.unsqueeze(-1)).sum(dim=-2)
            delta = v_token - predicted
            if beta_value.ndim == 2:
                delta = delta * beta_value.unsqueeze(-1)
            else:
                delta = delta * beta_value
            state.add_(k_token.unsqueeze(-1) * delta.unsqueeze(-2))
            output[:, :] = (state * q_token.unsqueeze(-1)).sum(dim=-2).to(output.dtype)

            write_state_indices = (
                ssm_state_write_indices.to(device=v.device, dtype=torch.long)
                if ssm_state_write_indices is not None
                else state_indices
            )
            final_state.index_copy_(0, write_state_indices, state.to(final_state.dtype))
            return output.view_as(v), final_state

    for sequence, (begin, end) in enumerate(sequence_ranges):
        if read_indices is None:
            state_index = sequence
        elif read_indices.ndim == 1:
            state_index = int(read_indices[sequence].item())
        else:
            state_step = accepted[sequence] - 1 if accepted is not None else 0
            state_index = int(read_indices[sequence, state_step].item())
        if initial_state is None:
            state = torch.zeros((value_heads, key_dim, value_dim), dtype=torch.float32, device=v.device)
        else:
            state = initial_state[state_index].float().clone()

        if begin == end:
            # Empty varlen sequences do not update their recurrent state. Keep
            # the input state in the returned batch instead of leaving this
            # row of final_state uninitialized (the caller writes all rows
            # back to the state cache after prefill).
            if not inplace_final_state:
                final_state[sequence].copy_(state.to(final_state.dtype))
            continue

        for token in range(begin, end):
            q_token = q_flat[token].float().index_select(0, head_map)
            k_token = k_flat[token].float().index_select(0, head_map)
            v_token = v_flat[token].float()
            if use_qk_l2norm_in_kernel:
                q_token = q_token * torch.rsqrt(q_token.square().sum(dim=-1, keepdim=True) + 1e-6)
                k_token = k_token * torch.rsqrt(k_token.square().sum(dim=-1, keepdim=True) + 1e-6)
            q_token = q_token * scale

            if log_decay is not None:
                alpha = a_flat[token].float() + dt_bias_fp32
                decay_gate = -log_decay.exp() * torch.nn.functional.softplus(alpha)
                beta_value = b_flat[token].float().sigmoid()
            else:
                decay_gate = g_flat[token].float()
                beta_value = beta_flat[token].float()
            if decay_gate.ndim == 0:
                decay_gate = decay_gate.expand(value_heads)
            state.mul_(decay_gate.exp().view(value_heads, 1, 1))

            predicted = torch.bmm(state.transpose(1, 2), k_token.unsqueeze(-1)).squeeze(-1)
            delta = v_token - predicted
            if beta_value.ndim == 0:
                delta = delta * beta_value
            elif beta_value.numel() == value_heads:
                delta = delta * beta_value.reshape(value_heads, 1)
            else:
                delta = delta * beta_value.reshape(value_heads, value_dim)
            state.add_(k_token.unsqueeze(-1) * delta.unsqueeze(1))
            output[token] = torch.bmm(state.transpose(1, 2), q_token.unsqueeze(-1)).squeeze(-1).to(output.dtype)

            if inplace_final_state:
                if write_indices is None:
                    write_index = state_index
                elif write_indices.ndim == 1:
                    write_index = int(write_indices[sequence].item())
                else:
                    write_index = int(write_indices[sequence, token - begin].item())
                final_state[write_index].copy_(state.to(final_state.dtype))

        if not inplace_final_state:
            final_state[sequence].copy_(state.to(final_state.dtype))

    return output.view_as(v), final_state


def select_kv_index_from_req(req_to_token_indexs, b_req_idx, b_seq_len):
    return req_to_token_indexs[b_req_idx.to(torch.long), b_seq_len.to(torch.long) - 1]


def select_kv_index_from_req_prefill(
    req_to_token_indexs,
    b_req_idx,
    b_seq_len,
    b_ready_cache_len,
    b_start_loc,
    max_q_seq_len,
    token_num,
):
    out_memindex = torch.empty((token_num,), dtype=torch.int32, device=req_to_token_indexs.device)
    req_rows, req_positions, flat_positions = _expand_mps_prefill_metadata(
        b_req_idx, b_seq_len, b_ready_cache_len, b_start_loc, token_num
    )
    out_memindex[flat_positions] = req_to_token_indexs[req_rows, req_positions]
    return out_memindex


def destindex_copy_kv(K, DestLoc, Out):
    from .scatter import copy_kv

    if copy_kv(K, DestLoc, Out):
        return
    dest_indexes = DestLoc.to(device=Out.device, dtype=torch.long)
    Out.index_copy_(0, dest_indexes, K)
    return


def gen_cumsum_pad0_tensor(b_q_seq_len: torch.Tensor, b_kv_seq_len: torch.Tensor):
    assert len(b_q_seq_len.shape) == 1
    assert b_q_seq_len.shape == b_kv_seq_len.shape
    assert b_q_seq_len.is_contiguous()
    b1_cu_q_seq_len = torch.empty((b_q_seq_len.shape[0] + 1,), dtype=torch.int32, device=b_q_seq_len.device)
    b1_cu_kv_seq_len = torch.empty((b_kv_seq_len.shape[0] + 1,), dtype=torch.int32, device=b_kv_seq_len.device)
    b1_cu_q_seq_len[0] = 0
    b1_cu_kv_seq_len[0] = 0
    b1_cu_q_seq_len[1:] = torch.cumsum(b_q_seq_len, dim=0, dtype=torch.int32)
    b1_cu_kv_seq_len[1:] = torch.cumsum(b_kv_seq_len, dim=0, dtype=torch.int32)
    return b1_cu_q_seq_len, b1_cu_kv_seq_len


def gen_prefill_params(input_token_num: int, b_ready_cache_len: torch.Tensor, b_seq_len: torch.Tensor):
    assert b_ready_cache_len.shape[0] == b_seq_len.shape[0]
    b_q_seq_len = b_seq_len - b_ready_cache_len
    b1_cu_q_seq_len, b1_cu_kv_seq_len = gen_cumsum_pad0_tensor(b_q_seq_len, b_seq_len)
    position_bias = b_ready_cache_len - b1_cu_q_seq_len[:-1]
    token_position_offsets = torch.repeat_interleave(
        position_bias,
        b_q_seq_len.to(torch.int64),
        output_size=input_token_num,
    )
    position_ids = torch.arange(input_token_num, dtype=torch.int32, device=b_seq_len.device)
    position_ids = position_ids + token_position_offsets
    assert position_ids.numel() == input_token_num
    return b_q_seq_len, b1_cu_q_seq_len, b_seq_len, b1_cu_kv_seq_len, position_ids


def gen_decode_params(b_seq_len: torch.Tensor):
    b_q_seq_len = torch.ones_like(b_seq_len)
    b1_cu_q_seq_len, b1_cu_kv_seq_len = gen_cumsum_pad0_tensor(b_q_seq_len, b_seq_len)
    return b_q_seq_len, b1_cu_q_seq_len, b_seq_len, b1_cu_kv_seq_len, b_seq_len - 1


def silu_and_mul_fwd(
    input: torch.Tensor,
    output: torch.Tensor,
    layout="blocked",
    limit=None,
    alpha=None,
    run_config=None,
    clamp_up_add_one=True,
):
    assert input.stride(-1) == 1
    assert output.is_contiguous()
    assert (limit is None and alpha is None) or (limit is not None and alpha is not None)
    if input.ndim != 2:
        raise ValueError("The MPS SiLU-and-multiply fallback expects a 2D input tensor.")
    size_n = input.shape[-1] // 2
    if layout == "interleaved":
        gate, up = input[:, 0::2], input[:, 1::2]
    elif layout == "blocked":
        gate, up = input[:, :size_n], input[:, size_n:]
    else:
        raise ValueError(f"Unsupported SiLU-and-multiply layout: {layout}")

    if limit is None and input.device.type == "mps" and not ffn_use_tanh_approximate_gelu():
        from .elementwise import silu_product

        if silu_product(input, output, layout):
            return

    gate_f = gate.float()
    if limit is not None and alpha is not None:
        gate_f = torch.clamp(gate_f, max=limit)
        up = torch.clamp(up, min=-limit, max=limit)
        gate_f = (gate_f * alpha).sigmoid() * gate_f
        gate = gate_f.to(input.dtype)
        if clamp_up_add_one:
            up = up + 1
    elif ffn_use_tanh_approximate_gelu():
        gate_cubed = gate_f * gate_f * gate_f
        tanh_arg = 0.7978845608028654 * (gate_f + 0.044715 * gate_cubed)
        gate = (0.5 * gate_f * (1.0 + torch.tanh(tanh_arg))).to(input.dtype)
    else:
        gate = (gate_f.sigmoid() * gate_f).to(input.dtype)

    output.copy_(up * gate)
    return


def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    query_start_loc: Optional[torch.Tensor] = None,
    cache_indices: Optional[torch.Tensor] = None,
    has_initial_state: Optional[torch.Tensor] = None,
    conv_states: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
    pad_slot_id: int = -1,
    **kwargs,
):
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    if x.stride(-1) != 1:
        x = x.contiguous()
    return _causal_conv1d_mps(
        x,
        weight,
        bias,
        query_start_loc,
        cache_indices,
        has_initial_state,
        conv_states,
        activation,
        pad_slot_id,
    )


def fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
    run_config: Optional[dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    from .gating import gating

    result = gating(A_log, a, b, dt_bias, beta=beta, threshold=threshold)
    if result is not None:
        return result
    a_fp32 = a.float() + dt_bias.float()
    g = -A_log.float().exp() * torch.nn.functional.softplus(a_fp32, beta=beta, threshold=threshold)
    # The Triton kernel rounds beta to b's dtype before storing it in the
    # float32 output buffer; preserve that behavior on MPS.
    beta_output = b.float().sigmoid().to(b.dtype).float()
    return g, beta_output


def conv_pack_gdn_decode_inputs(
    mixed_qkv: torch.Tensor,
    z_raw: torch.Tensor,
    a_raw: torch.Tensor,
    b_raw: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    conv_state_indices: torch.Tensor,
    activation: str,
    conv_size: int,
    num_k_heads: int,
    head_k_dim: int,
    num_v_heads: int,
    head_v_dim: int,
):
    batch = mixed_qkv.shape[0]
    q_dim = num_k_heads * head_k_dim
    k_dim = q_dim
    v_dim = num_v_heads * head_v_dim
    gate_dim = num_v_heads
    conv_dim = q_dim + k_dim + v_dim
    assert conv_size >= 2, f"conv kernel size must be at least 2, got {conv_size}"
    assert mixed_qkv.shape[1] == conv_dim, f"mixed_qkv shape mismatch: {mixed_qkv.shape[1]} != {conv_dim}"
    assert conv_weight.shape[0] == conv_dim, f"conv_weight shape mismatch: {conv_weight.shape[0]} != {conv_dim}"
    assert conv_weight.shape[1] == conv_size, f"conv_weight kernel mismatch: {conv_weight.shape[1]} != {conv_size}"
    assert conv_state.shape[1] == conv_dim, f"conv_state shape mismatch: {conv_state.shape[1]} != {conv_dim}"
    assert (
        conv_state.shape[2] >= conv_size - 1
    ), f"conv_state width must be at least conv_size - 1, got {conv_state.shape[2]} and {conv_size}"
    from .convolution import decode as conv_decode

    packed = conv_decode(mixed_qkv, conv_state, conv_weight, conv_bias, conv_state_indices, activation, conv_size)
    if packed is not None:
        q = packed[:, :q_dim].view(batch, 1, num_k_heads, head_k_dim)
        k = packed[:, q_dim : q_dim + k_dim].view(batch, 1, num_k_heads, head_k_dim)
        v = packed[:, q_dim + k_dim :].view(batch, 1, num_v_heads, head_v_dim)
        return (
            q,
            k,
            v,
            z_raw.reshape(batch, num_v_heads, head_v_dim),
            a_raw.reshape(batch, gate_dim),
            b_raw.reshape(batch, gate_dim),
        )

    state_indices = conv_state_indices.to(device="cpu", dtype=torch.long).tolist()
    conv_values = mixed_qkv.float()
    conv_weight_fp32 = conv_weight.float()
    conv_bias_fp32 = conv_bias.float() if conv_bias is not None else None
    if len(set(state_indices)) == len(state_indices):
        state_indexes = conv_state_indices.to(device=conv_state.device, dtype=torch.long)
        state_batch = conv_state.index_select(0, state_indexes)
        history_weight = conv_weight_fp32[:, : conv_size - 1].unsqueeze(0)
        conv_output = (state_batch[:, :, : conv_size - 1].float() * history_weight).sum(dim=-1)
        conv_output = conv_output + conv_values * conv_weight_fp32[:, conv_size - 1]
        if conv_bias_fp32 is not None:
            conv_output = conv_output + conv_bias_fp32
        if activation in ("silu", "swish"):
            conv_output = torch.nn.functional.silu(conv_output)

        updated_state = state_batch.clone()
        if conv_size > 2:
            updated_state[:, :, : conv_size - 2] = state_batch[:, :, 1 : conv_size - 1]
        updated_state[:, :, conv_size - 2] = mixed_qkv
        conv_state.index_copy_(0, state_indexes, updated_state)
    else:
        # Preserve the original sequential semantics if a caller aliases
        # multiple rows to the same recurrent state slot.
        conv_output = torch.empty_like(conv_values)
        for row, state_idx in enumerate(state_indices):
            state = conv_state[state_idx]
            current = conv_values[row]
            result = (state[:, : conv_size - 1].float() * conv_weight_fp32[:, : conv_size - 1]).sum(dim=-1)
            result = result + current * conv_weight_fp32[:, conv_size - 1]
            if conv_bias_fp32 is not None:
                result = result + conv_bias_fp32
            if activation in ("silu", "swish"):
                result = torch.nn.functional.silu(result)
            conv_output[row] = result

            if conv_size > 2:
                state[:, : conv_size - 2].copy_(state[:, 1 : conv_size - 1].clone())
            state[:, conv_size - 2].copy_(mixed_qkv[row])

    q = conv_output[:, :q_dim].view(batch, 1, num_k_heads, head_k_dim).to(mixed_qkv.dtype)
    k = conv_output[:, q_dim : q_dim + k_dim].view(batch, 1, num_k_heads, head_k_dim).to(mixed_qkv.dtype)
    v = conv_output[:, q_dim + k_dim :].view(batch, 1, num_v_heads, head_v_dim).to(mixed_qkv.dtype)
    z = z_raw.reshape(batch, num_v_heads, head_v_dim)
    a = a_raw.reshape(batch, gate_dim)
    b = b_raw.reshape(batch, gate_dim)
    return q, k, v, z, a, b


def mrope_triton_fused(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section: torch.Tensor,
    is_interleaved: bool,
    partial_rotary_factor: float = 1.0,
    run_config: Optional[dict] = None,
):
    from .rotary import rotary

    if rotary(q, k, cos, sin, mrope_section, is_interleaved, partial_rotary_factor):
        return
    head_dim = int(q.shape[2] * partial_rotary_factor)
    num_tokens = q.shape[0]
    section_t, section_h, section_w = mrope_section.detach().to(device="cpu").tolist()
    half_dim = head_dim // 2
    offsets = torch.arange(half_dim, device=q.device)
    if is_interleaved:
        h_mask = ((offsets % 3) == 1) & (offsets <= 3 * section_h)
        w_mask = ((offsets % 3) == 2) & (offsets <= 3 * section_w)
        t_mask = ~(h_mask | w_mask)
    else:
        h_end = section_t + section_h
        t_mask = offsets < section_t
        h_mask = (offsets >= section_t) & (offsets < h_end)
        w_mask = offsets >= h_end

    cos = (
        cos[0, :num_tokens, :half_dim] * t_mask
        + cos[1, :num_tokens, :half_dim] * h_mask
        + cos[2, :num_tokens, :half_dim] * w_mask
    )
    sin = (
        sin[0, :num_tokens, :half_dim] * t_mask
        + sin[1, :num_tokens, :half_dim] * h_mask
        + sin[2, :num_tokens, :half_dim] * w_mask
    )
    cos = cos[:, None, :]
    sin = sin[:, None, :]

    def apply_rotary(tensor: torch.Tensor) -> None:
        first = tensor[..., :half_dim].float()
        second = tensor[..., half_dim:head_dim].float()
        rotated_first = first * cos - second * sin
        rotated_second = first * sin + second * cos
        tensor[..., :half_dim].copy_(rotated_first.to(tensor.dtype))
        tensor[..., half_dim:head_dim].copy_(rotated_second.to(tensor.dtype))

    apply_rotary(q)
    apply_rotary(k)
    return


def sigmoid_mul_(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    x_arg = x.view(-1, x.shape[-1])
    gate_arg = gate.view(-1, gate.shape[-1])
    assert gate_arg.shape[0] == x_arg.shape[0] and gate_arg.shape[1] in (1, x_arg.shape[1])
    from .elementwise import sigmoid_product

    if sigmoid_product(x_arg, gate_arg):
        return x
    rounded_gate = gate_arg.float().sigmoid().to(gate_arg.dtype)
    x_arg.mul_(rounded_gate)
    return x


def fused_recurrent_gated_delta_rule(
    q,
    k,
    v,
    g=None,
    beta=None,
    scale=None,
    initial_state=None,
    inplace_final_state=True,
    cu_seqlens=None,
    ssm_state_indices=None,
    ssm_state_write_indices=None,
    num_accepted_tokens=None,
    use_qk_l2norm_in_kernel=False,
    A_log=None,
    dt_bias=None,
    a_raw=None,
    b_raw=None,
    out=None,
):
    if (
        inplace_final_state
        and initial_state is not None
        and q.shape[1] == 1
        and cu_seqlens is None
        and ssm_state_indices is not None
        and ssm_state_indices.ndim == 1
        and ssm_state_write_indices is None
        and num_accepted_tokens is None
    ):
        from .recurrent import decode

        result = decode(
            q,
            k,
            v,
            g,
            beta,
            initial_state,
            ssm_state_indices,
            use_qk_l2norm_in_kernel,
            scale,
            A_log,
            dt_bias,
            a_raw,
            b_raw,
            out,
        )
        if result is not None:
            return result
    return _fused_recurrent_gated_delta_rule_mps(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        inplace_final_state,
        cu_seqlens,
        ssm_state_indices,
        ssm_state_write_indices,
        num_accepted_tokens,
        use_qk_l2norm_in_kernel,
        A_log,
        dt_bias,
        a_raw,
        b_raw,
        out,
    )


def tensor_copy_3dim(dest_tensor, source_tensor):
    dest_tensor.copy_(source_tensor)


def rmsnorm_forward(x, weight, eps, out=None, **kwargs):
    from .normalization import rms

    result = rms(x, weight, eps, out=out)
    if result is not None:
        return result
    value = x.float()
    value = (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps) * weight.float()).to(x.dtype)
    if out is None:
        return value
    out.copy_(value)
    return out


def layernorm_forward(x, weight, bias, eps, **kwargs):
    return torch.nn.functional.layer_norm(x, [x.shape[-1]], weight, bias, eps)


def qk_rmsnorm_fused_forward(q, k, w_q, w_k, eps, **kwargs):
    for tensor, weight in ((q, w_q), (k, w_k)):
        view = tensor.reshape(-1, weight.numel())
        tensor.copy_(rmsnorm_forward(view, weight, eps).reshape_as(tensor))


def gated_rmsnorm_forward(x, weight, bias, eps, z, out=None, gate_type="silu", **kwargs):
    from .normalization import rms

    result = rms(x, weight, eps, out=out, z=z, gate=gate_type)
    if result is not None:
        return result
    value = rmsnorm_forward(x.float(), weight.float(), eps)
    gate = torch.nn.functional.silu(z.float()) if gate_type == "silu" else z.float().sigmoid()
    value = (value * gate).to(x.dtype)
    if out is None:
        return value
    out.copy_(value)
    return out


def embedding(input_ids, weight, vob_start_id, vob_end_id, out):
    ids = input_ids.long() - vob_start_id
    valid = (ids >= 0) & (ids < vob_end_id - vob_start_id)
    out.copy_(torch.nn.functional.embedding(ids.clamp(0, weight.shape[0] - 1), weight))
    out.mul_(valid[:, None])
    return out


def multimodal_emb(
    out,
    prompt_ids,
    text_weight_embs,
    embed_cache,
    img_token_lens,
    img_start_token_ids,
    img_start_locs_in_cache,
    tp_text_start_token_id,
    tp_text_end_token_id,
    tp_world_size,
    text_embed_scale=1.0,
):
    if img_token_lens.numel():
        raise NotImplementedError("Metal Qwen3.5 currently supports text inputs only")
    embedding(prompt_ids, text_weight_embs, tp_text_start_token_id, tp_text_end_token_id, out)
    out.mul_(text_embed_scale)


def copy_kv_index_to_req(req_to_token_indexs, b_req_idx, b_seq_len, memindex):
    req_to_token_indexs[b_req_idx.long(), b_seq_len.long() - 1] = memindex.to(req_to_token_indexs.dtype)


def copy_kv_index_to_req_prefill(
    req_to_token_indexs, b_req_idx, b_seq_len, b_ready_cache_len, b_start_loc, memindex, max_q_seq_len
):
    rows, positions, flat = _expand_mps_prefill_metadata(
        b_req_idx, b_seq_len, b_ready_cache_len, b_start_loc, memindex.numel()
    )
    req_to_token_indexs[rows, positions] = memindex[flat].to(req_to_token_indexs.dtype)


def gather_token(req_to_next_token_ids, b_req_idx, b_mtp_index):
    return req_to_next_token_ids[b_req_idx.long(), b_mtp_index.long()]


def gather_token_prefill_decode_mixed(
    input_ids, req_to_next_token_ids, b_req_idx, b_mtp_index, b_is_decode_req, b_prefill_start_loc
):
    from .tokens import gather_mixed

    if not gather_mixed(input_ids, req_to_next_token_ids, b_req_idx, b_mtp_index, b_is_decode_req, b_prefill_start_loc):
        active = b_is_decode_req.bool()
        values = req_to_next_token_ids[b_req_idx[active].long(), b_mtp_index[active].long()]
        input_ids[b_prefill_start_loc[active].long()] = values.to(input_ids.dtype)
    return input_ids


def scatter_token(next_token_ids, req_to_next_token_ids, b_req_idx, b_mtp_index, b_has_out=None):
    rows, cols = b_req_idx.long(), b_mtp_index.long()
    if b_has_out is not None:
        mask = b_has_out.bool()
        rows, cols, next_token_ids = rows[mask], cols[mask], next_token_ids[mask]
    req_to_next_token_ids[rows, cols] = next_token_ids.to(req_to_next_token_ids.dtype)


def token_id_counter(prompt_ids, out_token_id_counter):
    ids = prompt_ids.long()
    ids = ids[(ids >= 0) & (ids < out_token_id_counter.numel())]
    out_token_id_counter.index_add_(0, ids, torch.ones_like(ids, dtype=out_token_id_counter.dtype))


def update_req_to_token_id_counter(b_req_idx, next_token_ids, req_to_out_token_id_counter, mask=None):
    from .elementwise import update_counter

    if update_counter(b_req_idx, next_token_ids, req_to_out_token_id_counter, mask):
        return
    rows, cols = b_req_idx.long(), next_token_ids.long()
    if mask is not None:
        rows, cols = rows[mask.bool()], cols[mask.bool()]
    valid = (cols >= 0) & (cols < req_to_out_token_id_counter.shape[1])
    rows, cols = rows[valid], cols[valid]
    req_to_out_token_id_counter.index_put_(
        (rows, cols), torch.ones_like(rows, dtype=req_to_out_token_id_counter.dtype), accumulate=True
    )


def _eos_penalty(Logits, b_req_idx, b_length_penalty_param, b_mask_eos_reqs, eos_ids, manager):
    ids = eos_ids.long()
    decay = manager.req_to_exponential_decay_length_penalty[b_req_idx.long()]
    scale = decay.pow(b_length_penalty_param) - 1
    eos_logits = Logits[:, ids]
    eos_logits = eos_logits + eos_logits.abs() * scale[:, None]
    Logits[:, ids] = eos_logits.masked_fill(b_mask_eos_reqs.bool()[:, None], -10000000.0)


def apply_penalty_gpu_cache(
    Logits, b_req_idx, b_length_penalty_param, b_mask_eos_reqs, eos_ids, sampling_params_manager
):
    m = sampling_params_manager
    rows = b_req_idx.long()
    counts = m.req_to_out_token_id_counter[rows].to(Logits.device)
    repetition = m.req_to_repetition_penalty[rows, None]
    adjusted = torch.where(Logits > 0, Logits / repetition, Logits * repetition)
    Logits.copy_(
        torch.where(counts > 0, adjusted, Logits)
        - counts * m.req_to_frequency_penalty[rows, None]
        - (counts > 0) * m.req_to_presence_penalty[rows, None]
    )
    _eos_penalty(Logits, b_req_idx, b_length_penalty_param, b_mask_eos_reqs, eos_ids, m)


def apply_invalid_token_ids(Logits, invalid_token_ids, cu_invalid_token_num):
    boundaries = cu_invalid_token_num.cpu().tolist()
    for row, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        Logits[row, invalid_token_ids[start:end].long()] = -float("inf")


def gather_columns(input, dim, index):
    # macOS 14's MPSGraph gather can materialize a quadratic temporary for a
    # large vocabulary and abort at 4 GiB. Advanced indexing selects directly.
    if input.ndim != 2 or dim != 1:
        raise ValueError("Metal sampling gather expects a matrix along dimension 1")
    rows = torch.arange(input.shape[0], device=input.device)[:, None]
    return input[rows, index.long()]


def apply_penalty(
    Logits,
    b_req_idx,
    b_length_penalty_param,
    b_mask_eos_reqs,
    p_token_ids,
    p_token_counts,
    p_cumsum_seq_len,
    eos_ids,
    sampling_params_manager,
):
    m = sampling_params_manager
    starts = p_cumsum_seq_len.cpu().tolist()
    for row, (start, end) in enumerate(zip(starts, starts[1:])):
        ids = p_token_ids[start:end].long()
        counts = p_token_counts[start:end]
        req = b_req_idx[row].long()
        values = Logits[row, ids]
        repetition = m.req_to_repetition_penalty[req]
        values = torch.where(values > 0, values / repetition, values * repetition)
        Logits[row, ids] = values - counts * m.req_to_frequency_penalty[req] - m.req_to_presence_penalty[req]
    _eos_penalty(Logits, b_req_idx, b_length_penalty_param, b_mask_eos_reqs, eos_ids, m)


def copy_linear_att_state_to_kv_buffer(
    b_req_idx, big_page_buffer_ids, gpu_conv_state, gpu_ssm_state, cpu_kv_conv_state, cpu_kv_ssm_state, mtp_step
):
    width = cpu_kv_conv_state.shape[-1]
    for req, slot in zip(b_req_idx.cpu().tolist(), big_page_buffer_ids.cpu().tolist()):
        if slot >= 0:
            cpu_kv_conv_state[slot].copy_(gpu_conv_state[:, req, :, :width])
            cpu_kv_ssm_state[slot].copy_(gpu_ssm_state[:, req * (mtp_step + 1)])
