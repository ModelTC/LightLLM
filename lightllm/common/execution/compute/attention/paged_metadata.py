import torch


def prepare_page_table(metadata, batch, limits, scratch, page_size):
    from lightllm.common.basemodel.triton_kernel.fa3_utils import page_table_copy

    width = (limits.max_seq_len + page_size - 1) // page_size
    metadata.page_table = scratch.view(torch.int32)[: batch.batch_size * width].view(batch.batch_size, width)
    active_width = (batch.max_kv_len + page_size - 1) // page_size
    page_table_copy(metadata.page_table[:, :active_width], metadata.req_to_tokens, metadata.req_indices, page_size)
