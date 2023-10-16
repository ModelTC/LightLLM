def init_bloc(b_loc, b_seq_len, max_len_in_batch, alloc_mem_index):
    start_index = 0
    b_seq_len_numpy = b_seq_len.cpu().numpy()
    for i in range(len(b_seq_len)):
        cur_seq_len = b_seq_len_numpy[i]
        b_loc[i, max_len_in_batch - cur_seq_len:max_len_in_batch] = alloc_mem_index[start_index:start_index + cur_seq_len]
        start_index += cur_seq_len
    return