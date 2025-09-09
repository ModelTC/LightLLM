#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <ATen/ATen.h>
#include <THC/THCAtomics.cuh>

__device__ __forceinline__ int32_t index(int32_t total_col, int32_t row,
                                         int32_t col) {
  return row * total_col + col;
}

__global__ void check_kernel(
    const int32_t* input,           // [num_vocab, max_sequence_len]
    const int32_t* sequence_len,    // [num_vocab]
    const int32_t* shift_table,         // [node_num, symbol_num, max_edge_num_for_same_t]
    const int32_t* edge_num_table,      // [node_num, symbol_num]
    const int32_t* push_table,          // [edge_num, max_push_len]
    const int32_t* pop_table,           // [edge_num, max_pop_len]
    const int32_t* dest_table,          // [edge_num]
    const int32_t* cur_stack,
    int32_t* state_list,
    int cur_stack_top,
    int num_seqs,
    int num_symbol,
    int num_node, 
    int max_seq_len,
    int max_edge_num_for_same_t,
    int max_stack_depth,
    int max_pop_len,
    int max_push_len
) {
    int seq_idx = blockIdx.x * blockDim.x + threadIdx.x;  // 每个线程对应一个输入序列
    int tid = threadIdx.x;
    if (seq_idx >= num_seqs) return;
    int seq_len = sequence_len[seq_idx];  // 输入序列长度

    int stack_top = 0;
    // extern __shared__ int32_t stack[];
    int32_t stack[64];

    int cur_state = state_list[seq_idx];

    // 初始化栈
    for (int i = 0; i < cur_stack_top; ++i) {
        stack[i] = cur_stack[i];  // 初始状态
    }
    stack_top = cur_stack_top;
    for (int i = 0; i < seq_len; ++i) {
        if (cur_state == -1) break;  // 如果状态已经变为无效，结束循环

        int input_symbol = input[seq_idx * max_seq_len + i];
        int edge_num = edge_num_table[index(num_symbol, cur_state, input_symbol)];
        int edge_start_index = cur_state * num_symbol * max_edge_num_for_same_t + input_symbol * max_edge_num_for_same_t;
        bool match = false;

        for (int edge_idx = 0; edge_idx < edge_num; ++edge_idx) {
            int edge = shift_table[edge_start_index + edge_idx];
            const int32_t* push_list = push_table + edge * max_push_len;
            const int32_t* pop_list = pop_table + edge * max_pop_len;
            int dest_state = dest_table[edge];

            // 检查栈顶是否匹配 pop_list
            bool check = true;
            for (int j = 0; j < max_pop_len && pop_list[j] != -1; ++j) {
                //if (stack_top - j - 1 < 0 || stack[index(max_stack_depth, tid, stack_top - j - 1)] != pop_list[j]) {
                if (stack_top - j - 1 < 0 || stack[stack_top - j - 1] != pop_list[j]) {
                    check = false;
                    break;
                }
            }
            if (check && !match) {
                cur_state = dest_state;
                // 执行 pop 操作
                for (int j = 0; j < max_pop_len && pop_list[j] != -1; ++j) {
                    --stack_top;
                }
                // 执行 push 操作
                for (int j = 0; j < max_push_len && push_list[j] != -1; ++j) {
                    // stack[index(max_stack_depth, tid, stack_top)] = push_list[j];
                    stack[stack_top] = push_list[j];
                    stack_top++;
                }

                match = true;
                break;
            }
        }

        if (!match) {
            cur_state = -1;  // 无法匹配，设置为无效状态
            break;
        }
    }

    // 存储最终的状态和栈指针
    state_list[seq_idx] = cur_state;
}

void check_dpda(
    torch::Tensor input_sequences,
    torch::Tensor sequence_len,
    torch::Tensor shift_table,
    torch::Tensor edge_num_table,
    torch::Tensor push_table,
    torch::Tensor pop_table,
    torch::Tensor dest_table,
    torch::Tensor current_stack,
    torch::Tensor start_state,
    int max_stack_depth
) {
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int num_seqs = input_sequences.size(0);
    const int max_seq_len = input_sequences.size(1);
    const int num_node = edge_num_table.size(0);
    const int num_symbol = edge_num_table.size(1);
    const int max_edge_num_for_same_t = shift_table.size(2);
    const int max_pop_len = pop_table.size(1);
    const int max_push_len = push_table.size(1);

    check_kernel<<<(num_seqs + 255) / 256, 256, 0, stream>>>(
        input_sequences.data_ptr<int>(),
        sequence_len.data_ptr<int>(),
        shift_table.data_ptr<int>(),
        edge_num_table.data_ptr<int>(),
        push_table.data_ptr<int>(),
        pop_table.data_ptr<int>(),
        dest_table.data_ptr<int>(),
        current_stack.data_ptr<int>(),
        start_state.data_ptr<int>(),
        current_stack.size(0),
        num_seqs,
        num_symbol,
        num_node,
        max_seq_len,
        max_edge_num_for_same_t,
        max_stack_depth,
        max_pop_len,
        max_push_len
    );
}

__global__ void batched_check_kernel(
    const int32_t* input,           // [num_seqs, max_sequence_len]
    const int32_t* sequence_len,    // [num_seqs]
    const int32_t* shift_table,         // [node_num, symbol_num, max_edge_num_for_same_t]
    const int32_t* edge_num_table,      // [node_num, symbol_num]
    const int32_t* push_table,          // [edge_num, max_push_len]
    const int32_t* pop_table,           // [edge_num, max_pop_len]
    const int32_t* dest_table,          // [edge_num]
    int32_t* cur_stack,
    int32_t* start_state_list,
    int32_t* cur_stack_top,
    int32_t* output,
    int num_batch,
    int num_seqs,
    int num_symbol,
    int num_node, 
    int max_seq_len,
    int max_edge_num_for_same_t,
    int max_stack_depth,
    int max_pop_len,
    int max_push_len,
    const int seq_per_thread,
    const int block_size
) {
    int batch_idx = blockIdx.x;
    int seq_idx_start = (blockIdx.y * block_size) + threadIdx.x * seq_per_thread;  // 每个线程对应一个输入序列
    int seq_idx_end = seq_idx_start + seq_per_thread;
    if (batch_idx >= num_batch) return;
    int32_t stack[64];
    // if(batch_idx == 0) printf("%d %d %d %d %d %d\n", batch_idx, seq_idx_start, seq_idx_end, blockIdx.y, threadIdx.x, blockDim.x);
    // extern __shared__ int32_t stack[];

    for (int seq_idx = seq_idx_start; seq_idx < seq_idx_end; seq_idx++) {
        if (seq_idx >= num_seqs) return;
        int seq_len = sequence_len[seq_idx];  // 输入序列长度
        int stack_top = cur_stack_top[batch_idx];
        
        int cur_state = start_state_list[batch_idx];
        // 初始化栈
        for (int i = 0; i < stack_top; ++i) {
            stack[i] = cur_stack[batch_idx * max_stack_depth + i];  // 初始状态
        }

        for (int i = 0; i < seq_len; ++i) {
            if (cur_state == -1) break;  // 如果状态已经变为无效，结束循环

            int input_symbol = input[seq_idx * max_seq_len + i];
            int edge_num = edge_num_table[index(num_symbol, cur_state, input_symbol)];
            int edge_start_index = cur_state * num_symbol * max_edge_num_for_same_t + input_symbol * max_edge_num_for_same_t;
            bool match = false;

            for (int edge_idx = 0; edge_idx < edge_num; ++edge_idx) {
                int edge = shift_table[edge_start_index + edge_idx];
                const int32_t* push_list = push_table + edge * max_push_len;
                const int32_t* pop_list = pop_table + edge * max_pop_len;
                int dest_state = dest_table[edge];

                // 检查栈顶是否匹配 pop_list
                bool check = true;
                for (int j = 0; j < max_pop_len && pop_list[j] != -1; ++j) {
                    //if (stack_top - j - 1 < 0 || stack[index(max_stack_depth, tid, stack_top - j - 1)] != pop_list[j]) {
                    if (stack_top - j - 1 < 0 || stack[stack_top - j - 1] != pop_list[j]) {
                        check = false;
                        break;
                    }
                }
                if (check && !match) {
                    cur_state = dest_state;
                    // 执行 pop 操作
                    for (int j = 0; j < max_pop_len && pop_list[j] != -1; ++j) {
                        --stack_top;
                    }
                    // 执行 push 操作
                    for (int j = 0; j < max_push_len && push_list[j] != -1; ++j) {
                        // stack[index(max_stack_depth, tid, stack_top)] = push_list[j];
                        stack[stack_top] = push_list[j];
                        stack_top++;
                    }

                    match = true;
                    break;
                }
            }

            if (!match) {
                cur_state = -1;  // 无法匹配，设置为无效状态
                break;
            }
        }

        // 存储最终的状态和栈指针
        output[batch_idx * num_seqs + seq_idx] = cur_state;
    }
}

void batched_check_dpda(
    torch::Tensor input_sequences,
    torch::Tensor sequence_len,
    torch::Tensor shift_table,
    torch::Tensor edge_num_table,
    torch::Tensor push_table,
    torch::Tensor pop_table,
    torch::Tensor dest_table,
    torch::Tensor current_stack,            // [batch_size, max_stack_depth]
    torch::Tensor current_stack_top,        // [batch_size]
    torch::Tensor start_state,              // [batch_size * num_seqs]
    torch::Tensor output,
    int max_stack_depth
) {
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int num_batch = current_stack_top.size(0);
    const int num_seqs = input_sequences.size(0);
    const int max_seq_len = input_sequences.size(1);
    const int num_node = edge_num_table.size(0);
    const int num_symbol = edge_num_table.size(1);
    const int max_edge_num_for_same_t = shift_table.size(2);
    const int max_pop_len = pop_table.size(1);
    const int max_push_len = push_table.size(1);

    // const int block_dim = (num_seqs + 255) / 256;
    // const int block_size = 256;
    // const int batch_size = current_stack.size(0);

    const int seq_per_thread = 32;
    const int block_size = seq_per_thread * 512;
    // printf("%d %d\n", num_seqs, sequence_len.size(0));
    dim3 grid(current_stack.size(0), (num_seqs + block_size - 1) / block_size, 1);

    batched_check_kernel<<<grid, block_size / seq_per_thread, 0, stream>>>(
        input_sequences.data_ptr<int>(),
        sequence_len.data_ptr<int>(),
        shift_table.data_ptr<int>(),
        edge_num_table.data_ptr<int>(),
        push_table.data_ptr<int>(),
        pop_table.data_ptr<int>(),
        dest_table.data_ptr<int>(),
        current_stack.data_ptr<int>(),
        start_state.data_ptr<int>(),
        current_stack_top.data_ptr<int>(),
        output.data_ptr<int>(),
        num_batch,
        num_seqs,
        num_symbol,
        num_node,
        max_seq_len,
        max_edge_num_for_same_t,
        max_stack_depth,
        max_pop_len,
        max_push_len,
        seq_per_thread,
        block_size
    );
}

PYBIND11_MODULE(lightllm_constraint_decode_kernel, m) {
  m.def("check_dpda", &check_dpda, "check_dpda");
  m.def("batched_check_dpda", &batched_check_dpda, "batched_check_dpda");
}
