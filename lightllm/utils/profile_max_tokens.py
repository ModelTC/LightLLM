import os
import gc
import json
import torch
from transformers import AutoModelForCausalLM
import argparse
from lightllm.common.build_utils import repair_config
from lightllm.utils.dist_utils import get_current_device_id

data_type_dict = {"float32": 4, "float16": 2, "bfloat16": 2, "fp32": 4, "fp16": 2, "bf16": 2, "int8": 1, "int4": 0.5}


def get_available_gpu_memory(world_size):
    """
    Get available memory.
    """
    torch.cuda.empty_cache()
    free_gpu_memory, _ = torch.cuda.mem_get_info(get_current_device_id())
    if world_size > 1:
        tensor = torch.tensor(free_gpu_memory, dtype=torch.float32).to(f"cuda:{get_current_device_id()}")
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MIN)
        free_gpu_memory = tensor.item()
    return free_gpu_memory / (1024 ** 3)


def get_total_gpu_memory():
    """
    Get the total GPU memory of the machine
    """
    total_memory = torch.cuda.get_device_properties(0).total_memory
    return total_memory / (1024 ** 3)  # Convert to GB


def load_config(weight_dir_):
    """
    Load model configuration from the specified directory
    Args:
        weight_dir_: Path to the weight directory
    Returns:
        config: Model configuration
    """
    with open(os.path.join(weight_dir_, "config.json"), "r") as json_file:
        config = json.load(json_file)
    repair_config(config, same_names=["num_attention_heads", "n_head"])
    repair_config(config, same_names=["hidden_size", "n_embd", "n_embed"])
    repair_config(config, same_names=["num_hidden_layers", "n_layer"])
    return config


def load_model(model_dir, tp_size, data_type):
    """
    Load the model and apply parallelization
    Args:
        model_dir: Model directory
        tp_size: Tensor Parallel size (number of GPUs)
        data_type: Data type (float32, float16, bf16)
    """
    torch.cuda.empty_cache()
    gc.collect()

    # Memory usage before loading the model
    before_memory = torch.cuda.memory_allocated()
    # Load the model
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(tp_size))
    _ = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    # Memory usage after loading the model
    after_memory = torch.cuda.memory_allocated()
    model_size = (after_memory - before_memory) / (1024 ** 3)  # Convert to GB
    model_size = model_size / 2 * data_type_dict[data_type]
    return model_size


def get_per_kv_cache_size(weight_dir_, tp_size=1, data_type="bf16"):
    """
    Calculate the memory requirement for KV Cache
    Args:
        weight_dir_: Path to the weight directory
        tp_size: Tensor Parallel size (number of GPUs)
        data_type: Data type for KV Cache
    Returns:
        per_kv_cache: Memory requirement for KV Cache per GPU in GB
    """
    # Get model configuration
    config = load_config(weight_dir_)
    hidden_size = config["hidden_size"]
    num_attention_heads = config["num_attention_heads"]
    head_dim = hidden_size // num_attention_heads
    n_layer = config["n_layer"]
    num_kv_heads = config.get("num_key_value_heads", num_attention_heads)

    # Total KV cache size
    per_kv_cache = 2 * n_layer * num_kv_heads * head_dim * data_type_dict[data_type]
    # If using Tensor Parallel, divide by the number of GPUs
    per_kv_cache /= tp_size
    return per_kv_cache * 1.0 / (1024 ** 3)  # Convert to GB


def get_total_token_nums(model_dir, tp_size, weight_data_type, kv_data_type, mem_fraction):
    """
    Calculate the maximum number of tokens that can be processed
    Args:
        model_dir: Model directory
        tp_size: Tensor Parallel size (number of GPUs)
        weight_data_type: Data type for model parameters
        kv_data_type: Data type for KV Cache
        mem_fraction: Fraction of memory usage
    """
    # Load model and calculate size
    model_size = load_model(model_dir, tp_size, weight_data_type)
    print(f"Model size (determined by GPU memory usage): {model_size:.2f} GB")
    gpu_total_size = get_total_gpu_memory()
    print(f"One GPU total size: {gpu_total_size:.2f} GB")
    # Calculate KV cache size
    kv_cache_size = get_per_kv_cache_size(model_dir, tp_size=tp_size, data_type=kv_data_type)
    print(f"KV Cache size per token for one GPU (TP size {tp_size}): {kv_cache_size:.6f} GB")

    max_total_token_num = (gpu_total_size * mem_fraction - model_size) / kv_cache_size
    print("The recommended max_total_token_num is", int(max_total_token_num))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--tp", type=int, default=1, help="Tensor Parallel size (number of GPUs)")
    parser.add_argument(
        "--weight_data_type",
        type=str,
        default="bf16",
        choices=["float32", "float16", "bfloat16", "fp32", "fp16", "bf16", "int8", "int4"],
        help="Data type for model parameters",
    )
    parser.add_argument("--mem_fraction", type=float, default=0.9, help="Fraction memory usage.")
    parser.add_argument(
        "--kv_data_type",
        type=str,
        default="bf16",
        choices=["float32", "float16", "bfloat16", "fp32", "fp16", "bf16", "int8", "int4"],
        help="Data type for KV Cache",
    )
    args = parser.parse_args()
    model_dir = args.model_dir
    tp_size = args.tp
    weight_data_type = args.weight_data_type
    kv_data_type = args.kv_data_type
    mem_fraction = args.mem_fraction
    get_total_token_nums(model_dir, tp_size, weight_data_type, kv_data_type, mem_fraction)


if __name__ == "__main__":
    main()
