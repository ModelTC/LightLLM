import os
import sys
from model_infer_batchs import test_model_inference
from process_utils import kill_gpu_processes
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from datetime import datetime


from lightllm.models.bloom.model import BloomTpPartModel
from lightllm.models.llama.model import LlamaTpPartModel
from lightllm.models.llama_wquant.model import LlamaTpPartModelWQuant
from lightllm.models.starcoder.model import StarcoderTpPartModel
from lightllm.models.starcoder_wquant.model import StarcoderTpPartModelWQuant
from lightllm.models.qwen.model import QWenTpPartModel
from lightllm.models.qwen_wquant.model import QWenTpPartModelWQuant
from lightllm.models.baichuan7b.model import Baichuan7bTpPartModel
from lightllm.models.baichuan13b.model import Baichuan13bTpPartModel
from lightllm.models.baichuan2_7b.model import Baichuan2_7bTpPartModel
from lightllm.models.chatglm2.model import ChatGlm2TpPartModel
from lightllm.models.internlm.model import InternlmTpPartModel
from lightllm.models.yi.model import YiTpPartModel


base_dir = "/nvme/"

model_to_class_and_path = {
    "llama-7b" : (LlamaTpPartModel, os.path.join(base_dir, "llama-7b")),
    "llama-13b" :(LlamaTpPartModel, os.path.join(base_dir, "")),
    "internal-20b" : (InternlmTpPartModel, os.path.join(base_dir, "")),
    "llama-65b" : (LlamaTpPartModel, os.path.join(base_dir, "")),
    "llama2-70b" : (LlamaTpPartModel, os.path.join(base_dir, "")),
    "qwen-7b" : (QWenTpPartModelWQuant, os.path.join(base_dir, "")),
    "qwen-14b" : (QWenTpPartModelWQuant, os.path.join(base_dir, "")),
    "chatglm2-6b" : (ChatGlm2TpPartModel, os.path.join(base_dir, ""))
}

def test_all_setting(gpu_name, model_name, mode, log_dir, world_sizes, in_out_lens, batch_sizes):
    log_dir = os.path.join(log_dir, gpu_name, str(model_name))
    os.makedirs(log_dir, exist_ok=True)

    model_class, model_path = model_to_class_and_path[model_name]
    kill_gpu_processes()
    for world_size in world_sizes:
        for in_len, out_len in in_out_lens: 
            kill_gpu_processes()
            mode_str = "_".join(mode)
            log_file_name = f"{model_name}##{mode_str}##{world_size}##{in_len}##{out_len}##batch_size##.log"
            log_path = os.path.join(log_dir, log_file_name)
            print(log_path)
            test_model_inference(world_size, 
                                 model_path, 
                                 model_class, 
                                 batch_sizes, 
                                 in_len, 
                                 out_len,
                                 mode, 
                                 log_path)
    log_md_file = log_dir + ".md"
    md_file = open(log_md_file, "w")
    # write head
    heads = ['mode', 'world_size', 'batch_size', 'input_len', 'output_len', 'prefill_cost', 'first_step_latency', 'last_step_latency', 'mean_latency', 
             'prefill_throughput', 'decode_throughput', 'total_throughput',
             'card_num_per_qps']
    md_file.write(f"test model: {model_name} \r\n")
    md_file.write('|')
    for head in heads:
        md_file.write(head + "|")
    md_file.write('\r\n')
    md_file.write('|')
    for _ in range(len(heads)):
        md_file.write('------|')
    md_file.write('\r\n')
    log_files = list(os.listdir(log_dir))
    sorted(log_files, key=lambda x: tuple(map(int, x.split("##")[2:6])))
    for log_file in log_files:
        _, mode, world_size, input_len, output_len, batch_size, _ = log_file.split("##")
        fp_file = open(os.path.join(log_dir, log_file), "r") 
        all_lines = fp_file.readlines()
        fp_file.close()
        if len(all_lines) <= 2:
            continue
        prefill_cost = float(all_lines[0].split(":")[1].strip())
        firststep_cost = float(all_lines[1].split(":")[1].strip())
        laststep_cost = float(all_lines[-2].split(":")[1].strip())
        all_step_cost = float(all_lines[-1].split(":")[1].strip())
        mean_step_cost = (all_step_cost - prefill_cost) / float(output_len)
        card_num_per_qps =  float(world_size) / (float(batch_size) / (all_step_cost / 1000))
        prefill_throughput = float(batch_size) * float(input_len) / (prefill_cost / 1000)
        decode_throughput = float(batch_size) * float(output_len) / ((all_step_cost - prefill_cost) / 1000)
        total_throughput = float(batch_size) * (float(input_len) + float(output_len)) / (all_step_cost / 1000)
        md_file.write('|')
        infos = [mode, world_size, batch_size, input_len, output_len, prefill_cost, firststep_cost, laststep_cost, mean_step_cost,
                 prefill_throughput, decode_throughput, total_throughput,
                  card_num_per_qps]
        for info in infos:
            md_file.write(str(format(info, ".4f")) if isinstance(info, float) else str(info))
            md_file.write("|")
        md_file.write('\r\n')
    md_file.close()


gpu_name = "A800"
in_out_lens = [(128, 128), (256, 256)] # in_out_lens 中的数据必须以从短到长的顺序排列，否则可能有问题。
batch_sizes = [1, 2] # batch_sizes 中的数字也必须从小到大排列。


test_all_setting(gpu_name,
                 "llama-7b", 
                 mode=["triton_int8weight", "ppl_fp16_flashdecoding"], # mode 为 【】 为普通 fp16 的格式。
                 log_dir="./", 
                 world_sizes=[1], 
                 in_out_lens=in_out_lens, 
                 batch_sizes=batch_sizes)
