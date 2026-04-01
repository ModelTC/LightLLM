import re
import sys
from collections import defaultdict

def detailed_stats(file_path: str):
    # 1. 定义需要匹配的指标及其正则表达式
    # 这里添加了你提到的两个字段，你可以继续在这里添加新的字段
    patterns = {
        'mtp_avg_token_per_step': r'mtp_avg_token_per_step:\s*(\d+\.?\d*)',
        'mtp_avg_verify_tokens_per_step': r'mtp_avg_verify_tokens_per_step:\s*(\d+\.?\d*)'
    }
    
    # 2. 存储提取到的数值，结构为 {指标名: [数值列表]}
    metric_values = defaultdict(list)
    
    # (可选) 如果你想按 Step 记录并覆盖旧值，可以启用下面这行 (类似你原本的逻辑)
    # step_values = defaultdict(float) 
    # 注：如果要按Step记录为列表，建议用: step_values = defaultdict(lambda: defaultdict(list))

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 遍历所有定义的指标进行匹配
            for metric_name, pattern in patterns.items():
                match = re.search(pattern, line)
                if match:
                    value = float(match.group(1))
                    metric_values[metric_name].append(value)
                    
                    # --- (可选) 按Step记录逻辑 ---
                    # step_match = re.search(r'Step\s+(\d+)', line)
                    # if step_match:
                    #     step = int(step_match.group(1))
                    #     step_values[step] = value # 这里会保留最后一个Step的值

    if not metric_values:
        print("未找到任何匹配的数据")
        return

    # 3. 定义打印统计结果的辅助函数
    def print_stats(name, values):
        if not values:
            return
            
        avg = sum(values) / len(values)
        
        print(f"\n{'='*50}")
        print(f"指标: {name}")
        print(f"{'='*50}")
        print(f"总匹配数: {len(values)}")
        print(f"均值 (Avg): {avg:.4f}")
        print(f"最小值 (Min): {min(values):.4f}")
        print(f"最大值 (Max): {max(values):.4f}")
        
        # 分段统计 (前中后 1/3)
        if len(values) >= 10:
            n = len(values) // 3
            print(f"\n前 1/3 平均: {sum(values[:n])/n:.4f}")
            print(f"中 1/3 平均: {sum(values[n:2*n])/n:.4f}")
            print(f"后 1/3 平均: {sum(values[2*n:])/n:.4f}")

    # 4. 输出标题并遍历所有找到的指标进行打印
    print(f"\n{'#'*60}")
    print(f"文件: {file_path} - 统计结果")
    print(f"{'#'*60}")

    for name, values in metric_values.items():
        print_stats(name, values)
        
    print(f"\n{'#'*60}")

if __name__ == "__main__":
    # 如果没有传入文件路径，默认找 output.log
    file_path = sys.argv[1] if len(sys.argv) > 1 else "output.log"
    detailed_stats(file_path)