import subprocess
import re


def kill_gpu_processes():
    try:
        output = subprocess.check_output(["nvidia-smi", "-q", "-x"])
        output = output.decode("utf-8")

        # 使用正则表达式提取进程信息
        process_info = re.findall(r"<process_info>(.*?)</process_info>", output, re.DOTALL)

        if process_info:
            print("找到以下占用显卡的进程：")
            for info in process_info:
                pid = re.search(r"<pid>(.*?)</pid>", info).group(1)
                process_name = re.search(r"<process_name>(.*?)</process_name>", info).group(1)
                print("进程ID:", pid)
                print("进程名字:", process_name)

            for info in process_info:
                pid = re.search(r"<pid>(.*?)</pid>", info).group(1)
                subprocess.call(["sudo", "kill", "-9", pid])
                print("进程ID", pid, "被终止")
        else:
            print("没有找到占用显卡的进程")

    except subprocess.CalledProcessError:
        print("无法执行nvidia-smi命令")


if __name__ == "__main__":
    kill_gpu_processes()
