<div align="center">
  <picture>
    <img alt="LightLLM" src="assets/lightllm.drawio.png" width=90%>
  </picture>
</div>

---
<div align="center">

[![docs](https://img.shields.io/badge/docs-latest-blue)](https://lightllm-en.readthedocs.io/en/latest/)
[![Docker](https://github.com/ModelTC/lightllm/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/ModelTC/lightllm/actions/workflows/docker-publish.yml)
[![stars](https://img.shields.io/github/stars/ModelTC/lightllm?style=social)](https://github.com/ModelTC/lightllm)
![visitors](https://komarev.com/ghpvc/?username=lightllm&label=visitors)
[![Discord Banner](https://img.shields.io/discord/1139835312592392214?logo=discord&logoColor=white)](https://discord.gg/WzzfwVSguU)
[![license](https://img.shields.io/github/license/ModelTC/lightllm)](https://github.com/ModelTC/lightllm/blob/main/LICENSE)
</div>

LightLLM is a Python-based LLM (Large Language Model) inference and serving framework, notable for its lightweight design, easy scalability, and high-speed performance. LightLLM harnesses the strengths of numerous well-regarded open-source implementations, including but not limited to FasterTransformer, TGI, vLLM, and FlashAttention.


[English Docs](https://lightllm-en.readthedocs.io/en/latest/) | [中文文档](https://lightllm-cn.readthedocs.io/en/latest/) | [Blogs](https://modeltc.github.io/lightllm-blog/)

## Features

- Tri-process asynchronous collaboration: tokenization, model inference, and detokenization are performed asynchronously, leading to a considerable improvement in GPU utilization.
- Nopad (Unpad): offers support for nopad attention operations across multiple models to efficiently handle requests with large length disparities.
- Dynamic Batch: enables dynamic batch scheduling of requests
- [FlashAttention](https://github.com/Dao-AILab/flash-attention): incorporates FlashAttention to improve speed and reduce GPU memory footprint during inference.
- Tensor Parallelism: utilizes tensor parallelism over multiple GPUs for faster inference.
- [Token Attention](./docs/TokenAttention.md): implements token-wise's KV cache memory management mechanism, allowing for zero memory waste during inference.
- High-performance Router: collaborates with Token Attention to meticulously manage the GPU memory of each token, thereby optimizing system throughput.
- Int8KV Cache: This feature will increase the capacity of tokens to almost twice as much. only llama support.

## Supported Model List

The following table provides a list of supported models along with any special arguments required for their configuration and annotations.

| Model Name                     | Comments                                                                                     |
|--------------------------------|-------------------------------------------------------------------------------------------------------|
| [BLOOM](https://huggingface.co/bigscience/bloom) | None                                                                                                  |
| [LLaMA](https://github.com/facebookresearch/llama) | None                                                                                                  |
| [LLaMA V2](https://huggingface.co/meta-llama) | None                                                                                                  |
| [StarCoder](https://github.com/bigcode-project/starcoder) | None                                                                                                  |
| [Qwen-7b](https://github.com/QwenLM/Qwen-7B) | `--eos_id 151643 --trust_remote_code`                                                                 |
| [ChatGLM2-6b](https://github.com/THUDM/ChatGLM2-6B) | `--trust_remote_code`                                                                                 |
| [InternLM-7b](https://github.com/InternLM/InternLM) | `--trust_remote_code`                                                                                 |
| [InternVL-Chat](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-5) | `--eos_id 32007 --trust_remote_code` (Phi3) or `--eos_id 92542 --trust_remote_code` (InternLM2)       |
| [Qwen-VL](https://huggingface.co/Qwen/Qwen-VL) | None                                                                                                  |
| [Qwen-VL-Chat](https://huggingface.co/Qwen/Qwen-VL-Chat) | None                                                                                                  |
| [Qwen2-VL](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) | `--eos_id 151645 --trust_remote_code`, and run `pip install git+https://github.com/huggingface/transformers` |
| [Llava-7b](https://huggingface.co/liuhaotian/llava-v1.5-7b) | None                                                                                                  |
| [Llava-13b](https://huggingface.co/liuhaotian/llava-v1.5-13b) | None                                                                                                  |
| [Mixtral](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) | None                                                                                                  |
| [Stablelm](https://huggingface.co/stabilityai/stablelm-2-1_6b) | `--trust_remote_code`                                                                                 |
| [MiniCPM](https://huggingface.co/openbmb/MiniCPM-2B-sft-bf16) | None                                                                                                  |
| [Phi-3](https://huggingface.co/collections/microsoft/phi-3-6626e15e9585a200d2d761e3) | Only supports Mini and Small                                                                          |
| [CohereForAI](https://huggingface.co/CohereForAI/c4ai-command-r-plus) | None                                                                                                  |
| [DeepSeek-V2-Lite](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite) | `--data_type bfloat16`                                                                                |
| [DeepSeek-V2](https://huggingface.co/deepseek-ai/DeepSeek-V2) | `--data_type bfloat16`                                                                                |

## Get started

### Installation

Use lightllm with `docker`.

```shell
docker pull ghcr.io/modeltc/lightllm:main
```

To start a container with GPU support and port mapping:

```shell
docker run -it --gpus all -p 8080:8080                  \
        --shm-size 1g -v your_local_path:/data/         \
        ghcr.io/modeltc/lightllm:main /bin/bash
```


    Note: If multiple GPUs are used, `--shm-size` in `docker run` command should be increased.


Alternatively, you can [build the docker image](https://lightllm-en.readthedocs.io/en/latest/getting_started/installation.html#installing-with-docker) or [install from source with pip](https://lightllm-en.readthedocs.io/en/latest/getting_started/installation.html#installing-from-source).

### Quick Start

Lightllm provides LLM inference services with the state-of-the-art throughput performance via efficient request routers and TokenAttention. 

We provide examples to launch the LightLLM service and query the model (via console and python) for both text and multimodal models.

- [Quick Start](https://lightllm-en.readthedocs.io/en/latest/getting_started/quickstart.html)
- [Text Model Service](https://lightllm-en.readthedocs.io/en/latest/models/test.html#llama)
- [Multimodal Model Service](https://lightllm-en.readthedocs.io/en/latest/models/test.html#llava)

    Note: additional parameters for multimodal models (`--enable_multimodal`, `--cache_capacity`) require larger `--shm-size`.
    If the lightllm is run with `--tp > 1`, the visual model will run on the gpu 0.
    Input images format: list for dict like `{'type': 'url'/'base64', 'data': xxx}`
    The special image tag for Qwen-VL is `<img></img>` (`<image>` for Llava), the length of `data["multimodal_params"]["images"]` should be the same as the count of tags, The number can be 0, 1, 2, ...


### Other

Please refer to the [documentation](https://lightllm-en.readthedocs.io/en/latest/) for more information.

## Performance

Lightllm provides high throughput services. The performance comparison between LightLLM and vLLM is shown [here](https://lightllm-en.readthedocs.io/en/latest/dev/performance.html). Up to vllm=0.1.2, we have achieved a 2x larger throughput than vLLM.


### FAQ

Please refer to the [FAQ](https://lightllm-en.readthedocs.io/en/latest/faq.html) for more information.

## Projects using lightllm

We welcome any coopoeration and contribution. If there is a project requires lightllm's support, please contact us via email or create a pull request.


1. <details><summary> <b><a href=https://github.com/LazyAGI/LazyLLM>LazyLLM</a></b>: Easyest and lazyest way for building multi-agent LLMs applications.</summary>

    Once you have installed `lightllm` and `lazyllm`, and then you can use the following code to build your own chatbot:

    ~~~python
    from lazyllm import TrainableModule, deploy, WebModule
    # Model will be download automatically if you have an internet connection
    m = TrainableModule('internlm2-chat-7b').deploy_method(deploy.lightllm)
    WebModule(m).start().wait()
    ~~~

    Documents: https://lazyllm.readthedocs.io/

    </details>


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=ModelTC/lightllm&type=Timeline)](https://star-history.com/#ModelTC/lightllm&Timeline)

## Community

For further information and discussion, [join our discord server](https://discord.gg/WzzfwVSguU).

## License

This repository is released under the [Apache-2.0](LICENSE) license.

## Acknowledgement

We learned a lot from the following projects when developing LightLLM.
- [Faster Transformer](https://github.com/NVIDIA/FasterTransformer)
- [Text Generation Inference](https://github.com/huggingface/text-generation-inference)
- [vLLM](https://github.com/vllm-project/vllm)
- [Flash Attention 1&2](https://github.com/Dao-AILab/flash-attention)
- [OpenAI Triton](https://github.com/openai/triton)
