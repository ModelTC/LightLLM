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

## News
- [2025/02] 🔥 LightLLM v1.0.0 release, achieving the **fastest DeepSeek-R1** serving performance on single H200 machine.

## Get started

- [Install LightLLM](https://lightllm-en.readthedocs.io/en/latest/getting_started/installation.html)
- [Quick Start](https://lightllm-en.readthedocs.io/en/latest/getting_started/quickstart.html)
- [LLM Service](https://lightllm-en.readthedocs.io/en/latest/models/test.html#llama)
- [VLM Service](https://lightllm-en.readthedocs.io/en/latest/models/test.html#llava)


## Performance

Learn more in the release blogs: [v1.0.0 blog](https://www.light-ai.top/lightllm-blog//by%20mtc%20team/2025/02/16/lightllm/).

## FAQ

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

## Community

For further information and discussion, [join our discord server](https://discord.gg/WzzfwVSguU). Welcome to be a member and look forward to your contribution!

## License

This repository is released under the [Apache-2.0](LICENSE) license.

## Acknowledgement

We learned a lot from the following projects when developing LightLLM.
- [Faster Transformer](https://github.com/NVIDIA/FasterTransformer)
- [Text Generation Inference](https://github.com/huggingface/text-generation-inference)
- [vLLM](https://github.com/vllm-project/vllm)
- [SGLang](https://github.com/sgl-project/sglang)
- [flashinfer](https://github.com/flashinfer-ai/flashinfer/tree/main)
- [Flash Attention 1&2](https://github.com/Dao-AILab/flash-attention)
- [OpenAI Triton](https://github.com/openai/triton)
