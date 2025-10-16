<div align="center">
  <picture>
    <img alt="LightLLM" src="assets/logo_new.png" width=90%>
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
- [2025/09] 🔥 LightLLM [v1.1.0](https://www.light-ai.top/lightllm-blog/2025/09/03/lightllm.html) release!
- [2025/08] Pre $^3$ achieves the outstanding paper award of [ACL2025](https://2025.aclweb.org/program/awards/).
- [2025/05] LightLLM paper on constrained decoding accepted by [ACL2025](https://arxiv.org/pdf/2506.03887) (Pre $^3$: Enabling Deterministic Pushdown Automata for Faster Structured LLM Generation). For a more accessible overview of the research with key insights and examples, check out our blog post: [LightLLM Blog](https://www.light-ai.top/lightllm-blog/2025/06/15/pre3.html)
- [2025/04] LightLLM paper on request scheduler published in [ASPLOS’25](https://dl.acm.org/doi/10.1145/3676641.3716011) (Past-Future Scheduler for LLM Serving under SLA Guarantees)
- [2025/02] 🔥 LightLLM v1.0.0 release, achieving the **fastest DeepSeek-R1** serving performance on single H200 machine.

## Get started

- [Install LightLLM](https://lightllm-en.readthedocs.io/en/latest/getting_started/installation.html)
- [Quick Start](https://lightllm-en.readthedocs.io/en/latest/getting_started/quickstart.html)
- [TuTorial](https://lightllm-en.readthedocs.io/en/latest/tutorial/deepseek_deployment.html)


## Performance

Learn more in the release blogs: [v1.0.0 blog](https://www.light-ai.top/lightllm-blog//by%20mtc%20team/2025/02/16/lightllm/).

## FAQ

Please refer to the [FAQ](https://lightllm-en.readthedocs.io/en/latest/faq.html) for more information.

## Projects using LightLLM

We welcome any coopoeration and contribution. If there is a project requires LightLLM's support, please contact us via email or create a pull request.

Projects based on LightLLM or referenced LightLLM components:
- [LazyLLM](https://github.com/LazyAGI/LazyLLM)
- [LoongServe, Peking University](https://github.com/LoongServe/LoongServe)
- [OmniKV, Ant Group](https://github.com/antgroup/OmniKV)
- [vLLM](https://github.com/vllm-project/vllm) (some LightLLM's kernel used)
- [SGLang](https://github.com/sgl-project/sglang) (some LightLLM's kernel used)
- [ParrotServe](https://github.com/microsoft/ParrotServe), Microsoft
- [Aphrodite](https://github.com/aphrodite-engine/aphrodite-engine) (some LightLLM's kernel used)
- [S-LoRA](https://github.com/S-LoRA/S-LoRA)

Also, LightLLM's pure-python design and token-level KC Cache management make it easy to use as the basis for research projects.

Academia works based on or use part of LightLLM:
- [ParrotServe (OSDI’24)](https://www.usenix.org/conference/osdi24/presentation/lin-chaofan)
- [SLoRA (MLSys’24)](https://proceedings.mlsys.org/paper_files/paper/2024/hash/906419cd502575b617cc489a1a696a67-Abstract-Conference.html)
- [LoongServe (SOSP’24)](https://dl.acm.org/doi/abs/10.1145/3694715.3695948)
- [ByteDance’s CXL (Eurosys’24)](https://dl.acm.org/doi/10.1145/3627703.3650061)
- [VTC (OSDI’24)](https://www.usenix.org/conference/osdi24/presentation/sheng)
- [OmniKV (ICLR’25)](https://openreview.net/forum?id=ulCAPXYXfa)
- [CaraServe](https://arxiv.org/abs/2401.11240), [LoRATEE](https://ieeexplore.ieee.org/abstract/document/10890445), [FastSwitch](https://arxiv.org/abs/2411.18424) ...


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


## Citation

We have published a number of papers around components or features of LightLLM, if you use LightLLM in your work, please consider citing the relevant paper.

**constrained decoding**: accepted by [ACL2025](https://arxiv.org/pdf/2506.03887) and achieved the outstanding paper award.
```bibtex
@inproceedings{
anonymous2025pre,
title={Pre\${\textasciicircum}3\$: Enabling Deterministic Pushdown Automata for Faster Structured {LLM} Generation},
author={Anonymous},
booktitle={Submitted to ACL Rolling Review - February 2025},
year={2025},
url={https://openreview.net/forum?id=g1aBeiyZEi},
note={under review}
}
```

**Request scheduler**: accepted by [ASPLOS’25](https://dl.acm.org/doi/10.1145/3676641.3716011):
```bibtex
@inproceedings{gong2025past,
  title={Past-Future Scheduler for LLM Serving under SLA Guarantees},
  author={Gong, Ruihao and Bai, Shihao and Wu, Siyu and Fan, Yunqian and Wang, Zaijun and Li, Xiuhong and Yang, Hailong and Liu, Xianglong},
  booktitle={Proceedings of the 30th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2},
  pages={798--813},
  year={2025}
}
```
