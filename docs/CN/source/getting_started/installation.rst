.. _installation:

安装指南
============

Lightllm 是一个纯python开发的推理框架，其中的算子使用triton编写。

环境要求
------------

* 操作系统: Linux
* Python: 3.10–3.13（下方原生安装推荐使用 3.12）
* 默认原生运行环境需要 NVIDIA GPU 和兼容 CUDA 13.0 的驱动。
  具体硬件支持还取决于模型及所选算子。

使用 uv 原生安装
-----------------

在本版本源码目录中执行：

.. code-block:: console

    $ uv venv --python 3.12
    $ source .venv/bin/activate
    $ uv pip install .
    $ lightllm serve /models/my-model

安装包已经声明 PyTorch、Transformers、HTTP 服务、XGrammar 及 SGLang kernel 等运行依赖，
无需先安装 ``requirements.txt``。开发时可使用 ``uv pip install -e .``。
默认的 PyTorch 2.14.1 Linux wheel 使用 CUDA 13.0；安装 Python 包不会安装 GPU 驱动。
首次启动建议选择 LightLLM 支持、单卡显存可容纳的非量化模型。

``lightllm serve --help`` 显示常用参数，``--help-all`` 显示完整参数。
原有服务参数及默认值均保留，包括 ``--model_dir``，原 Python 模块启动入口也仍然可用。

驱动感知安装器
~~~~~~~~~~~~~~~~

交付 wheel、依赖锁文件和 ``installer-manifest.json`` 后，用户不需要手工判断
CUDA 版本。安装器先执行 ``nvidia-smi``，再按驱动版本选择 manifest 中最高兼容的
profile，并使用 ``uv`` 创建环境、安装对应依赖和 LightLLM wheel：

.. code-block:: console

    $ python lightllm-install.py
    $ lightllm serve /models/my-model

可以使用 ``--dry-run`` 只查看选择结果和安装命令，使用 ``--profile cuda128``
显式选择 profile。安装器不会升级 NVIDIA 驱动；如果 manifest 中没有当前驱动可用
的 wheel，会直接报告可用 profile，并建议改用对应 Docker 镜像或补充 wheel profile。
profile 的 ``wheel`` 和 ``requirements`` 可以是本地文件或 HTTPS URL，发布新的 CUDA
组合不需要修改安装器代码。

交付安装包：

.. code-block:: console

    $ uv build
    $ uv pip install dist/lightllm-1.2.0-py3-none-any.whl

``uv build`` 还会从源码包构建 wheel，以验证源码包的完整性。
包含本次修改的版本发布后，用户即可使用 ``uv pip install lightllm==<发布版本>``。
已有的旧版 PyPI 包不一定包含这个命令入口和依赖声明。

稳定版与 nightly
~~~~~~~~~~~~~~~~

两个通道都安装 ``lightllm``，并使用相同的 ``lightllm serve`` 入口。
稳定版由与 ``VERSION`` 一致的 ``vX.Y.Z`` 标签构建；主分支的每日构建和手动
``nightly`` 构建使用下一补丁版本的开发版本号，例如 ``1.2.1.dev1234``，
其中 ``1234`` 是 CI 构建编号。稳定版读取 ``requirements/runtime.txt``；nightly
读取 ``requirements/nightly.txt`` 中完整冻结的运行环境。

nightly 按 ``requirements/nightly.in`` 的顺序优先选择最新可安装版本，包含上游
nightly 和 RC：Torch、Triton、CUDA 优先，其次是模型加载和推理算子，
然后是服务框架、协议适配与辅助库。每选定一个包就固定其版本；后面的包发生冲突时，
只选择与前面兼容的最高版本，无解则构建失败。可选且未安装的扩展不限制主环境。
因此 SGLang kernel 可以为新 Torch 降级，LiteLLM 可以为新模型加载依赖降级。
版本声明兼容不代表二进制兼容。构建还必须检查原生算子导入；若旧 wheel 与新 Torch
存在 ABI 冲突，则该候选构建失败，需要提供按新 Torch 重编译的算子 wheel，
不能通过忽略依赖或继续降低高优先级核心库来冒充成功。
维护者可通过 ``UV_FIND_LINKS`` 提供项目重编译的内核 wheel，再执行依赖更新。
同一构建保持最初选定的核心快照；原生内核重建后，可用 ``UV_CONSTRAINT`` 指向
这些核心包的固定版本或 wheel URL，再继续解析。构建期间新上传的版本留给下一轮。
这类 nightly 必须同时交付内核 wheel，用户解压完整产物后执行
``uv pip install dist/*.whl``；单独交付 LightLLM wheel 前，必须先让其原生依赖
在对应的 nightly 索引中可用。

维护者执行 ``python tools/update_nightly.py`` 更新完整依赖清单，CI 的 nightly
构建会自动执行这一步（需要 uv 和 packaging）。当前 nightly 目标为
Linux x86_64、CPython 3.12，使用 PyTorch 官方 CUDA 13.4 nightly 索引和 PyPI。
Torch、Triton、torchvision 的 wheel URL 和 SHA256 写入包依赖，安装时无需手动配置额外索引。
新组合通过 GPU 模型和接口回归后才能作为已验证版本交付。

``Native package`` 工作流目前生成 wheel、源码包和完整安装版本记录，
保存为 GitHub Actions artifacts，尚未自动上传 PyPI 或 nightly 索引。
下载后可直接 ``uv pip install <wheel路径>``；包上传到索引后，可按实际版本安装：

.. code-block:: console

    $ uv pip install lightllm==1.2.0
    $ uv pip install lightllm==1.2.1.dev1234

更新时允许依赖降级；用户安装 wheel 时按包内冻结版本安装，不再现场选择新的依赖组合。
CPU 打包检查不能替代 GPU 模型回归。

基础安装包含 PyTorch/Triton 算子及 Qwen3.5 线性注意力所需的 SGLang kernel。
FlashInfer、独立 FlashMLA、DeepGEMM、DeepEP、
NIXL、LightMem 及 vLLM 量化算子仍需要额外安装兼容的构建产物；
AWQ 和部分 FP8 模型也依赖这些额外算子。请参照对应模型或部署指南，
现有 Docker 构建包含更多此类扩展。根目录 ``requirements.txt`` 仍用于旧的镜像/开发环境，
原生安装以安装包内的依赖声明为准。

MUSA 等厂商运行环境需要使用对应的 PyTorch 和 Triton 构建。
按照平台指南准备好依赖后，使用 ``uv pip install --no-deps .`` 保留厂商运行环境。
默认原生依赖集面向 NVIDIA CUDA，不会自动解决所有硬件平台的依赖差异。

原生依赖版本
~~~~~~~~~~~~

``requirements/runtime.txt`` 是原生安装的依赖来源。核心版本固定为：

.. list-table::
   :header-rows: 1

   * - 组件
     - 版本
   * - PyTorch / torchvision
     - 2.14.1 / 0.29.1
   * - Triton / SGLang kernel
     - 3.8.0 / 0.4.10
   * - CuPy CUDA 13 / Transformers
     - 14.2.0 / 5.20.2
   * - XGrammar / TVM FFI
     - 0.2.8 / 0.1.12
   * - LiteLLM
     - 1.106.2（小于 1.107）

PyTorch、torchvision、Triton 和 SGLang kernel 需要成组升级：PyTorch 约束 Triton，
SGLang kernel 0.4.10 要求 PyTorch 2.14.1。更新依赖后应在新环境中检查安装、
实际模型启动、文本和图像请求，以及结构化输出；依赖解析成功不代表 GPU 算子兼容。
调优配置按 Triton 版本和 GPU 区分；升级后需要重新测量性能，必要时重新调优。
TVM FFI 0.1.12 满足可选 TileLang 0.1.15 的 ``<0.1.13`` 约束；
LiteLLM 1.106.2 要求 ``filelock<4``、``huggingface-hub<2``。
其余依赖使用兼容范围，由 uv 解析；Python 3.12 可使用 Librosa 1.0、Pillow 13、
WebSockets 17，较旧 Python 自动选择支持的版本。
交付时可用 ``uv pip freeze`` 记录完整环境版本。根目录的旧镜像依赖独立维护，
不要在这个原生环境中叠加安装 ``requirements.txt``。

.. _build_from_docker:

使用docker安装
----------------
安装lightllm最简单的方法是使用官方镜像，你可以直接拉取官方镜像并运行：

.. code-block:: console

    $ # 拉取官方镜像
    $ docker pull ghcr.io/modeltc/lightllm:main
    $
    $ # 运行服务, 注意现在的lightllm服务非常的依赖共享内存部分，在启动
    $ # 前请确保你的docker设置中已经分配了足够的共享内存，否则可能导致
    $ # 服务无法正常启动。
    $ # 1.如果是纯文本服务，建议分配2GB以上的共享内存, 如果你的内存充足，建议分配16GB以上的共享内存.
    $ # 2.如果是多模态服务，建议分配16GB以上的共享内存，具体可以根据实际情况进行调整.
    $ # 如果你没有足够的共享内存，可以尝试在启动服务的时候调低 --running_max_req_size 参数，这会降低
    $ # 服务的并发请求数量，但可以减少共享内存的占用。如果是多模态服务，也可以通过降低 --cache_capacity
    $ # 参数来减少共享内存的占用。
    $ docker run -it --gpus all -p 8080:8080            \
    $   --shm-size 2g -v your_local_path:/data/         \
    $   ghcr.io/modeltc/lightllm:main /bin/bash

你也可以使用源码手动构建镜像并运行,建议手动构建镜像,因为更新比较频繁：

.. code-block:: console

    $ # 进入代码仓库的根目录
    $ cd /lightllm
    $ # 手动构建镜像。
    $ docker build -t <image_name> -f ./docker/Dockerfile .
    $
    $ # 运行
    $ docker run -it --gpus all -p 8080:8080            \
    $   --shm-size 2g -v your_local_path:/data/         \
    $   <image_name> /bin/bash

或者你也可以直接使用脚本一键启动镜像并且运行：

.. code-block:: console

    $ # 查看脚本参数
    $ python tools/quick_launch_docker.py --help

.. note::
    如果你使用多卡，你也许需要提高上面的 –shm_size 的参数设置。

.. _build_from_source:

使用源码安装
----------------

你也可以使用源码安装Lightllm：

.. code-block:: console

    $ # (推荐) 创建一个新的 conda 环境
    $ conda create -n lightllm python=3.10 -y
    $ conda activate lightllm
    $
    $ # 下载lightllm的最新源码
    $ git clone https://github.com/ModelTC/lightllm.git
    $ cd lightllm
    $
    $ # 安装lightllm的依赖 (CUDA 13.0)
    $ pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130
    $
    $ # 安装lightllm的依赖 (摩尔线程 GPU)
    $ ./generate_requirements_musa.sh
    $ pip install -r requirements-musa.txt
    $
    $ # 安装lightllm
    $ pip install --no-deps .
