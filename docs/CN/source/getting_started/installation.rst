.. _installation:

安装指南
============

Lightllm 是一个纯python开发的推理框架，其中的算子使用triton编写。

环境要求
------------

* 操作系统: Linux
* Python: 3.9
* GPU: 计算能力 7.0 以上 (e.g., V100, T4, RTX20xx, A100, L4, H100, 等等.)

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
    $ # 手动构建镜像, docker 目录下有不同功能场景的镜像构建文件，按需构建。
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
    如果你使用多卡，你也许需要提高上面的 –shm_size 的参数设置。如果需要跑DeepSeek模型的EP模式，请使用镜像
    ghcr.io/modeltc/lightllm:main-deepep。

.. _build_from_source:

使用源码安装
----------------

你也可以使用源码安装Lightllm：

.. code-block:: console

    $ # (推荐) 创建一个新的 conda 环境
    $ conda create -n lightllm python=3.9 -y
    $ conda activate lightllm
    $
    $ # 下载lightllm的最新源码
    $ git clone https://github.com/ModelTC/lightllm.git
    $ cd lightllm
    $
    $ # 安装lightllm的依赖 (cuda 12.4)
    $ pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
    $
    $ # 安装lightllm
    $ python setup.py install

.. note::

    Lightllm 的代码在多种GPU上都进行了测试，包括 V100, A100, A800, 4090, 和 H800。
    如果你使用 A100 、A800 等显卡，那么推荐你安装 triton==3.0.0 ：

    .. code-block:: console

        $ pip install triton==3.0.0 --no-deps

    如果你使用 H800、V100 等显卡，那么推荐你安装 triton-nightly：

    .. code-block:: console

        $ pip install -U --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly --no-deps
    
    具体原因可以参考：`issue <https://github.com/triton-lang/triton/issues/3619>`_ 和 `fix PR <https://github.com/triton-lang/triton/pull/3638>`_

