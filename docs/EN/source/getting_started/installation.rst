.. _installation:

Installation Guide
==================

Lightllm is a pure Python-based inference framework with operators written in Triton.

Environment Requirements
------------------------

* Operating System: Linux
* Python: 3.9
* GPU: Compute Capability 7.0 or higher (e.g., V100, T4, RTX20xx, A100, L4, H100, etc.)

.. _build_from_docker:

Installation via Docker
-----------------------
The easiest way to install Lightllm is using the official image. You can directly pull the official image and run it:

.. code-block:: console

    $ # Pull the official image
    $ docker pull ghcr.io/modeltc/lightllm:main
    $
    $ # Run，The current LightLLM service relies heavily on shared memory.
    $ # Before starting, please make sure that you have allocated enough shared memory 
    $ # in your Docker settings; otherwise, the service may fail to start properly.
    $ #
    $ # 1. For text-only services, it is recommended to allocate more than 2GB of shared memory. 
    $ # If your system has sufficient RAM, allocating 16GB or more is recommended.
    $ # 2.For multimodal services, it is recommended to allocate 16GB or more of shared memory. 
    $ # You can adjust this value according to your specific requirements.
    $ #
    $ # If you do not have enough shared memory available, you can try lowering 
    $ # the --running_max_req_size parameter when starting the service. 
    $ # This will reduce the number of concurrent requests, but also decrease shared memory usage.
    $ docker run -it --gpus all -p 8080:8080            \
    $   --shm-size 2g -v your_local_path:/data/         \
    $   ghcr.io/modeltc/lightllm:main /bin/bash

You can also manually build the image from source and run it:

.. code-block:: console
    
    $ # move into lightllm root dir
    $ cd /lightllm
    $ # Manually build the image
    $ docker build -t <image_name> -f ./docker/Dockerfile .
    $
    $ # Run, 
    $ docker run -it --gpus all -p 8080:8080            \
    $   --shm-size 2g -v your_local_path:/data/         \
    $   <image_name> /bin/bash

Or you can directly use the script to launch the image and run it with one click:

.. code-block:: console
    
    $ # View script parameters
    $ python tools/quick_launch_docker.py --help

.. note::
    If you use multiple GPUs, you may need to increase the --shm-size parameter setting above. If you need to run DeepSeek models in EP mode, please use the image
    ghcr.io/modeltc/lightllm:main-deepep.

.. _build_from_source:

Installation from Source
------------------------

You can also install Lightllm from source:

.. code-block:: console

    $ # (Recommended) Create a new conda environment
    $ conda create -n lightllm python=3.9 -y
    $ conda activate lightllm
    $
    $ # Download the latest Lightllm source code
    $ git clone https://github.com/ModelTC/lightllm.git
    $ cd lightllm
    $
    $ # Install Lightllm dependencies (cuda 12.4)
    $ pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
    $
    $ # Install Lightllm
    $ python setup.py install

.. note::

    Lightllm code has been tested on various GPUs including V100, A100, A800, 4090, and H800.
    If you use A100, A800 and other graphics cards, it is recommended to install triton==3.0.0:

    .. code-block:: console

        $ pip install triton==3.0.0 --no-deps

    If you use H800, V100 and other graphics cards, it is recommended to install triton-nightly:

    .. code-block:: console

        $ pip install -U --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly --no-deps
    
    For specific reasons, please refer to: `issue <https://github.com/triton-lang/triton/issues/3619>`_ and `fix PR <https://github.com/triton-lang/triton/pull/3638>`_