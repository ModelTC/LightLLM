.. _installation:

Installation
============

Lightllm is a Python-based inference framework, with operators implemented in Triton.

Requirements
------------

* Operating System: Linux
* Python: 3.9
* GPU: Compute Capability 7.0 or higher (e.g., V100, T4, RTX20xx, A100, L4, H100, etc.).


.. _build_from_docker:

Installing with Docker
-----------------------
The easiest way to install Lightllm is by using the official image. You can directly pull and run the official image:

.. code-block:: console

    $ # Pull the official image
    $ docker pull ghcr.io/modeltc/lightllm:main
    $
    $ # Run the image
    $ docker run -it --gpus all -p 8080:8080            \
    $   --shm-size 1g -v your_local_path:/data/         \
    $   ghcr.io/modeltc/lightllm:main /bin/bash

You can also manually build and run the image from the source:


.. code-block:: console

    $ # Manually build the image
    $ docker build -t <image_name> .
    $
    $ # Run the image
    $ docker run -it --gpus all -p 8080:8080            \
    $   --shm-size 1g -v your_local_path:/data/         \
    $   <image_name> /bin/bash

Alternatively, you can use a script to automatically build and run the image:


.. code-block:: console

    $ # View script parameters
    $ python tools/quick_launch_docker.py --help

.. note::
    If you are using multiple GPUs, you may need to increase the --shm-size parameter setting above.

.. _build_from_source:

Installing from Source
-----------------------

You can also install Lightllm from source:

.. code-block:: console

    $ # (Recommended) Create a new conda environment
    $ conda create -n lightllm python=3.9 -y
    $ conda activate lightllm
    $
    $ # Download the latest source code for Lightllm
    $ git clone https://github.com/ModelTC/lightllm.git
    $ cd lightllm
    $
    $ # Install Lightllm's dependencies
    $ pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
    $
    $ # Install Lightllm
    $ python setup.py install

NOTE: If you are using torch with cuda 11.x instead, run `pip install nvidia-nccl-cu12==2.20.5` to support torch cuda graph.

.. note::

    The Lightllm code has been tested on various GPUs, including V100, A100, A800, 4090, and H800.
    If you are using A100, A800, or similar GPUs, it is recommended to install triton==3.0.0:

    .. code-block:: console

        $ pip install triton==3.0.0 --no-deps

    If you are using H800, V100, or similar GPUs, it is recommended to install triton-nightly:

    .. code-block:: console

        $ pip install -U --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly --no-deps

    For more details, refer to: `issue <https://github.com/triton-lang/triton/issues/3619>`_ and `fix PR <https://github.com/triton-lang/triton/pull/3638>`_
