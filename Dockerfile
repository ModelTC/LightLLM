FROM nvcr.io/nvidia/tritonserver:24.04-py3-min as base
ARG PYTORCH_VERSION=2.6.0
ARG PYTHON_VERSION=3.9
ARG CUDA_VERSION=12.4
ARG MAMBA_VERSION=23.1.0-1
ARG TARGETPLATFORM

ENV PATH=/opt/conda/bin:$PATH \
    CONDA_PREFIX=/opt/conda

RUN chmod 777 -R /tmp && apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl-dev \
    curl \
    g++ \
    make \
    git && \
    rm -rf /var/lib/apt/lists/*

RUN case ${TARGETPLATFORM} in \
    "linux/arm64")  MAMBA_ARCH=aarch64  ;; \
    *)              MAMBA_ARCH=x86_64   ;; \
    esac && \
    curl -fsSL -o ~/mambaforge.sh -v "https://github.com/conda-forge/miniforge/releases/download/${MAMBA_VERSION}/Mambaforge-${MAMBA_VERSION}-Linux-${MAMBA_ARCH}.sh" && \
    bash ~/mambaforge.sh -b -p /opt/conda && \
    rm ~/mambaforge.sh

RUN case ${TARGETPLATFORM} in \
    "linux/arm64")  exit 1 ;; \
    *)              /opt/conda/bin/conda update -y conda &&  \
    /opt/conda/bin/conda install -y "python=${PYTHON_VERSION}" ;; \
    esac && \
    /opt/conda/bin/conda clean -ya


WORKDIR /root

COPY ./requirements.txt /lightllm/requirements.txt
RUN pip install -r /lightllm/requirements.txt --no-cache-dir --ignore-installed --extra-index-url https://download.pytorch.org/whl/cu124

RUN pip install --no-cache-dir https://github.com/ModelTC/flash-attn-3-build/releases/download/v2.7.4.post1/flash_attn-3.0.0b1-cp39-cp39-linux_x86_64.whl

RUN pip install --no-cache-dir nvidia-nccl-cu12==2.25.1  # for allreduce hang issues in multinode H100

COPY . /lightllm
RUN pip install -e /lightllm --no-cache-dir
