FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04 as base
ARG PYTORCH_VERSION=2.0.0
ARG PYTHON_VERSION=3.9
ARG CUDA_VERSION=11.8
ARG MAMBA_VERSION=23.1.0-1
ARG CUDA_CHANNEL=nvidia
ARG INSTALL_CHANNEL=pytorch
ARG TARGETPLATFORM

ENV PATH=/opt/conda/bin:$PATH \
    CONDA_PREFIX=/opt/conda

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
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
    /opt/conda/bin/conda install -c "${INSTALL_CHANNEL}" -c "${CUDA_CHANNEL}" -y "python=${PYTHON_VERSION}" pytorch==$PYTORCH_VERSION "pytorch-cuda=$(echo $CUDA_VERSION | cut -d'.' -f 1-2)" -c anaconda -c conda-forge ;; \
    esac && \
    /opt/conda/bin/conda clean -ya

# workaround
RUN mkdir ~/cuda-nvcc && cd ~/cuda-nvcc && \
    curl -fsSL -o package.tar.bz2 https://conda.anaconda.org/nvidia/label/cuda-12.1.1/linux-64/cuda-nvcc-12.1.105-0.tar.bz2 && \
    tar xf package.tar.bz2 &&\
    mkdir -p /usr/local/cuda/bin && \
    mkdir -p /usr/local/cuda/include && \
    cp bin/ptxas /usr/local/cuda/bin/ptxas && \
    curl -fsSL -o /usr/local/cuda/include/cuda.h https://raw.githubusercontent.com/openai/triton/d1ce4c495052a1ac06302213cae8eb5532a67259/python/triton/third_party/cuda/include/cuda.h \
    && rm ~/cuda-nvcc -rf

WORKDIR /root

COPY ./requirements.txt /lightllm/requirements.txt
RUN pip install -r /lightllm/requirements.txt --no-cache-dir

COPY . /lightllm
RUN pip install -e /lightllm --no-cache-dir