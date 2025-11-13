ARG CUDA_VERSION=12.6.1
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu22.04
ARG PYTHON_VERSION=3.10
ARG MAMBA_VERSION=24.7.1-0
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
    curl -fsSL -o ~/mambaforge.sh "https://github.com/conda-forge/miniforge/releases/download/${MAMBA_VERSION}/Mambaforge-${MAMBA_VERSION}-Linux-${MAMBA_ARCH}.sh" && \
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
RUN pip install -U pip
RUN pip install -r /lightllm/requirements.txt --no-cache-dir

RUN pip install --no-cache-dir vllm --pre --extra-index-url https://wheels.vllm.ai/nightly 

# TODO: offline compile
# RUN git clone https://github.com/ModelTC/LightKernel.git && cd LightKernel && pip install --no-deps -v .

RUN apt-get update && apt-get install -y libnuma-dev # for sgl_kernel

COPY . /lightllm
RUN pip install -e /lightllm --no-cache-dir
