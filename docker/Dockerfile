ARG CUDA_VERSION=13.0.0
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu22.04

ARG PYTHON_VERSION=3.10
ARG MAMBA_VERSION=24.7.1-0
ARG VLLM_VERSION=0.21.0
ARG NIXL_REF=v1.1.0
ARG FLASH_MLA_REF=47c35a7
ARG DEEPGEMM_REF=891d57b4db1071624b5c8fa0d1e51cb317fa709f
ARG TARGETPLATFORM
ARG ENABLE_DEEPEP=1
ARG ENABLE_NIXL=1
ARG ENABLE_CACHE=1
ARG ENABLE_SM100=0

ENV PATH=/opt/conda/bin:$PATH \
    CONDA_PREFIX=/opt/conda

RUN chmod 777 -R /tmp && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      ca-certificates \
      libssl-dev \
      curl \
      g++ \
      make \
      git \
      wget && \
    rm -rf /var/lib/apt/lists/*

RUN case ${TARGETPLATFORM} in \
    "linux/arm64")  MAMBA_ARCH=aarch64  ;; \
    *)              MAMBA_ARCH=x86_64   ;; \
    esac && \
    wget -O ~/mambaforge.sh "https://github.com/conda-forge/miniforge/releases/download/${MAMBA_VERSION}/Mambaforge-${MAMBA_VERSION}-Linux-${MAMBA_ARCH}.sh" && \
    bash ~/mambaforge.sh -b -p /opt/conda && \
    rm ~/mambaforge.sh

RUN case ${TARGETPLATFORM} in \
    "linux/arm64")  exit 1 ;; \
    *)              /opt/conda/bin/conda update -y conda && \
                    /opt/conda/bin/conda install -y "python=${PYTHON_VERSION}" ;; \
    esac && \
    /opt/conda/bin/conda clean -ya

WORKDIR /root

COPY ./requirements.txt /lightllm/requirements.txt
RUN pip install -U pip
RUN pip install --no-cache-dir \
    -i https://pypi.org/simple \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    vllm==${VLLM_VERSION}
RUN pip install -r /lightllm/requirements.txt --no-cache-dir \
    -i https://pypi.org/simple \
    --extra-index-url https://download.pytorch.org/whl/cu130
RUN export CPATH=/usr/local/cuda/targets/x86_64-linux/include/cccl:/usr/local/cuda/targets/x86_64-linux/include${CPATH:+:${CPATH}} && \
    git clone https://github.com/deepseek-ai/FlashMLA.git /root/FlashMLA && \
    cd /root/FlashMLA && \
    git checkout ${FLASH_MLA_REF} && \
    git submodule update --init --recursive && \
    FLASH_MLA_DISABLE_SM100="$(if [ "${ENABLE_SM100}" = "1" ]; then echo 0; else echo 1; fi)" \
    pip install --no-cache-dir .

RUN apt-get update && apt-get install -y libnuma-dev && rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME=/usr/local/cuda \
    GDRCOPY_HOME=/usr/src/gdrdrv-2.4.4/

RUN if [ "${ENABLE_CACHE}" = "1" ]; then \
      apt-get update && apt-get install -y libboost-dev && rm -rf /var/lib/apt/lists/*; \
      LIGHTMEM_REF=5900baf92d85ef4dbda6124093506b0af906011a; \
      pip install --no-deps -v "git+https://github.com/ModelTC/LightMem.git@${LIGHTMEM_REF}#egg=light_mem"; \
    fi

RUN if [ "${ENABLE_NIXL}" = "1" ] || [ "${ENABLE_DEEPEP}" = "1" ]; then \
      apt-get update && apt-get install -y wget devscripts debhelper dh-make build-essential dkms && \
      apt-get install -y ibverbs-providers infiniband-diags perftest rdma-core libibverbs-dev librdmacm-dev && \
      rm -rf /var/lib/apt/lists/*; \
      mkdir -p /tmp/gdrcopy && cd /tmp \
      && git clone https://github.com/NVIDIA/gdrcopy.git -b v2.4.4 \
      && cd gdrcopy/packages \
      && CUDA=/usr/local/cuda ./build-deb-packages.sh \
      && dpkg -i gdrdrv-dkms_*.deb libgdrapi_*.deb gdrcopy-tests_*.deb gdrcopy_*.deb \
      && cd / && rm -rf /tmp/gdrcopy; \
    fi

RUN if [ "${ENABLE_DEEPEP}" = "1" ]; then \
      set -e; \
      ln -sf /usr/lib/x86_64-linux-gnu/libmlx5.so.1 /usr/lib/x86_64-linux-gnu/libmlx5.so; \
      python -m pip install --upgrade --no-deps \
        "nvidia-nccl-cu13==2.30.4" \
        "nvidia-nvshmem-cu13==3.6.5"; \
      cd /root && git clone https://github.com/deepseek-ai/DeepEP.git && cd DeepEP && git checkout b306af06afd412c88e51e71802951606e40b7358; \
      ln -sf /opt/conda/lib/python${PYTHON_VERSION}/site-packages/nvidia/nvshmem/lib/libnvshmem_host.so.3 /opt/conda/lib/python${PYTHON_VERSION}/site-packages/nvidia/nvshmem/lib/libnvshmem_host.so; \
      ln -sf /opt/conda/lib/python${PYTHON_VERSION}/site-packages/nvidia/nccl/lib/libnccl.so.2 /opt/conda/lib/python${PYTHON_VERSION}/site-packages/nvidia/nccl/lib/libnccl.so; \
      pip install --no-build-isolation .; \
    fi

RUN cd /root && git clone https://github.com/deepseek-ai/DeepGEMM.git && \
    cd DeepGEMM && git checkout ${DEEPGEMM_REF} && \
    git submodule update --init --recursive && \
    pip install --no-build-isolation .

RUN if [ "${ENABLE_NIXL}" = "1" ]; then \
      apt-get update && apt-get install -y cmake automake autotools-dev libtool libz-dev && \
      DEBIAN_FRONTEND=noninteractive apt-get -y install --reinstall libibverbs-dev rdma-core ibverbs-utils libibumad-dev; \
      rm -rf /usr/lib/ucx && rm -rf /opt/hpcx/ucx && \
      cd /usr/local/src && \
      git clone https://github.com/openucx/ucx.git && \
      cd ucx && \
      git checkout v1.19.x && \
      ./autogen.sh && ./configure \
      --enable-shared \
      --disable-static \
      --disable-doxygen-doc \
      --enable-optimizations \
      --enable-cma \
      --enable-devel-headers \
      --with-cuda=/usr/local/cuda \
      --with-verbs=yes \
      --with-dm \
      --with-gdrcopy=/usr/local \
      --with-efa \
      --enable-mt && \
      make -j && \
      make -j install-strip && \
      ldconfig; \
      apt-get update && apt-get install -y pkg-config tmux net-tools && \
      cd /usr/local/src; \
      pip install --upgrade meson pybind11 patchelf; \
      git clone https://github.com/ai-dynamo/nixl.git -b ${NIXL_REF} && \
      cd nixl && \
      rm -rf build && \
      mkdir build && \
      meson setup build/ --prefix=/usr/local/nixl --buildtype=release && \
      cd build && \
      ninja && \
      ninja install && \
      cd .. && pip install . --no-deps; \
    fi

COPY . /lightllm
RUN pip install -e /lightllm --no-cache-dir
