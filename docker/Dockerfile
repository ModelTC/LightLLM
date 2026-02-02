ARG CUDA_VERSION=12.8.0
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu22.04

ARG PYTHON_VERSION=3.10
ARG MAMBA_VERSION=24.7.1-0
ARG VLLM_VERSION=0.11.0
ARG TARGETPLATFORM
ARG ENABLE_DEEPEP=1
ARG ENABLE_NIXL=1
ARG ENABLE_CACHE=1

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
    *)              /opt/conda/bin/conda update -y conda && \
                    /opt/conda/bin/conda install -y "python=${PYTHON_VERSION}" ;; \

    esac && \
    /opt/conda/bin/conda clean -ya

WORKDIR /root

COPY ./requirements.txt /lightllm/requirements.txt
RUN pip install -U pip
RUN pip install -r /lightllm/requirements.txt --no-cache-dir
RUN pip install --no-cache-dir vllm==${VLLM_VERSION}
RUN git clone https://github.com/ModelTC/LightKernel.git && cd LightKernel && pip install --no-deps -v . && cd /root;

RUN apt-get update && apt-get install -y libnuma-dev && rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME=/usr/local/cuda \
    GDRCOPY_HOME=/usr/src/gdrdrv-2.4.4/

RUN set -e; \
    if [ "${ENABLE_CACHE}" = "1" ]; then \
      /opt/conda/bin/conda install -y boost && /opt/conda/bin/conda clean -ya; \
      LIGHTMEM_REF=5900baf92d85ef4dbda6124093506b0af906011a; \
      pip install --no-deps -v "git+https://github.com/ModelTC/LightMem.git@${LIGHTMEM_REF}#egg=light_mem"; \
    fi; \
    if [ "${ENABLE_NIXL}" = "1" ] || [ "${ENABLE_DEEPEP}" = "1" ]; then \
      apt-get update && apt-get install -y wget devscripts debhelper dh-make build-essential dkms && \
      apt-get install -y ibverbs-providers infiniband-diags perftest rdma-core libibverbs-dev librdmacm-dev && \
      rm -rf /var/lib/apt/lists/*; \
      mkdir -p /tmp/gdrcopy && cd /tmp \
      && git clone https://github.com/NVIDIA/gdrcopy.git -b v2.4.4 \
      && cd gdrcopy/packages \
      && CUDA=/usr/local/cuda ./build-deb-packages.sh \
      && dpkg -i gdrdrv-dkms_*.deb libgdrapi_*.deb gdrcopy-tests_*.deb gdrcopy_*.deb \
      && cd / && rm -rf /tmp/gdrcopy; \
    fi; \
    if [ "${ENABLE_DEEPEP}" = "1" ]; then \
      ln -sf /usr/lib/x86_64-linux-gnu/libmlx5.so.1 /usr/lib/x86_64-linux-gnu/libmlx5.so; \
      NVSHMEM_VERSION=3.3.9; \
      CUDA_ARCHS=90; \
      wget https://developer.download.nvidia.com/compute/redist/nvshmem/${NVSHMEM_VERSION}/source/nvshmem_src_cuda12-all-all-${NVSHMEM_VERSION}.tar.gz \
      && tar -xf nvshmem_src_cuda12-all-all-${NVSHMEM_VERSION}.tar.gz && mv nvshmem_src nvshmem \
      && cd nvshmem \
      && rm -f /root/nvshmem_src_cuda12-all-all-${NVSHMEM_VERSION}.tar.gz \
      && NVSHMEM_SHMEM_SUPPORT=0 \
         NVSHMEM_UCX_SUPPORT=0 \
         NVSHMEM_USE_NCCL=0 \
         NVSHMEM_MPI_SUPPORT=0 \
         NVSHMEM_IBGDA_SUPPORT=1 \
         NVSHMEM_PMIX_SUPPORT=0 \
         NVSHMEM_TIMEOUT_DEVICE_POLLING=0 \
         NVSHMEM_USE_GDRCOPY=1 \
         cmake -S . -B build/ -DCMAKE_INSTALL_PREFIX=/root/nvshmem/install -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS} \
      && cmake --build build --target install -j64; \
      DEEPEP_COMMIT=b6ce310bb0b75079682d09bc2ebc063a074fbd58; \
      git clone https://github.com/deepseek-ai/DeepEP.git && cd DeepEP && git checkout ${DEEPEP_COMMIT} && cd ..; \
      cd /root/DeepEP && NVSHMEM_DIR=/root/nvshmem/install python setup.py install; \
    fi; \
    if [ "${ENABLE_NIXL}" = "1" ]; then \
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
      git clone https://github.com/ai-dynamo/nixl.git -b main && \
      cd nixl && \
      rm -rf build && \
      mkdir build && \
      meson setup build/ --prefix=/usr/local/nixl --buildtype=release && \
      cd build && \
      ninja && \
      ninja install && \
      cd .. && pip install . --no-deps; \
    fi;

COPY . /lightllm
RUN pip install -e /lightllm --no-cache-dir
