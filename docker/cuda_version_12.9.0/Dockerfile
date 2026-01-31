ARG CUDA_VERSION=12.9.0
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu22.04
ARG PYTHON_VERSION=3.10
ARG TARGETPLATFORM

ENV PATH=/root/.local/bin:/root/.cargo/bin:$PATH \
    UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=0 \
    UV_HTTP_TIMEOUT=300

# Install system dependencies
RUN chmod 777 -R /tmp && apt-get update --allow-insecure-repositories && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl-dev \
    curl \
    g++ \
    make \
    git \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-venv \
    python3-pip \
    libnuma-dev && \
    rm -rf /var/lib/apt/lists/*

# Set python alias
RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1

# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /lightllm

# Copy project files for dependency installation
COPY pyproject.toml README.md ./
COPY lightllm/__init__.py lightllm/

# Install dependencies using uv (faster than pip)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -e . --no-cache-dir --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu129

# Install vllm for additional kernels
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system vllm --pre --index-strategy unsafe-best-match --extra-index-url https://wheels.vllm.ai/nightly

# Copy full source code and install in editable mode
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -e . --no-deps

WORKDIR /root
