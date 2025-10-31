uv build \
  -C override=cmake.options.Python_ROOT_DIR=$PWD/.venv \
  -C override=cmake.options.CMAKE_PREFIX_PATH=$(python -c "import torch;print(torch.utils.cmake_prefix_path,end='')") \
  -C override=cmake.options.CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc

pip uninstall -y kvcache
pip install dist/*.whl
