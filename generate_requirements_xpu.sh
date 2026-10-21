#!/bin/bash
# Script to generate requirements-xpu.txt from requirements.txt
# Intel XPU is not compatible with CUDA packages, so they need to be removed
# Torch-related packages come from the Intel XPU wheel index and are installed beforehand

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_FILE="${SCRIPT_DIR}/requirements.txt"
OUTPUT_FILE="${SCRIPT_DIR}/requirements-xpu.txt"

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: requirements.txt not found at $INPUT_FILE"
    exit 1
fi

echo "Generating requirements-xpu.txt from requirements.txt..."

# Define patterns to remove (CUDA-specific packages)
# These packages are not compatible with Intel XPU
CUDA_PACKAGES=(
    "^cupy"              # cupy-cuda13x and similar
    "^cuda_bindings"     # CUDA bindings
    "^cuda-tile"         # CUDA tile DSL
    "^nixl"              # NIXL (NVIDIA Inter-node eXchange Library)
    "^flashinfer"        # flashinfer-python / flashinfer-cubin (CUDA-specific attention kernels)
    "^sgl-kernel"        # SGL kernel (CUDA build); XPU uses the sgl-kernel-xpu wheel
    "^sglang-kernel"     # SGL kernel (CUDA build)
    "^nvidia-"           # nvidia-cutlass-dsl, nvidia-cudnn-frontend
    "^nccl4py"           # NCCL python bindings
    "^xformers"          # xformers (CUDA build)
    "^torch_memory_saver" # CUDA-only memory saver
)

# Define torch-related packages (installed from the XPU wheel index, remove version pins)
TORCH_PACKAGES=(
    "^torch=="
    "^torchvision=="
)

# Packages that must be installed with --no-deps, otherwise they pull in CUDA triton
NO_DEPS_PACKAGES=(
    "^xgrammar=="
)

# Create the output file with a header comment
cat > "$OUTPUT_FILE" << 'EOF'
# Requirements for Intel XPU (Arc / Data Center GPU)
# Auto-generated from requirements.txt by generate_requirements_xpu.sh
# CUDA-specific packages have been removed
# Torch-related packages have version pins removed (install torch/torchvision/pytorch-triton-xpu from the XPU index first)
# Packages listed at the end as comments must be installed with --no-deps

EOF

NO_DEPS_LINES=()

# Process the requirements file
while IFS= read -r line || [ -n "$line" ]; do
    # Skip empty lines and comments (but keep them in output)
    if [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]]; then
        echo "$line" >> "$OUTPUT_FILE"
        continue
    fi

    # Drop extra index urls (the FlashInfer index is CUDA-only)
    if [[ "$line" =~ ^--extra-index-url ]]; then
        echo "  Removing index: $line"
        continue
    fi

    # Extract package name (before ==, >=, <=, ~=, etc.)
    pkg_name=$(echo "$line" | sed -E 's/^([a-zA-Z0-9_-]+).*/\1/')

    # Check if this is a CUDA package to skip
    skip=false
    for pattern in "${CUDA_PACKAGES[@]}"; do
        if [[ "$pkg_name" =~ $pattern ]]; then
            echo "  Removing CUDA package: $line"
            skip=true
            break
        fi
    done

    if $skip; then
        continue
    fi

    # Check if this is a torch-related package (remove version pin)
    for pattern in "${TORCH_PACKAGES[@]}"; do
        if [[ "$line" =~ $pattern ]]; then
            # Remove version pin, keep just the package name
            pkg_only=$(echo "$line" | sed -E 's/==.*//')
            echo "  Unpinning version for: $pkg_only (installed from the XPU wheel index)"
            echo "$pkg_only" >> "$OUTPUT_FILE"
            skip=true
            break
        fi
    done

    if $skip; then
        continue
    fi

    # Check if this package must be installed without dependencies
    for pattern in "${NO_DEPS_PACKAGES[@]}"; do
        if [[ "$line" =~ $pattern ]]; then
            echo "  Moving to --no-deps list: $line"
            NO_DEPS_LINES+=("$line")
            skip=true
            break
        fi
    done

    if $skip; then
        continue
    fi

    # Keep the package as-is
    echo "$line" >> "$OUTPUT_FILE"

done < "$INPUT_FILE"

# Record the --no-deps packages at the end
{
    echo ""
    echo "# Install separately with: pip install --no-deps <package>"
    for line in "${NO_DEPS_LINES[@]}"; do
        echo "# $line"
    done
} >> "$OUTPUT_FILE"

echo ""
echo "Successfully generated: $OUTPUT_FILE"
echo ""
echo "Summary of changes:"
echo "  - Removed CUDA-specific packages: cupy, cuda_bindings, cuda-tile, nixl, flashinfer, sgl-kernel, nvidia-*, nccl4py, xformers, torch_memory_saver"
echo "  - Unpinned torch-related packages: torch, torchvision (install from the XPU wheel index)"
echo "  - Install separately with --no-deps: ${NO_DEPS_LINES[*]}"
