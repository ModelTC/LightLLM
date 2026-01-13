#!/bin/bash
# Script to generate requirements-musa.txt from requirements.txt
# MUSA is not compatible with CUDA packages, so they need to be removed
# Torch-related packages are pre-installed in the MUSA docker container

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_FILE="${SCRIPT_DIR}/requirements.txt"
OUTPUT_FILE="${SCRIPT_DIR}/requirements-musa.txt"

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: requirements.txt not found at $INPUT_FILE"
    exit 1
fi

echo "Generating requirements-musa.txt from requirements.txt..."

# Define patterns to remove (CUDA-specific packages)
# These packages are not compatible with MUSA
CUDA_PACKAGES=(
    "^cupy"           # cupy-cuda12x and similar
    "^cuda_bindings"  # CUDA bindings
    "^nixl"           # NIXL (NVIDIA Inter-node eXchange Library)
    "^flashinfer"     # flashinfer-python (CUDA-specific attention kernel)
    "^sgl-kernel"     # SGL kernel (CUDA-specific)
)

# Define torch-related packages (pre-installed in MUSA container, remove version pins)
TORCH_PACKAGES=(
    "^torch=="
    "^torchvision=="
)

# Create the output file with a header comment
cat > "$OUTPUT_FILE" << 'EOF'
# Requirements for MUSA (Moore Threads GPU)
# Auto-generated from requirements.txt by generate_requirements_musa.sh
# CUDA-specific packages have been removed
# Torch-related packages have version pins removed (pre-installed in MUSA container)

EOF

# Process the requirements file
while IFS= read -r line || [ -n "$line" ]; do
    # Skip empty lines and comments (but keep them in output)
    if [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]]; then
        echo "$line" >> "$OUTPUT_FILE"
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
            echo "  Unpinning version for: $pkg_only (pre-installed in MUSA container)"
            echo "$pkg_only" >> "$OUTPUT_FILE"
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

# Add MUSA-specific packages at the end
cat >> "$OUTPUT_FILE" << 'EOF'

# MUSA-specific packages
torch_musa
torchada
EOF

echo ""
echo "Successfully generated: $OUTPUT_FILE"
echo ""
echo "Summary of changes:"
echo "  - Removed CUDA-specific packages: cupy-cuda12x, cuda_bindings, nixl, flashinfer-python, sgl-kernel"
echo "  - Unpinned torch-related packages: torch, torchvision (pre-installed in MUSA container)"
echo "  - Added MUSA-specific packages: torch_musa, torchada"

