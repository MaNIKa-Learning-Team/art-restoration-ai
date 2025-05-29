#!/bin/bash

# Configuration
ENV_NAMES=("art-restoration-ai" "sm-legacy")
DISPLAY_NAMES=("art-restoration-ai" "sm-legacy")
YML_FILES=("configs/art-restoration-ai.yml" "configs/sm-legacy.yml")

# Loop over environments
for i in "${!ENV_NAMES[@]}"; do
    ENV_NAME="${ENV_NAMES[$i]}"
    DISPLAY_NAME="${DISPLAY_NAMES[$i]}"
    ENV_YML="${YML_FILES[$i]}"

    # Remove existing environment by name
    if conda env list | grep -q "$ENV_NAME"; then
        echo "Conda environment '$ENV_NAME' already exists. Removing it..."
        conda env remove -n "$ENV_NAME"
    fi

    # If the environment folder still exists, delete it manually
    ENV_PATH="$HOME/.conda/envs/$ENV_NAME"
    if [ -d "$ENV_PATH" ]; then
        echo "Environment directory still exists at $ENV_PATH. Attempting forced deletion..."
        rm -rf "$ENV_PATH" || {
            echo "Standard rm failed. Trying with find..."
            find "$ENV_PATH" -type f -delete
            find "$ENV_PATH" -type d -delete
        }
    fi

    # Create the environment
    echo "Creating new conda environment '$ENV_NAME' from '$ENV_YML'..."
    conda env create -f "$ENV_YML"

    # Activate Conda
    CONDA_BASE=$(conda info --base)
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME" || {
        echo "Failed to activate conda environment '$ENV_NAME'."
        exit 1
    }

    # Register as a Jupyter kernel
    python -m ipykernel install --user --name "$ENV_NAME" --display-name "$DISPLAY_NAME"
    echo "Jupyter kernel '$DISPLAY_NAME' registered successfully."
done