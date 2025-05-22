#!/bin/bash

# Configuration
ENV_NAME="art-restoration-ai"
DISPLAY_NAME="art-restoration-ai"
ENV_YML="configs/environment.yml"

# Remove existing environment by name
if conda env list | grep -q "$ENV_NAME"; then
    echo "Conda environment '$ENV_NAME' already exists. Removing it..."
    conda env remove -n "$ENV_NAME"
fi

# If the environment folder still exists, delete it manually
ENV_PATH="$HOME/.conda/envs/$ENV_NAME"
if [ -d "$ENV_PATH" ]; then
    echo "Environment directory still exists at $ENV_PATH. Removing manually..."
    rm -rf "$ENV_PATH"
fi

# Create the environment
echo "Creating new conda environment '$ENV_NAME'..."
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
