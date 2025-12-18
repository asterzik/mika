#!/bin/bash

# --- Configuration ---
REPO_URL="https://github.com/asterzik/mika.git"
REPO_NAME="lspri_analysis"  # The folder name created by git clone
ENV_FILE="environment.yml"
CONDA_ENV_NAME="mika"    # The name defined inside your environment.yml

# 1. Install Miniconda (if not already installed)
if ! command -v conda &> /dev/null; then
    echo "Conda not found. Installing Miniconda..."
    
    # Download the latest Miniconda installer for Linux
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    
    # Install silently to the home directory (-p $HOME/miniconda3)
    bash miniconda.sh -p $HOME/miniconda3
    
    # Remove the installer to save space
    rm miniconda.sh
    
    # Activate Conda for this script session specifically
    # (This avoids needing to restart the shell)
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
    echo "Conda is already installed."
    # Ensure conda is available in this subshell
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi

# 2. Clone the Repository
if [ -d "$REPO_NAME" ]; then
    echo "Repository '$REPO_NAME' already exists. Pulling latest changes..."
    cd "$REPO_NAME"
    git pull
else
    echo "Cloning repository..."
    git clone "$REPO_URL"
    cd "$REPO_NAME"
fi

# 3. Create/Update the Conda Environment
echo "Creating/Updating Conda environment from $ENV_FILE..."
conda env create -f "$ENV_FILE" || conda env update -f "$ENV_FILE" --prune

# 4. Activate Environment and Run
echo "Activating environment '$CONDA_ENV_NAME'..."
conda activate "$CONDA_ENV_NAME"

echo "Running application..."
python app.py

echo "=========================================="
echo "Done! Application finished."
echo "=========================================="
