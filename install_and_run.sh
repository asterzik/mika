#!/bin/bash
set -e

echo "=========================================="
echo "1. Preparing System Dependencies..."
echo "=========================================="
PACKAGES="wget git libgl1 libegl1 libopengl0 libxkbcommon-x11-0 libdbus-1-3 libxcb-cursor0 libxcb-icccm4 libxcb-keysyms1 libxcb-shape0"

if command -v sudo &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y $PACKAGES
else
    apt-get update
    apt-get install -y $PACKAGES
fi

echo "=========================================="
echo "2. Installing Miniforge (Conda Alternative)..."
echo "=========================================="

# Check if conda is already installed
if ! command -v conda &> /dev/null; then
    wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" -O miniforge.sh
    
    # Run Installer Silently
    # -b : Batch mode (Accept license automatically, no questions)
    # -p : Installation path
    bash miniforge.sh -b -p "$HOME/miniforge3"
    
    # Cleanup
    rm miniforge.sh
    
    # Activate for this script session
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
else
    echo "Conda/Miniforge is already installed."
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi

echo "=========================================="
echo "3. Setup Environment & Run..."
echo "=========================================="

REPO_DIR="mika"
if [ ! -d "$REPO_DIR" ]; then
    git clone https://github.com/asterzik/mika.git
fi
cd "$REPO_DIR"

# Create Environment
conda env create -f environment.yml -y

conda clean -a -y

# Activate
conda activate mika

# Run App
python app.py
