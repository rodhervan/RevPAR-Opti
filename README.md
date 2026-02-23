# RevPAR-Opti: Reinforcement Learning for Hotel Revenue Management

This repository contains the codebase for optimization models and a Reinforcement Learning framework using Ray RLlib and DoubleML.


## System Requirements
* **OS:** Ubuntu or Windows with WSL2
* **GPU:** NVIDIA GeForce RTX50 series (Requires CUDA 12.8+ for Blackwell architecture support)
* **Python:** 3.12+

## Environment Setup Guide

Follow these steps to configure the isolated Linux virtual environment and enable distributed GPU training.

### 1. Navigate to the Project Directory
Open your WSL terminal (Ubuntu) and navigate to the native Linux project folder:
```bash
cd ~/RevPAR-Opti
```

### 2. Create and Activate the Virtual Environment

Create an isolated Python environment to prevent dependency conflicts with the system Python:

```bash
# Update package manager and install venv if necessary
sudo apt update
sudo apt install python3-venv python3-pip -y

# Create the environment named 'rl_env'
python3 -m venv rl_env

# Activate the environment
source rl_env/bin/activate
```

*(Note: You must run `source rl_env/bin/activate` every time you open a new terminal session to work on this project).*

### 3. Install Dependencies

Install the required packages from `requirements.txt`. Because this project utilizes PyTorch compiled specifically for CUDA 12.8, you must include the `--extra-index-url` flag so pip knows where to download the GPU-accelerated wheels.

```bash
pip install -r requirements.txt --extra-index-url [https://download.pytorch.org/whl/cu128](https://download.pytorch.org/whl/cu128)
```

### 4. Verify GPU Integration

Confirm that PyTorch is successfully communicating with the GPU through WSL2:

```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

*Expected Output: `CUDA Available: True` and `Device: NVIDIA GeForce RTX 50**`.*

### 5. IDE Configuration (VS Code / Cursor / Antigravity)

To ensure your IDE recognizes the Linux environment and provides correct linting/autocomplete:

1. Open the Command Palette (`Ctrl + Shift + P`).
2. Select **WSL: Reopen Folder in WSL** (if not already connected).
3. Open the Command Palette again and select **Python: Select Interpreter**.
4. Browse to and select `~/Ubuntu_Tesis_IIND/RevPAR-Opti/rl_env/bin/python`.

## Running the Training Loop

To execute the Reinforcement Learning agent:

```bash
python dummy_rl_training.py
```
