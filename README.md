# MindCore: GPT-style Large Language Model From Scratch

This project implements a minimal, GPT-style Large Language Model (LLM) based on the Transformer decoder-only architecture using PyTorch, primarily for educational and learning purposes.

For comprehensive documentation on the architecture, configuration, and detailed usage, please refer to the [DOCUMENTATION.md](DOCUMENTATION.md) file.

## Quick Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd MindCore
    ```

2.  **Set up the environment:**
    The project uses a virtual environment and requires PyTorch.

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```
    *Note: This sets up a virtual environment and installs dependencies from `requirements.txt`.*

3.  **Activate the environment:**
    ```bash
    source .venv/bin/activate
    ```

## Quick Usage

### 1. Training the Model

Use the provided script to start the training process.

```bash
./scripts/run_train.sh
```

### 2. Running the API Server

Run the inference API using FastAPI/Uvicorn.

```bash
./scripts/run_api.sh
```
The API will typically be available at `http://127.0.0.1:8000`.