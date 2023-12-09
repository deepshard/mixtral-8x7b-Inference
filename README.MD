# Mistral Simple Inference Server

## Introduction
Welcome to the Mistral Simple Inference Server, a straightforward setup to run inferences using the Mixtral model, recently released by Mistral. This guide will help you set up and run a basic inference server for the Mixtral model.

## Getting Started

### Step 1: Clone the Repository
First, you need to clone the repository containing the necessary scripts and files for the inference server.

```bash
git clone [URL to Mistral Inference Server repository]
cd [Repository Name]
```

### Step 2: Setup Environment

Before running the server, you need to set up the environment. This includes installing dependencies and preparing the model files.

Make setup.sh executable:

```bash
chmod +x setup.sh
```

Run the setup script:

```bash
./setup.sh
```

This script will install the Hugging Face Hub package, create a directory for models, download the Mixtral model, and consolidate the model files into a single file.

### Step 3: Run the Server

To run the server, use the following command:

```bash
python server.py {MODEL_DIR} --max_batch_size {BATCH_SIZE} --num_gpus {NUM_GPUS}
```

Replace the placeholders with appropriate values:
- {MODEL_DIR}: Path to the directory containing the consolidated model file.
- {BATCH_SIZE}: The maximum batch size for generation.
- {NUM_GPUS}: The number of GPUs to use.

Hardware Requirements

- Ensure you have at least 86GB of VRAM available for the server to run without any quantization.

### Step 4: Making Inference Requests

To send an inference request to the server, use the following curl command:

```bash
curl -X POST http://localhost:5000/generate -H "Content-Type: application/json" -d '{"prompts": ["Hello, world!"]}'
```
Additional Parameters

Alongside the prompts in your request, you can control the following parameters:

- temperature: Controls the randomness of the generation. Higher values lead to more random outputs. Typically between 0 and 1.
- top_p: Controls the diversity of the generation. A lower top_p reduces the likelihood of low-probability words being chosen. Typically between 0 and 1.
- max_gen_len: The maximum length of the generated sequence.

Example with additional parameters:

```bash
curl -X POST http://localhost:5000/generate -H "Content-Type: application/json" -d '{"prompts": ["Hello, world!"], "temperature": 0.7, "top_p": 0.9, "max_gen_len": 50}'
```