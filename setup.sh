#!/bin/bash

# Install Hugging Face Hub
pip install -r requirements.txt

# Create a directory for models
mkdir -p ./models

# Download the model using huggingface-cli
huggingface-cli download someone13574/mixtral-8x7b-32kseqlen --cache-dir ./models

# Navigate to the model's directory
cd ./models/someone13574/mixtral-8x7b-32kseqlen/snapshots/

# There should be only one directory in snapshots. Navigate into it.
cd $(ls -d */|head -n 1)

# Concatenate the model parts into a single file
cat consolidated.00.pth-split* > consolidated.00.pth
