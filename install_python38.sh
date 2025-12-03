#!/bin/bash
# Installation script for Python 3.8 compatibility

echo "Installing Python 3.8 compatible versions..."

# Install typing-extensions first
pip install typing-extensions>=4.0.0

# Install compatible ChromaDB version
pip install 'chromadb==0.4.18'

# Install other dependencies
pip install -r requirements.txt

echo "Installation complete!"

