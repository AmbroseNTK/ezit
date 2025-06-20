# Makefile for AI Image Workflow Processor

.PHONY: help setup install run clean test

# Default target
help:
	@echo "Available targets:"
	@echo "  setup    - Create virtual environment and install dependencies"
	@echo "  install  - Install dependencies in existing environment"
	@echo "  run      - Run the Streamlit application"
	@echo "  clean    - Remove virtual environment and cache files"
	@echo "  test     - Run basic tests"

# Create virtual environment and install dependencies
setup:
	@echo "Creating virtual environment..."
	python3 -m venv .venv
	@echo "Activating virtual environment..."
	@echo "To activate manually: source .venv/bin/activate"
	@echo "Installing dependencies..."
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	@echo "Setup complete! Activate with: source .venv/bin/activate"

# Install dependencies in existing environment
install:
	@echo "Installing dependencies..."
	pip install --upgrade pip
	pip install -r requirements.txt
	@echo "Dependencies installed!"
	@echo "Note: If you want interactive workflow visualization, install streamlit-agraph:"
	@echo "pip install streamlit-agraph"

# Run the Streamlit application
run:
	@echo "Starting Streamlit application..."
	@echo "Make sure to set your OPENAI_API_KEY environment variable"
	@echo "You can set it with: export OPENAI_API_KEY=your_key_here"
	streamlit run main.py

# Run with virtual environment
run-venv:
	@echo "Starting Streamlit application with virtual environment..."
	.venv/bin/streamlit run main.py

# Run desktop application
run-desktop:
	@echo "Starting PyQt desktop application..."
	@echo "Make sure to set your OPENAI_API_KEY environment variable"
	@echo "You can set it with: export OPENAI_API_KEY=your_key_here"
	python desktop_app.py

# Run desktop application with virtual environment
run-desktop-venv:
	@echo "Starting PyQt desktop application with virtual environment..."
	.venv/bin/python desktop_app.py

# Clean up virtual environment and cache files
clean:
	@echo "Cleaning up..."
	rm -rf .venv
	rm -rf __pycache__
	rm -rf .streamlit
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cleanup complete!"

# Run basic tests
test:
	@echo "Running basic tests..."
	python -c "import cv2; print('OpenCV version:', cv2.__version__)"
	python -c "import numpy as np; print('NumPy version:', np.__version__)"
	python -c "import streamlit as st; print('Streamlit version:', st.__version__)"
	python -c "import openai; print('OpenAI version:', openai.__version__)"
	@echo "All imports successful!"

# Check if virtual environment exists
check-env:
	@if [ -d ".venv" ]; then \
		echo "Virtual environment exists. Activate with: source .venv/bin/activate"; \
	else \
		echo "Virtual environment not found. Run 'make setup' to create one."; \
	fi

# Install development dependencies
install-dev:
	@echo "Installing development dependencies..."
	pip install -r requirements.txt
	pip install streamlit-agraph
	pip install black flake8 mypy pytest
	@echo "Development dependencies installed!"

# Format code
format:
	@echo "Formatting code with black..."
	black *.py

# Lint code
lint:
	@echo "Linting code with flake8..."
	flake8 *.py

# Type check
type-check:
	@echo "Running type checks with mypy..."
	mypy *.py --ignore-missing-imports

# Install desktop dependencies
install-desktop:
	@echo "Installing desktop application dependencies..."
	pip install PyQt6 PyQt6-Qt6 PyQt6-sip qt-material
	@echo "Desktop dependencies installed!" 