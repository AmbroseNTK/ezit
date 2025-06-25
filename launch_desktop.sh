#!/bin/bash

# Launch script for AI Image Workflow Processor Desktop App

echo "Starting AI Image Workflow Processor Desktop App..."

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if required packages are installed
echo "Checking dependencies..."
python3 -c "import PyQt6" 2>/dev/null || {
    echo "Error: PyQt6 is not installed. Please run: pip install PyQt6"
    exit 1
}

python3 -c "import cv2" 2>/dev/null || {
    echo "Error: OpenCV is not installed. Please run: pip install opencv-python"
    exit 1
}

python3 -c "import numpy" 2>/dev/null || {
    echo "Error: NumPy is not installed. Please run: pip install numpy"
    exit 1
}

# Try to import qfluentwidgets (optional)
python3 -c "import qfluentwidgets" 2>/dev/null || {
    echo "Warning: qfluentwidgets not found. App will use fallback UI components."
    echo "For full Fluent Design experience, install: pip install PyQt-Fluent-Widgets"
}

# Launch the application
echo "Launching desktop application..."
python3 desktop_app.py

# Check exit code
if [ $? -eq 0 ]; then
    echo "Application closed successfully."
else
    echo "Application exited with an error."
    exit 1
fi 