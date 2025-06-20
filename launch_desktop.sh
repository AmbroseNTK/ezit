#!/bin/bash

# AI Image Workflow Processor - Desktop Launcher

set -e

echo "🚀 Launching AI Image Workflow Processor - Desktop"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Please run setup first:"
    echo "   ./setup.sh"
    exit 1
fi

# Check if desktop dependencies are installed
if ! .venv/bin/python -c "import PyQt6" 2>/dev/null; then
    echo "📦 Installing desktop dependencies..."
    .venv/bin/pip install PyQt6 PyQt6-Qt6 PyQt6-sip qt-material
    echo "✅ Desktop dependencies installed!"
fi

# Check for API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  Warning: OPENAI_API_KEY not set"
    echo "   You can set it with: export OPENAI_API_KEY=your_key_here"
    echo "   Or enter it in the application when prompted"
fi

echo "🎨 Starting desktop application..."
.venv/bin/python desktop_app.py 