#!/bin/bash

# AI Image Workflow Processor Setup Script

set -e

echo "🚀 Setting up AI Image Workflow Processor..."

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "✅ Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
    echo "✅ Virtual environment created!"
else
    echo "✅ Virtual environment already exists!"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

echo "✅ Setup complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Set your OpenAI API key:"
echo "   export OPENAI_API_KEY=your_key_here"
echo ""
echo "2. Run the application:"
echo "   make run-venv"
echo "   or"
echo "   streamlit run main.py"
echo ""
echo "3. Open your browser and go to: http://localhost:8501"
echo ""
echo "📖 Available commands:"
echo "   make help     - Show all available commands"
echo "   make test     - Run basic tests"
echo "   make clean    - Clean up environment"
echo "   make format   - Format code"
echo "   make lint     - Lint code" 