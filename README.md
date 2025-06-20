# AI Image Workflow Processor

An intelligent image processing application that uses AI to generate and execute OpenCV workflows based on natural language descriptions.

## Features

- 🤖 **AI-Powered Workflow Generation**: Describe what you want to do in natural language
- 🖼️ **20+ Image Processing Operations**: Blur, edge detection, color conversion, and more
- 🔄 **Workflow Execution**: Automatic execution of generated workflows
- 🎨 **Streamlit UI**: Beautiful web interface for easy interaction
- 📊 **Real-time Results**: See your processed images instantly

## Quick Start

### Option 1: Using the Setup Script (Recommended)

```bash
# Run the setup script
./setup.sh

# Set your OpenAI API key
export OPENAI_API_KEY=your_key_here

# Run the application
make run-venv
```

### Option 2: Using Makefile

```bash
# Setup environment and install dependencies
make setup

# Activate virtual environment
source .venv/bin/activate

# Set your OpenAI API key
export OPENAI_API_KEY=your_key_here

# Run the application
make run-venv
```

### Option 3: Manual Setup

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set your OpenAI API key
export OPENAI_API_KEY=your_key_here

# Run the application
streamlit run main.py
```

## Available Makefile Commands

```bash
make help          # Show all available commands
make setup         # Create virtual environment and install dependencies
make install       # Install dependencies in existing environment
make run           # Run the Streamlit application
make run-venv      # Run with virtual environment
make test          # Run basic tests
make clean         # Remove virtual environment and cache files
make format        # Format code with black
make lint          # Lint code with flake8
make type-check    # Run type checks with mypy
```

## Usage

1. **Upload an Image**: Use the file uploader to select an image (JPG, PNG)
2. **Describe Your Request**: Enter a natural language description of what you want to do
   - Examples:
     - "Convert to grayscale and blur the image"
     - "Detect edges and apply threshold"
     - "Increase brightness and contrast"
     - "Apply Gaussian blur with kernel size 7x7"
3. **Generate & Execute**: Click the button to let AI generate and run the workflow
4. **View Results**: See the processed image and the generated workflow JSON

## Available Image Processing Operations

The system supports 20+ OpenCV operations:

- **Blur Operations**: Gaussian blur, median blur, bilateral filter
- **Edge Detection**: Canny, Sobel
- **Color Operations**: RGB to grayscale, color space conversion
- **Thresholding**: Binary, adaptive threshold
- **Morphological**: Erosion, dilation, opening, closing
- **Enhancement**: Histogram equalization, CLAHE
- **Geometric**: Resize, crop
- **Adjustments**: Brightness/contrast, gamma correction
- **Effects**: Noise addition

## Requirements

- Python 3.8 or higher
- OpenAI API key
- Internet connection for AI workflow generation

## Dependencies

- `streamlit` - Web interface
- `opencv-python` - Image processing
- `numpy` - Numerical operations
- `Pillow` - Image handling
- `openai` - AI API integration

## Project Structure

```
ezit/
├── main.py                 # Streamlit frontend
├── workflow_processor.py   # Core workflow engine
├── node_spec.txt          # Node specifications
├── requirements.txt       # Python dependencies
├── Makefile              # Build and run commands
├── setup.sh              # Setup script
└── README.md             # This file
```

## Troubleshooting

### Common Issues

1. **"No module named 'cv2'"**
   ```bash
   pip install opencv-python
   ```

2. **OpenAI API Key Error**
   ```bash
   export OPENAI_API_KEY=your_actual_key_here
   ```

3. **Permission Denied on setup.sh**
   ```bash
   chmod +x setup.sh
   ```

4. **Virtual Environment Issues**
   ```bash
   make clean
   make setup
   ```

### Debug Mode

The application includes debug output to help troubleshoot issues:

- Node specification loading
- Workflow JSON generation
- Node execution tracking

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `make test`
5. Format code: `make format`
6. Submit a pull request

## License

This project is open source and available under the MIT License.

## Support

If you encounter any issues, please check the debug output and ensure all dependencies are properly installed. 