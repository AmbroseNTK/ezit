# AI Image Workflow Processor

An AI-powered image processing application that uses natural language prompts to generate and execute OpenCV workflows. Available as both a Streamlit web app and a PyQt5 desktop application with a modern dark theme.

## Features

- 🤖 **AI-Powered Workflow Generation**: Describe what you want to do in natural language
- 🖼️ **21+ Image Processing Operations**: Comprehensive OpenCV-based processing nodes
- 📊 **Visual Workflow Diagrams**: Interactive graph visualization of processing pipelines
- 🎨 **Modern Dark Theme**: Professional desktop interface with flat design
- 🔄 **Real-time Processing**: Multi-threaded workflow execution with progress tracking
- 📁 **File Management**: Open, process, and export images with ease

## Installation

### Prerequisites

- Python 3.8 or higher
- OpenAI API key

### Quick Start

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ezit
   ```

2. **Install dependencies**:
   ```bash
   make install
   ```

3. **Set your OpenAI API key**:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

## Usage

### Web Application (Streamlit)

Run the Streamlit web app for a browser-based experience:

```bash
make run-streamlit
```

### Desktop Application (PyQt5)

Run the PyQt5 desktop app for a native desktop experience:

```bash
make run-desktop
```

#### Desktop App Features

- **Onboarding Screen**: Welcome screen with app information and quick start
- **Image Viewer**: Zoom and pan capabilities with cursor position tracking
- **Workflow Drawer**: Side panel for AI workflow generation and visualization
- **Settings Dialog**: API key management and configuration
- **Menu Bar**: File operations, workflow controls, and settings access
- **Status Bar**: Image information and cursor position display

## Desktop App Installation

### macOS
```bash
make install-pyqt5-macos
```

### Ubuntu/Debian
```bash
make install-pyqt5-ubuntu
```

### Windows
```bash
make install-pyqt5-windows
```

## Available Commands

```bash
make install              # Install all dependencies
make run-streamlit        # Run Streamlit web app
make run-desktop          # Run PyQt5 desktop app
make clean                # Clean up Python cache files
make test                 # Run tests
make help                 # Show all available commands
```

## Project Structure

```
ezit/
├── desktop/                 # Desktop application module
│   ├── __init__.py
│   ├── theme.py            # Dark theme configuration
│   ├── widgets.py          # Custom PyQt5 widgets
│   └── main_window.py      # Main window implementation
├── desktop_app.py          # Desktop app entry point
├── main.py                 # Streamlit web app
├── workflow_processor.py   # AI workflow processing engine
├── node_spec.txt          # Available processing nodes
├── requirements.txt       # Python dependencies
├── Makefile              # Build and run commands
└── README.md             # This file
```

## Processing Nodes

The application supports 21+ image processing operations including:

- **Basic Operations**: Grayscale conversion, blur, sharpen, resize
- **Color Processing**: Brightness, contrast, saturation, hue adjustment
- **Edge Detection**: Canny, Sobel, Laplacian edge detection
- **Morphological Operations**: Erosion, dilation, opening, closing
- **Thresholding**: Binary, adaptive, Otsu thresholding
- **Text Overlay**: Add text with custom fonts and colors
- **Drawing**: Lines, rectangles, circles, polygons
- **Filters**: Gaussian, median, bilateral filtering

## Examples

### Natural Language Prompts

- "Convert to grayscale and blur the image"
- "Add text 'Hello World' in white at the center"
- "Detect edges and apply threshold"
- "Increase brightness by 50% and contrast by 30%"
- "Apply Gaussian blur with radius 5 and add a red border"

### Workflow Generation

The AI analyzes your prompt and generates a JSON workflow that defines:
- Processing nodes and their parameters
- Node connections and execution order
- Input/output specifications

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key for workflow generation

### Settings

The desktop app includes a settings dialog for:
- API key management
- Loading keys from environment variables
- Loading keys from `.env` files

## Development

### Running Tests
```bash
make test
```

### Code Style
The project follows PEP 8 guidelines and uses type hints.

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenCV for image processing capabilities
- OpenAI for AI-powered workflow generation
- PyQt5 for the desktop application framework
- Streamlit for the web application framework 