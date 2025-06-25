# AI Image Workflow Processor Makefile

.PHONY: install run-streamlit run-desktop clean test

# Install dependencies
install:
	pip install -r requirements.txt

# Run Streamlit web app
run-streamlit:
	streamlit run main.py

# Run PyQt5 desktop app
run-desktop:
	python3 desktop_app.py

# Clean up
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

# Run tests
test:
	python -m pytest tests/ -v

# Install PyQt5 dependencies (macOS)
install-pyqt5-macos:
	brew install pyqt@5
	pip install PyQt5

# Install PyQt5 dependencies (Ubuntu/Debian)
install-pyqt5-ubuntu:
	sudo apt-get update
	sudo apt-get install python3-pyqt5 python3-pyqt5.qtcore python3-pyqt5.qtgui python3-pyqt5.qtwidgets
	pip install PyQt5

# Install PyQt5 dependencies (Windows)
install-pyqt5-windows:
	pip install PyQt5

# Show help
help:
	@echo "Available commands:"
	@echo "  install              - Install all dependencies"
	@echo "  run-streamlit        - Run Streamlit web app"
	@echo "  run-desktop          - Run PyQt5 desktop app"
	@echo "  clean                - Clean up Python cache files"
	@echo "  test                 - Run tests"
	@echo "  install-pyqt5-macos  - Install PyQt5 on macOS"
	@echo "  install-pyqt5-ubuntu - Install PyQt5 on Ubuntu/Debian"
	@echo "  install-pyqt5-windows- Install PyQt5 on Windows"
	@echo "  help                 - Show this help message" 