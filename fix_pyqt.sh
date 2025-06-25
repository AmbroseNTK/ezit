#!/bin/bash

echo "Fixing PyQt6 installation issues..."

# Deactivate virtual environment if active
if [ -n "$VIRTUAL_ENV" ]; then
    echo "Deactivating virtual environment..."
    deactivate
fi

# Remove existing PyQt packages
echo "Removing existing PyQt packages..."
pip uninstall -y PyQt5 PyQt6 PyQt6-Qt6 PyQt6-sip PyQt6-Fluent-Widgets qt-material

# Clear pip cache
echo "Clearing pip cache..."
pip cache purge

# Install PyQt6 and dependencies in correct order
echo "Installing PyQt6 and dependencies..."
pip install PyQt6==6.6.0
pip install PyQt6-Qt6
pip install PyQt6-sip
pip install PyQt6-Fluent-Widgets[full]

# Install other required packages
echo "Installing additional dependencies..."
pip install matplotlib networkx

echo "PyQt6 installation fixed!"
echo "You can now run: make desktop" 