#!/usr/bin/env python3
import sys
from PyQt5.QtWidgets import QApplication

from desktop.main_window import MainWindow
from desktop.theme import apply_dark_theme

def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("AI Image Workflow Processor")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("AI Workflow Tools")
    
    # Apply dark theme
    apply_dark_theme(app)
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Start event loop
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
