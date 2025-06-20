import sys
import os
import cv2
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QTextEdit, 
                             QFileDialog, QProgressBar, QSplitter, QFrame,
                             QScrollArea, QGridLayout, QGroupBox, QMessageBox,
                             QTabWidget, QTextBrowser, QComboBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QPixmap, QImage, QFont, QPalette, QColor, QIcon
from qt_material import apply_stylesheet
from workflow_processor import process_image_with_ai_workflow
import json

class WorkflowThread(QThread):
    """Thread for processing workflow to avoid blocking UI"""
    finished = pyqtSignal(np.ndarray, dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)
    
    def __init__(self, image, prompt, api_key):
        super().__init__()
        self.image = image
        self.prompt = prompt
        self.api_key = api_key
    
    def run(self):
        try:
            self.progress.emit(10)
            result_image, workflow_json = process_image_with_ai_workflow(
                self.image, self.prompt, self.api_key
            )
            self.progress.emit(100)
            self.finished.emit(result_image, workflow_json)
        except Exception as e:
            self.error.emit(str(e))

class ImageDisplayWidget(QLabel):
    """Custom widget for displaying images with zoom and pan"""
    
    def __init__(self):
        super().__init__()
        self.setMinimumSize(400, 300)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #cccccc;
                border-radius: 8px;
                background-color: #f8f9fa;
                color: #6c757d;
            }
        """)
        self.setText("No image loaded")
        self.original_pixmap = None
        self.scale_factor = 1.0
    
    def set_image(self, image):
        """Set image from numpy array"""
        if image is None:
            self.setText("No image loaded")
            self.original_pixmap = None
            return
        
        # Convert numpy array to QPixmap
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                # BGR to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            height, width, channel = image_rgb.shape
            bytes_per_line = 3 * width
            q_image = QImage(image_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        else:
            # Grayscale
            height, width = image.shape
            bytes_per_line = width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)
        
        self.original_pixmap = QPixmap.fromImage(q_image)
        self.update_display()
    
    def update_display(self):
        """Update the displayed image with current scale"""
        if self.original_pixmap:
            scaled_pixmap = self.original_pixmap.scaled(
                self.original_pixmap.size() * self.scale_factor,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.setPixmap(scaled_pixmap)
    
    def wheelEvent(self, event):
        """Handle mouse wheel for zooming"""
        if self.original_pixmap:
            delta = event.angleDelta().y()
            if delta > 0:
                self.scale_factor *= 1.1
            else:
                self.scale_factor *= 0.9
            
            self.scale_factor = max(0.1, min(5.0, self.scale_factor))
            self.update_display()

class WorkflowVisualizationWidget(QWidget):
    """Widget for displaying workflow visualization"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Workflow Visualization")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        layout.addWidget(title)
        
        # Workflow display area
        self.workflow_display = QTextBrowser()
        self.workflow_display.setMinimumHeight(200)
        self.workflow_display.setStyleSheet("""
            QTextBrowser {
                background-color: #ffffff;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 8px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 11px;
            }
        """)
        layout.addWidget(self.workflow_display)
        
        self.setLayout(layout)
    
    def update_workflow(self, workflow_json):
        """Update the workflow visualization"""
        if not workflow_json:
            self.workflow_display.setText("No workflow data")
            return
        
        # Create a simple text-based workflow representation
        text = "WORKFLOW DIAGRAM\n"
        text += "=" * 50 + "\n\n"
        
        # Show nodes
        text += "NODES:\n"
        for i, node in enumerate(workflow_json.get('nodes', []), 1):
            node_type = node['node_type']
            node_id = node['node_id']
            
            # Add icons based on node type
            if node_type == 'input':
                icon = "📥"
            elif node_type == 'output':
                icon = "📤"
            else:
                icon = "⚙️"
            
            text += f"{i}. {icon} {node_type} ({node_id})\n"
        
        text += "\nCONNECTIONS:\n"
        for i, conn in enumerate(workflow_json.get('connections', []), 1):
            text += f"{i}. {conn['from_node']} → {conn['to_node']}\n"
        
        text += "\nDETAILED NODE INFO:\n"
        for node in workflow_json.get('nodes', []):
            text += f"\n{node['node_type']} ({node['node_id']}):\n"
            if node.get('parameters'):
                for key, value in node['parameters'].items():
                    text += f"  {key}: {value}\n"
            else:
                text += "  No parameters\n"
        
        self.workflow_display.setText(text)

class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.workflow_thread = None
        self.original_image = None
        self.result_image = None
        
    def init_ui(self):
        self.setWindowTitle("AI Image Workflow Processor - Desktop")
        self.setGeometry(100, 100, 1400, 900)
        
        # Set application icon (if available)
        try:
            self.setWindowIcon(QIcon("icon.png"))
        except:
            pass
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout()
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel - Input controls
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)
        
        # Right panel - Results and visualization
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)
        
        # Set splitter proportions
        splitter.setSizes([400, 1000])
        
        main_layout.addWidget(splitter)
        central_widget.setLayout(main_layout)
        
        # Apply Fluent Design theme
        apply_stylesheet(self, theme='light_blue.xml')
    
    def create_left_panel(self):
        """Create the left control panel"""
        panel = QFrame()
        panel.setFrameStyle(QFrame.Shape.StyledPanel)
        panel.setMaximumWidth(400)
        
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("AI Image Workflow Processor")
        title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Description
        desc = QLabel("Upload an image and describe what you want to do in natural language. Let AI build and execute an OpenCV workflow for you!")
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #6c757d; margin: 10px;")
        layout.addWidget(desc)
        
        # API Key input
        api_group = QGroupBox("OpenAI API Configuration")
        api_layout = QVBoxLayout()
        
        self.api_key_input = QTextEdit()
        self.api_key_input.setMaximumHeight(60)
        self.api_key_input.setPlaceholderText("Enter your OpenAI API key...")
        api_layout.addWidget(self.api_key_input)
        
        # Load from environment
        env_button = QPushButton("Load from Environment")
        env_button.clicked.connect(self.load_api_key_from_env)
        api_layout.addWidget(env_button)
        
        api_group.setLayout(api_layout)
        layout.addWidget(api_group)
        
        # Image upload
        image_group = QGroupBox("Image Upload")
        image_layout = QVBoxLayout()
        
        self.upload_button = QPushButton("📁 Upload Image")
        self.upload_button.clicked.connect(self.upload_image)
        image_layout.addWidget(self.upload_button)
        
        self.image_info = QLabel("No image selected")
        self.image_info.setStyleSheet("color: #6c757d; font-size: 12px;")
        image_layout.addWidget(self.image_info)
        
        image_group.setLayout(image_layout)
        layout.addWidget(image_group)
        
        # Prompt input
        prompt_group = QGroupBox("Processing Request")
        prompt_layout = QVBoxLayout()
        
        self.prompt_input = QTextEdit()
        self.prompt_input.setPlaceholderText("Describe what you want to do...\n\nExamples:\n• Convert to grayscale and blur the image\n• Add text 'Hello World' in white\n• Detect edges and apply threshold\n• Increase brightness and contrast")
        self.prompt_input.setMaximumHeight(150)
        prompt_layout.addWidget(self.prompt_input)
        
        prompt_group.setLayout(prompt_layout)
        layout.addWidget(prompt_group)
        
        # Process button
        self.process_button = QPushButton("🚀 Generate & Execute Workflow")
        self.process_button.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                padding: 12px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.process_button.clicked.connect(self.process_workflow)
        layout.addWidget(self.process_button)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Status
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #28a745; font-weight: bold;")
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        panel.setLayout(layout)
        return panel
    
    def create_right_panel(self):
        """Create the right results panel"""
        panel = QFrame()
        panel.setFrameStyle(QFrame.Shape.StyledPanel)
        
        layout = QVBoxLayout()
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Images tab
        images_tab = QWidget()
        images_layout = QHBoxLayout()
        
        # Original image
        original_group = QGroupBox("Original Image")
        original_layout = QVBoxLayout()
        self.original_image_widget = ImageDisplayWidget()
        original_layout.addWidget(self.original_image_widget)
        original_group.setLayout(original_layout)
        images_layout.addWidget(original_group)
        
        # Result image
        result_group = QGroupBox("Result Image")
        result_layout = QVBoxLayout()
        self.result_image_widget = ImageDisplayWidget()
        result_layout.addWidget(self.result_image_widget)
        result_group.setLayout(result_layout)
        images_layout.addWidget(result_group)
        
        images_tab.setLayout(images_layout)
        self.tab_widget.addTab(images_tab, "📸 Images")
        
        # Workflow tab
        workflow_tab = QWidget()
        workflow_layout = QVBoxLayout()
        self.workflow_widget = WorkflowVisualizationWidget()
        workflow_layout.addWidget(self.workflow_widget)
        workflow_tab.setLayout(workflow_layout)
        self.tab_widget.addTab(workflow_tab, "📊 Workflow")
        
        # JSON tab
        json_tab = QWidget()
        json_layout = QVBoxLayout()
        self.json_display = QTextBrowser()
        self.json_display.setStyleSheet("""
            QTextBrowser {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 8px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 11px;
            }
        """)
        json_layout.addWidget(self.json_display)
        json_tab.setLayout(json_layout)
        self.tab_widget.addTab(json_tab, "🔧 JSON")
        
        layout.addWidget(self.tab_widget)
        panel.setLayout(layout)
        return panel
    
    def load_api_key_from_env(self):
        """Load API key from environment variable"""
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.api_key_input.setText(api_key)
            self.status_label.setText("API key loaded from environment")
            self.status_label.setStyleSheet("color: #28a745; font-weight: bold;")
        else:
            self.status_label.setText("No API key found in environment")
            self.status_label.setStyleSheet("color: #dc3545; font-weight: bold;")
    
    def upload_image(self):
        """Handle image upload"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if file_path:
            try:
                # Load image using OpenCV
                image = cv2.imread(file_path)
                if image is not None:
                    self.original_image = image
                    self.original_image_widget.set_image(image)
                    
                    # Update image info
                    height, width = image.shape[:2]
                    channels = image.shape[2] if len(image.shape) == 3 else 1
                    self.image_info.setText(f"Image: {width}x{height}, {channels} channel(s)")
                    
                    self.status_label.setText("Image loaded successfully")
                    self.status_label.setStyleSheet("color: #28a745; font-weight: bold;")
                else:
                    raise Exception("Failed to load image")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load image: {str(e)}")
                self.status_label.setText("Failed to load image")
                self.status_label.setStyleSheet("color: #dc3545; font-weight: bold;")
    
    def process_workflow(self):
        """Process the workflow"""
        # Validate inputs
        api_key = self.api_key_input.toPlainText().strip()
        if not api_key:
            QMessageBox.warning(self, "Warning", "Please enter your OpenAI API key")
            return
        
        if self.original_image is None:
            QMessageBox.warning(self, "Warning", "Please upload an image first")
            return
        
        prompt = self.prompt_input.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "Warning", "Please enter a processing request")
            return
        
        # Start processing
        self.process_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Processing...")
        self.status_label.setStyleSheet("color: #ffc107; font-weight: bold;")
        
        # Create and start processing thread
        self.workflow_thread = WorkflowThread(self.original_image, prompt, api_key)
        self.workflow_thread.finished.connect(self.on_workflow_finished)
        self.workflow_thread.error.connect(self.on_workflow_error)
        self.workflow_thread.progress.connect(self.progress_bar.setValue)
        self.workflow_thread.start()
    
    def on_workflow_finished(self, result_image, workflow_json):
        """Handle workflow completion"""
        self.result_image = result_image
        self.result_image_widget.set_image(result_image)
        
        # Update workflow visualization
        self.workflow_widget.update_workflow(workflow_json)
        
        # Update JSON display
        self.json_display.setText(json.dumps(workflow_json, indent=2))
        
        # Reset UI
        self.process_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Workflow completed successfully!")
        self.status_label.setStyleSheet("color: #28a745; font-weight: bold;")
        
        # Switch to results tab
        self.tab_widget.setCurrentIndex(0)
    
    def on_workflow_error(self, error_message):
        """Handle workflow error"""
        QMessageBox.critical(self, "Error", f"Workflow processing failed:\n{error_message}")
        
        # Reset UI
        self.process_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Processing failed")
        self.status_label.setStyleSheet("color: #dc3545; font-weight: bold;")

def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("AI Image Workflow Processor")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("AI Workflow Tools")
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Start event loop
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 