# Custom widgets for the desktop app

import cv2
import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QTextEdit, QProgressBar, QTabWidget, 
                             QTextBrowser, QGraphicsView, QGraphicsScene, 
                             QGraphicsPixmapItem, QScrollBar, QLineEdit, 
                             QDialog, QGroupBox, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPoint
from PyQt5.QtGui import QPixmap, QImage, QFont, QPainter, QPen, QWheelEvent, QMouseEvent

# Try to import matplotlib and networkx for graph visualization
try:
    import matplotlib
    matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import networkx as nx
    GRAPH_AVAILABLE = True
except ImportError:
    GRAPH_AVAILABLE = False
    class Figure:
        def __init__(self, *args, **kwargs):
            pass
        def add_subplot(self, *args, **kwargs):
            return None
        def clear(self):
            pass
        def tight_layout(self):
            pass
    
    class FigureCanvas(QWidget):
        def __init__(self, figure):
            super().__init__()
            self.figure = figure
        def draw(self):
            pass
    
    class FallbackNetworkX:
        class DiGraph:
            def __init__(self):
                self._nodes = {}
                self._edges = []
            def add_node(self, node_id, **kwargs):
                self._nodes[node_id] = kwargs
            def add_edge(self, from_node, to_node):
                self._edges.append((from_node, to_node))
            def nodes(self):
                return self._nodes.keys()
            def edges(self):
                return self._edges
        
        @staticmethod
        def spring_layout(G, **kwargs):
            return {}
        
        @staticmethod
        def draw_networkx_nodes(G, pos, **kwargs):
            pass
        
        @staticmethod
        def draw_networkx_edges(G, pos, **kwargs):
            pass
        
        @staticmethod
        def draw_networkx_labels(G, pos, **kwargs):
            pass
    
    nx = FallbackNetworkX()

from .theme import DARK_THEME

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
            from workflow_processor import process_image_with_ai_workflow
            result_image, workflow_json = process_image_with_ai_workflow(
                self.image, self.prompt, self.api_key
            )
            self.progress.emit(100)
            self.finished.emit(result_image, workflow_json)
        except Exception as e:
            self.error.emit(str(e))

class ImageViewer(QGraphicsView):
    """Custom image viewer with zoom and pan capabilities"""
    
    def __init__(self):
        super().__init__()
        self._scene = QGraphicsScene()
        self.setScene(self._scene)
        self.pixmap_item = None
        self.zoom_factor = 1.0
        self.pan_start = QPoint()
        self.panning = False
        
        # Set up the view
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        
        # Apply dark style
        self.setStyleSheet(f"""
            QGraphicsView {{
                background-color: {DARK_THEME['bg_tertiary']};
                border: 1px solid {DARK_THEME['border']};
                border-radius: 4px;
            }}
        """)
    
    def set_image(self, image):
        """Set image from numpy array"""
        if image is None:
            self._scene.clear()
            self.pixmap_item = None
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
            q_image = QImage(image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        else:
            # Grayscale
            height, width = image.shape
            bytes_per_line = width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        
        pixmap = QPixmap.fromImage(q_image)
        
        # Clear scene and add new pixmap
        self._scene.clear()
        self.pixmap_item = self._scene.addPixmap(pixmap)
        
        # Fit to view
        self.fit_to_view()
    
    def fit_to_view(self):
        """Fit image to view"""
        if self.pixmap_item:
            self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
            self.zoom_factor = 1.0
    
    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel for zooming"""
        if self.pixmap_item:
            # Zoom factor
            zoom_in_factor = 1.25
            zoom_out_factor = 1 / zoom_in_factor
            
            # Save the scene pos
            old_pos = self.mapToScene(event.pos())
            
            # Zoom
            if event.angleDelta().y() > 0:
                zoom_factor = zoom_in_factor
            else:
                zoom_factor = zoom_out_factor
            
            self.scale(zoom_factor, zoom_factor)
            self.zoom_factor *= zoom_factor
            
            # Get the new position
            new_pos = self.mapToScene(event.pos())
            
            # Move scene to old position
            delta = new_pos - old_pos
            self.translate(delta.x(), delta.y())
    
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press for panning"""
        if event.button() == Qt.LeftButton:
            self.pan_start = event.pos()
            self.panning = True
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move for panning"""
        if self.panning:
            delta = event.pos() - self.pan_start
            h_scrollbar = self.horizontalScrollBar()
            v_scrollbar = self.verticalScrollBar()
            if h_scrollbar:
                h_scrollbar.setValue(h_scrollbar.value() - delta.x())
            if v_scrollbar:
                v_scrollbar.setValue(v_scrollbar.value() - delta.y())
            self.pan_start = event.pos()
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release"""
        if event.button() == Qt.LeftButton:
            self.panning = False
        super().mouseReleaseEvent(event)

class OnboardingScreen(QWidget):
    """Onboarding screen shown when no image is loaded"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.init_ui()
        self.setMinimumSize(800, 600)
    
    def init_ui(self):
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(20)
        layout.setContentsMargins(40, 40, 40, 40)
        
        # App icon/logo placeholder
        icon_label = QLabel("🖼️")
        icon_label.setFont(QFont("Arial", 72))
        icon_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(icon_label)
        
        # Title
        title = QLabel("AI Image Workflow Processor")
        title.setFont(QFont("Arial", 24, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Description
        desc = QLabel(
            "Transform your images with AI-powered workflows.\n"
            "Describe what you want to do in natural language and let AI build\n"
            "and execute OpenCV processing pipelines for you."
        )
        desc.setFont(QFont("Arial", 12))
        desc.setAlignment(Qt.AlignCenter)
        layout.addWidget(desc)
        
        # Features list
        features = [
            "🤖 AI-Powered workflow generation",
            "🖼️ 21+ image processing operations",
            "📊 Visual workflow diagrams",
            "🎨 Professional desktop interface"
        ]
        
        for feature in features:
            feature_label = QLabel(feature)
            feature_label.setFont(QFont("Arial", 10))
            feature_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(feature_label)
        
        # Spacer
        layout.addStretch()
        
        # Open image button
        self.open_button = QPushButton("Open an Image")
        self.open_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.open_button.clicked.connect(self.open_image)
        layout.addWidget(self.open_button)
        
        # Spacer
        layout.addStretch()
        
        self.setLayout(layout)
        
        # Set background color
        self.setStyleSheet(f"""
            OnboardingScreen {{
                background-color: {DARK_THEME['bg_primary']};
                border-radius: 8px;
            }}
        """)
    
    def open_image(self):
        """Open image file"""
        if self.main_window and hasattr(self.main_window, 'open_image'):
            self.main_window.open_image()

class WorkflowGraphWidget(QWidget):
    """Widget for displaying workflow as a graph"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.workflow_data = None
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        self.setLayout(layout)
    
    def update_workflow(self, workflow_json):
        """Update the workflow graph visualization"""
        if not workflow_json:
            self.clear_graph()
            return
        
        self.workflow_data = workflow_json
        self.draw_workflow_graph()
    
    def clear_graph(self):
        """Clear the graph display"""
        self.figure.clear()
        self.canvas.draw()
    
    def draw_workflow_graph(self):
        """Draw the workflow as a directed graph"""
        if not self.workflow_data or not GRAPH_AVAILABLE:
            return
        
        # Clear previous graph
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        if ax is None:
            return
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add nodes
        nodes = self.workflow_data.get('nodes', [])
        for node in nodes:
            node_id = node['node_id']
            node_type = node['node_type']
            G.add_node(node_id, type=node_type)
        
        # Add edges
        connections = self.workflow_data.get('connections', [])
        for conn in connections:
            G.add_edge(conn['from_node'], conn['to_node'])
        
        # Define node colors based on type
        node_colors = []
        for node in list(G.nodes()):
            node_type = G.nodes[node]['type']
            if node_type == 'input':
                node_colors.append('#4CAF50')  # Green
            elif node_type == 'output':
                node_colors.append('#F44336')  # Red
            else:
                node_colors.append('#2196F3')  # Blue
        
        # Draw the graph
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=1000, ax=ax)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color='gray', 
                              arrows=True, arrowsize=20, ax=ax)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)
        
        # Set title and remove axes
        ax.set_title("Workflow Graph", fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Adjust layout and redraw
        self.figure.tight_layout()
        self.canvas.draw()

class WorkflowDrawer(QWidget):
    """Workflow side drawer"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.init_ui()
        self.workflow_thread = None
    
    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title = QLabel("AI Workflow")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        layout.addWidget(title)
        
        # Prompt input
        prompt_label = QLabel("Describe what you want to do:")
        prompt_label.setFont(QFont("Arial", 12))
        layout.addWidget(prompt_label)
        
        self.prompt_input = QTextEdit()
        self.prompt_input.setPlaceholderText(
            "Examples:\n"
            "• Convert to grayscale and blur the image\n"
            "• Add text 'Hello World' in white\n"
            "• Detect edges and apply threshold\n"
            "• Increase brightness and contrast"
        )
        self.prompt_input.setMaximumHeight(120)
        layout.addWidget(self.prompt_input)
        
        # Generate button
        self.generate_button = QPushButton("Generate Workflow")
        self.generate_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.generate_button.clicked.connect(self.generate_workflow)
        layout.addWidget(self.generate_button)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Create tab widget for workflow display
        self.workflow_tabs = QTabWidget()
        
        # Graph tab
        self.workflow_graph = WorkflowGraphWidget()
        self.workflow_tabs.addTab(self.workflow_graph, "Graph")
        
        # Text tab
        self.workflow_display = QTextBrowser()
        self.workflow_display.setStyleSheet(f"""
            QTextBrowser {{
                background-color: {DARK_THEME['bg_tertiary']};
                border: 1px solid {DARK_THEME['border']};
                border-radius: 4px;
                padding: 8px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 11px;
            }}
        """)
        self.workflow_tabs.addTab(self.workflow_display, "Text")
        
        layout.addWidget(self.workflow_tabs)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def generate_workflow(self):
        """Generate workflow from prompt"""
        prompt = self.prompt_input.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "Warning", "Please enter a processing request")
            return
        
        # Get API key from main window
        api_key = ""
        if self.main_window and hasattr(self.main_window, 'get_api_key'):
            api_key = self.main_window.get_api_key()
        if not api_key:
            QMessageBox.warning(self, "Warning", "Please set your OpenAI API key in Settings")
            return
        
        # Get current image from main window
        image = None
        if self.main_window and hasattr(self.main_window, 'get_current_image'):
            image = self.main_window.get_current_image()
        if image is None:
            QMessageBox.warning(self, "Warning", "Please load an image first")
            return
        
        # Start processing
        self.generate_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Create and start processing thread
        self.workflow_thread = WorkflowThread(image, prompt, api_key)
        self.workflow_thread.finished.connect(self.on_workflow_finished)
        self.workflow_thread.error.connect(self.on_workflow_error)
        self.workflow_thread.progress.connect(self.progress_bar.setValue)
        self.workflow_thread.start()
    
    def on_workflow_finished(self, result_image, workflow_json):
        """Handle workflow completion"""
        # Update workflow display
        self.update_workflow_display(workflow_json)
        
        # Send result to main window
        if self.main_window and hasattr(self.main_window, 'on_workflow_completed'):
            self.main_window.on_workflow_completed(result_image, workflow_json)
        
        # Reset UI
        self.generate_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        QMessageBox.information(self, "Success", "Workflow completed successfully!")
    
    def on_workflow_error(self, error_message):
        """Handle workflow error"""
        QMessageBox.critical(self, "Error", f"Workflow processing failed: {error_message}")
        
        # Reset UI
        self.generate_button.setEnabled(True)
        self.progress_bar.setVisible(False)
    
    def update_workflow_display(self, workflow_json):
        """Update the workflow display"""
        if not workflow_json:
            self.workflow_display.setText("No workflow data")
            self.workflow_graph.clear_graph()
            return
        
        # Update graph visualization
        self.workflow_graph.update_workflow(workflow_json)
        
        # Update text display
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

class SettingsDialog(QDialog):
    """Settings dialog for API key configuration"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.setFixedSize(400, 300)
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)
        
        # Title
        title = QLabel("Settings")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        layout.addWidget(title)
        
        # API Key section
        api_group = QGroupBox("OpenAI API Configuration")
        api_layout = QVBoxLayout()
        
        api_label = QLabel("OpenAI API Key:")
        api_label.setFont(QFont("Arial", 12))
        api_layout.addWidget(api_label)
        
        self.api_key_input = QLineEdit()
        self.api_key_input.setPlaceholderText("Enter your OpenAI API key...")
        self.api_key_input.setEchoMode(QLineEdit.Password)
        api_layout.addWidget(self.api_key_input)
        
        # Load from environment button
        env_button = QPushButton("Load from Environment")
        env_button.clicked.connect(self.load_from_env)
        api_layout.addWidget(env_button)
        
        # Load from .env file button
        env_file_button = QPushButton("Load from .env file")
        env_file_button.clicked.connect(self.load_from_env_file)
        api_layout.addWidget(env_file_button)
        
        api_group.setLayout(api_layout)
        layout.addWidget(api_group)
        
        # Save button
        save_button = QPushButton("Save Settings")
        save_button.setFont(QFont("Arial", 12, QFont.Bold))
        save_button.clicked.connect(self.save_settings)
        layout.addWidget(save_button)
        
        layout.addStretch()
        self.setLayout(layout)
        
        # Load current settings
        self.load_current_settings()
    
    def load_current_settings(self):
        """Load current settings"""
        import os
        api_key = os.getenv("OPENAI_API_KEY", "")
        self.api_key_input.setText(api_key)
    
    def load_from_env(self):
        """Load API key from environment variable"""
        import os
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.api_key_input.setText(api_key)
            QMessageBox.information(self, "Success", "API key loaded from environment")
        else:
            QMessageBox.warning(self, "Warning", "No API key found in environment")
    
    def load_from_env_file(self):
        """Load API key from .env file"""
        import os
        try:
            if os.path.exists(".env"):
                with open(".env", "r") as f:
                    for line in f:
                        if line.startswith("OPENAI_API_KEY="):
                            api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                            self.api_key_input.setText(api_key)
                            QMessageBox.information(self, "Success", "API key loaded from .env file")
                            return
                
                QMessageBox.warning(self, "Warning", "No OPENAI_API_KEY found in .env file")
            else:
                QMessageBox.warning(self, "Warning", "No .env file found")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load .env file: {str(e)}")
    
    def save_settings(self):
        """Save settings"""
        import os
        api_key = self.api_key_input.text().strip()
        
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            QMessageBox.information(self, "Success", "Settings saved successfully")
            self.accept()
        else:
            QMessageBox.warning(self, "Warning", "Please enter an API key") 