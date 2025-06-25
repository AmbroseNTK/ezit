# Main window for the desktop app

import sys
import os
import cv2
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QTabWidget, QMenuBar, QStatusBar, QAction,
                             QFileDialog, QMessageBox, QStackedWidget)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from .widgets import (OnboardingScreen, WorkflowDrawer, SettingsDialog, 
                     ImageViewer, WorkflowThread)
from .theme import DARK_THEME

class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.original_image = None
        self.result_image = None
        self.workflow_drawer_visible = False
        
    def init_ui(self):
        self.setWindowTitle("AI Image Workflow Processor")
        self.setGeometry(100, 100, 1400, 900)
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create status bar
        self.create_status_bar()
        
        # Create main content
        self.create_main_content()
    
    def create_menu_bar(self):
        """Create the menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("Files")
        
        open_action = QAction("Open Image", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_image)
        file_menu.addAction(open_action)
        
        export_action = QAction("Export Result", self)
        export_action.setShortcut("Ctrl+S")
        export_action.triggered.connect(self.export_result)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Workflow menu
        workflow_menu = menubar.addMenu("Workflow")
        
        toggle_drawer_action = QAction("Toggle Workflow Drawer", self)
        toggle_drawer_action.setShortcut("Ctrl+W")
        toggle_drawer_action.triggered.connect(self.toggle_workflow_drawer)
        workflow_menu.addAction(toggle_drawer_action)
        
        # Settings menu
        settings_menu = menubar.addMenu("Settings")
        
        settings_action = QAction("Settings", self)
        settings_action.triggered.connect(self.show_settings)
        settings_menu.addAction(settings_action)
    
    def create_status_bar(self):
        """Create the status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Image info
        self.image_info_label = QLabel("No image loaded")
        self.image_info_label.setFont(QFont("Arial", 10))
        self.status_bar.addWidget(self.image_info_label)
        
        # Cursor position
        self.cursor_pos_label = QLabel("")
        self.cursor_pos_label.setFont(QFont("Arial", 10))
        self.status_bar.addPermanentWidget(self.cursor_pos_label)
    
    def create_main_content(self):
        """Create the main content area"""
        # Create stacked widget for onboarding and edit screens
        self.stacked_widget = QStackedWidget()
        
        # Onboarding screen
        self.onboarding_screen = OnboardingScreen(self)
        self.stacked_widget.addWidget(self.onboarding_screen)
        
        # Edit screen
        self.edit_screen = self.create_edit_screen()
        self.stacked_widget.addWidget(self.edit_screen)
        
        # Set central widget
        self.setCentralWidget(self.stacked_widget)
        
        # Show onboarding screen initially
        self.stacked_widget.setCurrentIndex(0)
    
    def create_edit_screen(self):
        """Create the edit screen with image viewer and tabs"""
        widget = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Main content area
        main_content = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Tab widget for images
        self.tab_widget = QTabWidget()
        
        # Original image tab
        original_tab = QWidget()
        original_layout = QVBoxLayout()
        original_layout.setContentsMargins(0, 0, 0, 0)
        
        self.image_viewer = ImageViewer()
        self.image_viewer.mouseMoveEvent = self.on_image_mouse_move
        original_layout.addWidget(self.image_viewer)
        
        original_tab.setLayout(original_layout)
        self.tab_widget.addTab(original_tab, "Original")
        
        # Result image tab
        result_tab = QWidget()
        result_layout = QVBoxLayout()
        result_layout.setContentsMargins(0, 0, 0, 0)
        
        self.result_viewer = ImageViewer()
        result_layout.addWidget(self.result_viewer)
        
        result_tab.setLayout(result_layout)
        self.tab_widget.addTab(result_tab, "Result")
        
        main_layout.addWidget(self.tab_widget)
        main_content.setLayout(main_layout)
        
        # Workflow drawer
        self.workflow_drawer = WorkflowDrawer(self)
        self.workflow_drawer.setMaximumWidth(350)
        self.workflow_drawer.setVisible(False)
        
        # Add widgets to layout
        layout.addWidget(main_content)
        layout.addWidget(self.workflow_drawer)
        
        widget.setLayout(layout)
        return widget
    
    def open_image(self):
        """Open image file"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, 
                "Select Image", 
                "", 
                "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff *.tif);;All Files (*)"
            )
            
            if file_path and file_path.strip():
                # Load image using OpenCV
                image = cv2.imread(file_path)
                if image is not None:
                    self.original_image = image
                    self.image_viewer.set_image(image)
                    
                    # Update image info
                    height, width = image.shape[:2]
                    channels = image.shape[2] if len(image.shape) == 3 else 1
                    self.image_info_label.setText(f"Image: {width}x{height}, {channels} channel(s)")
                    
                    # Switch to edit screen
                    self.stacked_widget.setCurrentIndex(1)
                    
                    QMessageBox.information(self, "Success", f"Image loaded successfully: {file_path}")
                else:
                    raise Exception(f"OpenCV failed to load image from: {file_path}")
            else:
                # User cancelled the dialog
                return
                
        except Exception as e:
            error_msg = f"Failed to load image: {str(e)}"
            print(f"Error: {error_msg}")  # Debug print
            QMessageBox.critical(self, "Error", error_msg)
    
    def export_result(self):
        """Export result image"""
        if self.result_image is None:
            QMessageBox.warning(self, "Warning", "No result image to export")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Result Image", "", "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)"
        )
        
        if file_path:
            try:
                cv2.imwrite(file_path, self.result_image)
                QMessageBox.information(self, "Success", "Result image exported successfully")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export image: {str(e)}")
    
    def toggle_workflow_drawer(self):
        """Toggle workflow drawer visibility"""
        self.workflow_drawer_visible = not self.workflow_drawer_visible
        self.workflow_drawer.setVisible(self.workflow_drawer_visible)
    
    def show_settings(self):
        """Show settings dialog"""
        settings_dialog = SettingsDialog(self)
        settings_dialog.show()
    
    def get_api_key(self):
        """Get current API key"""
        return os.getenv("OPENAI_API_KEY", "")
    
    def get_current_image(self):
        """Get current image"""
        return self.original_image
    
    def on_workflow_completed(self, result_image, workflow_json):
        """Handle workflow completion"""
        self.result_image = result_image
        self.result_viewer.set_image(result_image)
        
        # Switch to result tab
        self.tab_widget.setCurrentIndex(1)
    
    def on_image_mouse_move(self, event):
        """Handle mouse move on image viewer"""
        if self.image_viewer.pixmap_item and self.original_image is not None:
            # Get scene position
            scene_pos = self.image_viewer.mapToScene(event.pos())
            
            # Get image coordinates
            if self.image_viewer.pixmap_item:
                item_pos = self.image_viewer.pixmap_item.mapFromScene(scene_pos)
                x, y = int(item_pos.x()), int(item_pos.y())
                
                # Check if within image bounds
                if 0 <= x < self.original_image.shape[1] and 0 <= y < self.original_image.shape[0]:
                    cursor_text = f"Cursor: ({x}, {y})"
                else:
                    cursor_text = "Cursor: Out of bounds"
                
                # Update cursor position if status bar is available
                if hasattr(self, 'cursor_pos_label') and self.cursor_pos_label:
                    self.cursor_pos_label.setText(cursor_text) 