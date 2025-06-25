# Dark theme configuration for the desktop app

# Dark theme colors
DARK_THEME = {
    'bg_primary': '#1e1e1e',
    'bg_secondary': '#252526',
    'bg_tertiary': '#2d2d30',
    'text_primary': '#cccccc',
    'text_secondary': '#969696',
    'accent': '#007acc',
    'accent_hover': '#005a9e',
    'border': '#3e3e42',
    'success': '#4caf50',
    'warning': '#ff9800',
    'error': '#f44336'
}

def apply_dark_theme(app):
    """Apply dark theme to the application"""
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtGui import QPalette, QColor
    
    app.setStyle('Fusion')
    
    # Create dark palette
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(DARK_THEME['bg_primary']))
    palette.setColor(QPalette.WindowText, QColor(DARK_THEME['text_primary']))
    palette.setColor(QPalette.Base, QColor(DARK_THEME['bg_secondary']))
    palette.setColor(QPalette.AlternateBase, QColor(DARK_THEME['bg_tertiary']))
    palette.setColor(QPalette.ToolTipBase, QColor(DARK_THEME['bg_primary']))
    palette.setColor(QPalette.ToolTipText, QColor(DARK_THEME['text_primary']))
    palette.setColor(QPalette.Text, QColor(DARK_THEME['text_primary']))
    palette.setColor(QPalette.Button, QColor(DARK_THEME['bg_secondary']))
    palette.setColor(QPalette.ButtonText, QColor(DARK_THEME['text_primary']))
    palette.setColor(QPalette.BrightText, QColor(DARK_THEME['text_primary']))
    palette.setColor(QPalette.Link, QColor(DARK_THEME['accent']))
    palette.setColor(QPalette.Highlight, QColor(DARK_THEME['accent']))
    palette.setColor(QPalette.HighlightedText, QColor(DARK_THEME['text_primary']))
    
    app.setPalette(palette)
    
    # Apply stylesheet
    app.setStyleSheet(f"""
        QMainWindow {{
            background-color: {DARK_THEME['bg_primary']};
            color: {DARK_THEME['text_primary']};
        }}
        
        QMenuBar {{
            background-color: {DARK_THEME['bg_secondary']};
            color: {DARK_THEME['text_primary']};
            border-bottom: 1px solid {DARK_THEME['border']};
        }}
        
        QMenuBar::item {{
            background-color: transparent;
            padding: 8px 12px;
        }}
        
        QMenuBar::item:selected {{
            background-color: {DARK_THEME['accent']};
        }}
        
        QMenu {{
            background-color: {DARK_THEME['bg_secondary']};
            color: {DARK_THEME['text_primary']};
            border: 1px solid {DARK_THEME['border']};
        }}
        
        QMenu::item {{
            padding: 8px 20px;
        }}
        
        QMenu::item:selected {{
            background-color: {DARK_THEME['accent']};
        }}
        
        QStatusBar {{
            background-color: {DARK_THEME['bg_secondary']};
            color: {DARK_THEME['text_primary']};
            border-top: 1px solid {DARK_THEME['border']};
        }}
        
        QPushButton {{
            background-color: {DARK_THEME['bg_secondary']};
            color: {DARK_THEME['text_primary']};
            border: 1px solid {DARK_THEME['border']};
            border-radius: 4px;
            padding: 8px 16px;
            font-weight: bold;
        }}
        
        QPushButton:hover {{
            background-color: {DARK_THEME['accent_hover']};
            border-color: {DARK_THEME['accent']};
        }}
        
        QPushButton:pressed {{
            background-color: {DARK_THEME['accent']};
        }}
        
        QPushButton:disabled {{
            background-color: {DARK_THEME['bg_tertiary']};
            color: {DARK_THEME['text_secondary']};
            border-color: {DARK_THEME['border']};
        }}
        
        QLineEdit, QTextEdit {{
            background-color: {DARK_THEME['bg_tertiary']};
            color: {DARK_THEME['text_primary']};
            border: 1px solid {DARK_THEME['border']};
            border-radius: 4px;
            padding: 8px;
        }}
        
        QLineEdit:focus, QTextEdit:focus {{
            border-color: {DARK_THEME['accent']};
        }}
        
        QTabWidget::pane {{
            border: 1px solid {DARK_THEME['border']};
            background-color: {DARK_THEME['bg_primary']};
        }}
        
        QTabBar::tab {{
            background-color: {DARK_THEME['bg_secondary']};
            color: {DARK_THEME['text_primary']};
            padding: 8px 16px;
            border: 1px solid {DARK_THEME['border']};
            border-bottom: none;
        }}
        
        QTabBar::tab:selected {{
            background-color: {DARK_THEME['bg_primary']};
            border-bottom: 2px solid {DARK_THEME['accent']};
        }}
        
        QProgressBar {{
            border: 1px solid {DARK_THEME['border']};
            border-radius: 4px;
            text-align: center;
            background-color: {DARK_THEME['bg_tertiary']};
        }}
        
        QProgressBar::chunk {{
            background-color: {DARK_THEME['accent']};
            border-radius: 3px;
        }}
        
        QGroupBox {{
            font-weight: bold;
            border: 1px solid {DARK_THEME['border']};
            border-radius: 4px;
            margin-top: 1ex;
            padding-top: 10px;
        }}
        
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
        }}
        
        QScrollBar:vertical {{
            background-color: {DARK_THEME['bg_secondary']};
            width: 12px;
            border-radius: 6px;
        }}
        
        QScrollBar::handle:vertical {{
            background-color: {DARK_THEME['border']};
            border-radius: 6px;
            min-height: 20px;
        }}
        
        QScrollBar::handle:vertical:hover {{
            background-color: {DARK_THEME['accent']};
        }}
    """) 