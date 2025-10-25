from PyQt6.QtWidgets import QLabel
from PyQt6.QtCore import Qt

class ClickableLabel(QLabel):
    def __init__(self, text="", parent=None):
        super().__init__(text=text, parent=parent)
        self.viewer = parent  # reference to VideoViewer

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.viewer.start_drag(event.position())

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton:
            self.viewer.update_drag(event.position())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.viewer.end_drag()