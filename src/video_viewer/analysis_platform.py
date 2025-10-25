import sys
from PyQt6.QtWidgets import (
    QWidget, QApplication, QHBoxLayout, QVBoxLayout, QSplitter
)
from PyQt6.QtCore import Qt
from video_viewer.file_manager import FileManager
from video_viewer.video_viewer import VideoViewer
from video_viewer.project_model import ProjectModel

class AnalysisPlatform(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Initialization code for AnalysisPlatform
        self.model = ProjectModel()

        self.file_manager = FileManager(project_model=self.model, parent=self)
        self.video_viewer = VideoViewer(project_model=self.model, parent=self)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(self.file_manager)
        self.splitter.addWidget(self.video_viewer)
        self.splitter.setSizes([300, 700])

        layout = QHBoxLayout()
        layout.addWidget(self.splitter)
        self.setLayout(layout)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = AnalysisPlatform()
    viewer.resize(1800, 900)
    viewer.show()
    sys.exit(app.exec())