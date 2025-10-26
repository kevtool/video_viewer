from PyQt6.QtWidgets import (
    QApplication, QLabel, QWidget, QPushButton, QHBoxLayout, QVBoxLayout,
    QFileDialog, QSplitter, QSizePolicy
)
from PyQt6.QtCore import QTimer, Qt, QPointF
from PyQt6.QtGui import QImage, QPixmap


# internal imports
from video_viewer.clickable_label import ClickableLabel
from video_viewer.box import DistortionBox, Zoombox
from video_viewer.project_model import ProjectModel
from video_viewer.qtroles import PATH_ROLE, ITEM_TYPE_ROLE, DATA_ROLE
from video_viewer.cov_pca import get_hist_features

class Toolbar(QWidget):
    def __init__(self, project_model: ProjectModel,parent=None):
        super().__init__(parent)
        # Initialization code for Toolbar

        self.model = project_model

        layout = QHBoxLayout()
        self.setLayout(layout)

        # Add toolbar buttons
        self.temp_button = QPushButton("Get RGB Data from Distortion Boxes")
        layout.addWidget(self.temp_button)

        self.temp_button.clicked.connect(self.get_rgb_data)

    def get_videos_with_boxes(self):
        """
        Return a dict mapping each video's path (or name if no path) to a list of boxes.
        Each box is returned as {'name': <label>, 'data': <DATA_ROLE dict>}.
        """
        videos = {}

        def _process(item):
            t = item.data(1, ITEM_TYPE_ROLE)
            if t == "video":
                path = item.data(0, PATH_ROLE) or item.text(0)
                boxes = []
                for i in range(item.childCount()):
                    child = item.child(i)
                    if child.data(1, ITEM_TYPE_ROLE) == "box":
                        boxes.append({
                            "name": child.text(0),
                            "data": child.data(2, DATA_ROLE)
                        })
                videos[path] = boxes
            elif t == "folder":
                for i in range(item.childCount()):
                    _process(item.child(i))

        for i in range(self.model.file_tree.topLevelItemCount()):
            _process(self.model.file_tree.topLevelItem(i))

        return videos

    def get_rgb_data(self):
        # Logic to get RGB data from the video

        videos = self.get_videos_with_boxes()
        for video, boxes in videos.items():
            print(f"Video: {video}")

            rois = []
            for box in boxes:
                print(f"  Box: {box['name']}, Data: {box['data']}")

                roi = (box['data']['x'], box['data']['y'], box['data']['width'])
                rois.append(roi)

            features = get_hist_features(video_path=video, rois=rois)

    