from PyQt6.QtCore import QObject, pyqtSignal

class ProjectModel(QObject):
    data_changed = pyqtSignal()     # emitted when any data changes
    video_selected = pyqtSignal(dict)  # emitted when a video is selected (with boxes)

    def __init__(self):
        super().__init__()
        self.videos = []  # list of {"path": str, "boxes": [ {...} ]}

    def add_video(self, path):
        video = {"path": path, "boxes": []}
        self.videos.append(video)
        self.data_changed.emit()

    def add_box(self, video_index, box):
        self.videos[video_index]["boxes"].append(box)
        self.data_changed.emit()

    def remove_box(self, video_index, box_index):
        del self.videos[video_index]["boxes"][box_index]
        self.data_changed.emit()

    def update_box(self, video_index, box_index, new_data):
        self.videos[video_index]["boxes"][box_index].update(new_data)
        self.data_changed.emit()

    def clear(self):
        self.videos = []
        self.data_changed.emit()