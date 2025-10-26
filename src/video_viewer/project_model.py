from PyQt6.QtCore import QObject, pyqtSignal

class ProjectModel(QObject):
    data_changed = pyqtSignal()     # emitted when any data changes
    file_selected = pyqtSignal(dict)  # emitted when a video is selected (with boxes)

    def __init__(self):
        super().__init__()
        self.active_video = None
        self.boxes = []
        self.selected_box = None

    def set_active_video(self, video_item, boxes, selected_box=None):
        """Set the currently active video and its boxes."""
        a_flag = self.active_video is not video_item
        b_flag = self.boxes is not boxes
        s_flag = self.selected_box is not selected_box

        if video_item is None:
            self.active_video = None
            self.boxes = []
            self.selected_box = None
        else:
            self.active_video = video_item
            self.boxes = boxes
            self.selected_box = selected_box

        # Emit signal with video data
        video_data = {
            "active_video_changed": a_flag,
            "boxes_changed": b_flag,
            "selected_box_changed": s_flag,
        }
        self.file_selected.emit(video_data)
        # self.data_changed.emit()

    def add_box_to_active_video(self, box_item):
        """Add a box to the currently active video."""
        if self.active_video and box_item not in self.boxes:
            self.boxes.append(box_item)

            video_data = {
                "active_video_changed": False,
                "boxes_changed": True,
                "selected_box_changed": False,
            }
            self.file_selected.emit(video_data)