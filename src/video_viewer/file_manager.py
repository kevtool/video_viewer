from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QTreeWidget,
    QTreeWidgetItem, QApplication, QInputDialog, QMenu, QLabel
)
from PyQt6.QtCore import Qt, QPoint
import sys, os, json

from video_viewer.project_model import ProjectModel
from video_viewer.qtroles import PATH_ROLE, ITEM_TYPE_ROLE, DATA_ROLE
from video_viewer.distortion import Video, Distortion


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (Video, Distortion)):
            return obj.to_dict()
        return super().default(obj)

class FileSystemWidget(QTreeWidget):
    """A QTreeWidget subclass that validates drag-and-drop rules."""

    def dropEvent(self, event):
        # Check proposed drop target
        target_item = self.itemAt(event.position().toPoint())
        selected_items = self.selectedItems()

        for item in selected_items:
            item_type = item.data(1, ITEM_TYPE_ROLE)
            target_type = target_item.data(1, ITEM_TYPE_ROLE) if target_item else None

            # Folder rules
            if item_type == "folder":
                if target_item is not None and target_type != "folder":
                    event.ignore()
                    return

            # Video rules
            elif item_type == "video":
                if target_item is not None and target_type != "folder":
                    event.ignore()
                    return

            # Subitem / box rules
            elif item_type == "box":
                if target_item is None or target_type != "video":
                    event.ignore()
                    return

        super().dropEvent(event) 

class FileManager(QWidget):
    def __init__(self, project_model=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("File Manager")
        self.resize(600, 400)

        self.model = project_model

        self.model.video_loaded.connect(self.on_video_loaded)
        self.model.video_cleared.connect(self.on_video_cleared)

        # Layout setup
        self.layout = QVBoxLayout(self)
        self.status_label = QLabel("No files loaded.")
        self.buttons_layout = QHBoxLayout()
        self.layout.addLayout(self.buttons_layout)
        self.layout.addWidget(self.status_label)


        # Buttons
        self.upload_files_btn = QPushButton("Upload File(s)")
        self.upload_files_btn.clicked.connect(self.upload_files)
        self.buttons_layout.addWidget(self.upload_files_btn)

        self.add_folder_btn = QPushButton("Add Folder")
        self.add_folder_btn.clicked.connect(self.add_folder)
        self.buttons_layout.addWidget(self.add_folder_btn)

        self.save_btn = QPushButton("Save Project")
        self.save_btn.clicked.connect(self.save_project)
        self.buttons_layout.addWidget(self.save_btn)

        self.save_as_btn = QPushButton("Save As")
        self.save_as_btn.clicked.connect(self.save_new_project)
        self.buttons_layout.addWidget(self.save_as_btn)

        self.load_project_btn = QPushButton("Load Project")
        self.load_project_btn.clicked.connect(self.load_project)
        self.buttons_layout.addWidget(self.load_project_btn)

        # project path
        self.project_path = None

        # File tree
        self.file_tree = FileSystemWidget()
        self.file_tree.setHeaderLabel("Files")
        self.file_tree.setDragDropMode(QTreeWidget.DragDropMode.InternalMove)
        self.file_tree.setSelectionMode(QTreeWidget.SelectionMode.ExtendedSelection)
        self.file_tree.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.file_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.file_tree.itemDoubleClicked.connect(self.on_item_activated)
        self.file_tree.customContextMenuRequested.connect(self.show_context_menu)
        self.layout.addWidget(self.file_tree)

        # Copy
        self.copied_items = None

    # ------------------ File & Folder Management ------------------

    def upload_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select File(s)")
        if files:
            count = 0
            for file_path in files:
                if file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    count += 1
                    file_item = QTreeWidgetItem([os.path.abspath(file_path)])
                    file_item.setData(0, PATH_ROLE, file_path)  # Store full path
                    file_item.setData(1, ITEM_TYPE_ROLE, "video")  # Mark as video
                    file_item.setData(2, DATA_ROLE, Video(
                            video_path=file_path,
                            setting=None,  # Default setting
                        )
                    )
                    self.file_tree.addTopLevelItem(file_item)

            if count == 0:
                self.status_label.setText("No valid video files selected.")
            elif count == 1:
                self.status_label.setText("Uploaded 1 video.")
            else:
                self.status_label.setText(f"Uploaded {count} videos.")

    def add_folder(self):
        folder_name, ok = QInputDialog.getText(self, "Folder Name", "Enter folder name:")
        if ok and folder_name:
            folder_item = QTreeWidgetItem([folder_name])
            folder_item.setData(1, ITEM_TYPE_ROLE, "folder")
            folder_item.setChildIndicatorPolicy(QTreeWidgetItem.ChildIndicatorPolicy.ShowIndicator)
            self.file_tree.addTopLevelItem(folder_item)


    # ------------------ Distortion Box ------------------

    def add_distortion_box(self, parent_item, name=None, x=0, y=0, size=128, specify_size=False):
        if specify_size:
            size, ok = QInputDialog.getInt(self, "Box Size", "Enter box size:", value=128, min=1)
            if not ok:
                return

        item_type = parent_item.data(1, ITEM_TYPE_ROLE)
        if item_type != "video":
            return  # Only add boxes to videos

        if name is None:
            name, ok = QInputDialog.getText(self, "Box Name", "Enter box label:")
            if not ok or not name:
                return

        box_item = QTreeWidgetItem([name])
        box_item.setData(1, ITEM_TYPE_ROLE, "box")
        box_item.setData(2, DATA_ROLE, Distortion(
            name=name,
            video_path=parent_item.data(0, PATH_ROLE),
            setting=parent_item.data(2, DATA_ROLE).setting,
            x=x,
            y=y,
            size=size
        ))
        parent_item.addChild(box_item)
        parent_item.setExpanded(True)


        if self.model and self.model.active_video == parent_item:
            self.model.add_box_to_active_video(box_item)

    def copy_distortion_boxes(self):
        self.copied_items = []
        selected_items = self.file_tree.selectedItems()
        for item in selected_items:
            item_type = item.data(1, ITEM_TYPE_ROLE)
            if item_type == "box":
                self.copied_items.append((item.data(2, DATA_ROLE).name,
                                          item.data(2, DATA_ROLE).x,
                                          item.data(2, DATA_ROLE).y,
                                            item.data(2, DATA_ROLE).size))

        if not self.copied_items:
            self.status_label.setText("No boxes selected to copy.")

    # ------------------ Context Menu ------------------

    def show_context_menu(self, pos: QPoint):
        item = self.file_tree.itemAt(pos)
        if not item:
            return

        menu = QMenu(self)

        item_type = item.data(1, ITEM_TYPE_ROLE)
        if item_type == "video":
            menu.addAction("Add Distortion Box", lambda: self.add_distortion_box(item))
            menu.addAction("Add Sized Distortion Box", lambda: self.add_distortion_box(item, specify_size=True))
            menu.addAction("Edit Properties", lambda: self.edit_video_properties(item))

            if self.copied_items is not None:
                menu.addAction("Paste Box(es)", lambda: [
                    self.add_distortion_box(item, name=name, x=x, y=y, size=size) for name, x, y, size in self.copied_items
                ])
        else:
            # currently, only folders and boxes can be renamed
            menu.addAction("Rename", lambda: self.rename_item(item))

        if item_type == "box":
            menu.addAction("Copy Box(es)", self.copy_distortion_boxes)

        menu.addAction("Remove Item", lambda: self.remove_item(item))

        menu.exec(self.file_tree.viewport().mapToGlobal(pos))


    def edit_video_properties(self, item=None):
        # assert item.data(1, ITEM_TYPE_ROLE) == "video", "Item must be a video."

        selected_items = self.file_tree.selectedItems()
        selected_videos = [it for it in selected_items if it.data(1, ITEM_TYPE_ROLE) == "video"]

        if selected_videos:
            targets = selected_videos
        else:
            if item and item.data(1, ITEM_TYPE_ROLE) == "video":
                targets = [item]
            else:
                return


        ground_truth_label, ok = QInputDialog.getInt(
            self, "Edit Video Ground Truth", "Enter new ground truth label:",
            value=0, min=0
        )

        if not ok:
            return
        
        for item in targets:
            # data = item.data(2, DATA_ROLE)
            # data.ground_truth_label = ground_truth_label
            # item.setData(2, DATA_ROLE, data)

            for i in range(item.childCount()):
                child = item.child(i)
                box_data = child.data(2, DATA_ROLE)
                box_data.ground_truth_label = ground_truth_label
                child.setData(2, DATA_ROLE, box_data)

    # ------------------ Item Activation ------------------

    def on_item_activated(self, item):
        item_type = item.data(1, ITEM_TYPE_ROLE)
        if item_type == "video":
            video_item = item

            # Collect all boxes of this video
            boxes = []
            for i in range(video_item.childCount()):
                child = video_item.child(i)
                if child.data(1, ITEM_TYPE_ROLE) == "box":
                    boxes.append(child)

            # Inform the model
            if self.model:
                self.model.set_active_video(video_item, boxes)

        elif item_type == "box":
            # The box itself
            box_item = item

            # Its parent video
            video_item = box_item.parent()
            if not video_item or video_item.data(1, ITEM_TYPE_ROLE) != "video":
                return  # Safety check

            # Collect all boxes of this video
            boxes = []
            for i in range(video_item.childCount()):
                child = video_item.child(i)
                if child.data(1, ITEM_TYPE_ROLE) == "box":
                    boxes.append(child)

            # Inform the model
            if self.model:
                self.model.set_active_video(video_item, boxes, selected_box=box_item)

    # ------------------ Item Removal ------------------
    
    def collect_videos(self, parent_item):
        videos = []
        for i in range(parent_item.childCount()):
            child = parent_item.child(i)
            child_type = child.data(1, ITEM_TYPE_ROLE)
            if child_type == "video":
                videos.append(child)
            elif child_type == "folder":
                videos.extend(self.collect_videos(child))
        return videos
    
    def remove_item(self, item):
        parent = item.parent()
        if parent:
            parent.removeChild(item)
        else:
            self.file_tree.takeTopLevelItem(self.file_tree.indexOfTopLevelItem(item))

        item_type = item.data(1, ITEM_TYPE_ROLE)
        if item_type == "video":
            video_item = item

            # Inform the model
            if self.model.active_video == video_item:
                self.model.set_active_video(None, None)

        elif item_type == "box":
            box_item = item
            video_item = parent
            if not video_item or video_item.data(1, ITEM_TYPE_ROLE) != "video":
                return  # Safety check
            print("Box being removed was selected; clearing selection.")

            # Inform the model
            if self.model.active_video == video_item:
                # Collect all boxes of this video
                boxes = []
                for i in range(video_item.childCount()):
                    child = video_item.child(i)
                    if child.data(1, ITEM_TYPE_ROLE) == "box":
                        boxes.append(child)
                self.model.set_active_video(video_item, boxes, selected_box=None)

        elif item_type == "folder":
            # Collect all videos under this folder
            videos = self.collect_videos(item)
            for video in videos:
                if self.model.active_video == video:
                    self.model.set_active_video(None, None)

    # ------------------ Rename Item -------------------

    def rename_item(self, item):
        new_name, ok = QInputDialog.getText(self, "Rename Item", "Enter new name:")
        if ok and new_name:
            item.setText(0, new_name)

    # ------------------ Save Project ------------------

    def item_to_dict(self, item):
        item_dict = {
            "name": item.text(0),
            "type": item.data(1, ITEM_TYPE_ROLE),
            "path": item.data(0, PATH_ROLE),
            "data": item.data(2, DATA_ROLE),
            "children": []
        }

        for i in range(item.childCount()):
            child = item.child(i)
            item_dict["children"].append(self.item_to_dict(child))

        return item_dict
    
    def dict_to_item(self, data):
        item = QTreeWidgetItem([data["name"]])
        item.setData(1, ITEM_TYPE_ROLE, data["type"])
        if data.get("path"):
            item.setData(0, PATH_ROLE, data["path"])

        obj_data = data.get("data")
        if isinstance(obj_data, dict):
            if data["type"] == "video":
                item.setData(2, DATA_ROLE, Video(**obj_data))
            elif data["type"] == "box":
                item.setData(2, DATA_ROLE, Distortion(**obj_data))
            else:
                item.setData(2, DATA_ROLE, obj_data)

        for child_data in data.get("children", []):
            child_item = self.dict_to_item(child_data)
            item.addChild(child_item)
        
        return item
    
    def save_project(self):
        if not self.project_path:
            self.save_new_project()
            return
        
        project_data = []
        for i in range(self.file_tree.topLevelItemCount()):
            item = self.file_tree.topLevelItem(i)
            project_data.append(self.item_to_dict(item))

        with open(self.project_path, "w", encoding="utf-8") as f:
            json.dump(project_data, f, cls=EnhancedJSONEncoder,indent=4)

        self.status_label.setText(f"Project saved to {self.project_path}")
        self.project_path = self.project_path


    def save_new_project(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Project", "", "Project Files (*.json)")
        if not file_path:
            return

        project_data = []
        for i in range(self.file_tree.topLevelItemCount()):
            item = self.file_tree.topLevelItem(i)
            project_data.append(self.item_to_dict(item))

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(project_data, f, cls=EnhancedJSONEncoder,indent=4)

        self.status_label.setText(f"Project saved to {file_path}")
        self.project_path = file_path

    def load_project(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Project", "", "Project Files (*.json)")
        if not file_path:
            return

        with open(file_path, "r", encoding="utf-8") as f:
            project_data = json.load(f)

        self.file_tree.clear()

        for item_data in project_data:
            item = self.dict_to_item(item_data)
            self.file_tree.addTopLevelItem(item)

        self.status_label.setText(f"Project loaded from {file_path}")
        self.project_path = file_path

    # ------------------ Model Event Handlers ------------------
    def on_video_loaded(self, video_path):
        self.status_label.setText(f"Loaded video: {video_path}")

    def on_video_cleared(self):
        self.status_label.setText("Video cleared.")

# ------------------ Run App ------------------

if __name__ == "__main__":
    app = QApplication(sys.argv)
    fm = FileManager()
    fm.show()
    sys.exit(app.exec())