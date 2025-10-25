import sys, os, json
from PyQt6.QtWidgets import (
    QApplication, QWidget, QPushButton,
    QHBoxLayout, QVBoxLayout, QTreeWidget, QTreeWidgetItem,
    QMenu, QFileDialog, QMessageBox, QInputDialog
)
from PyQt6.QtCore import Qt

from video_viewer.project_model import ProjectModel


class FileManager(QWidget):
    def __init__(self, project_model: ProjectModel, parent=None):
        super().__init__(parent)
        self.setWindowTitle("File Manager")

        # Layout setup
        self.layout = QVBoxLayout(self)
        self.buttons_layout = QHBoxLayout()
        self.layout.addLayout(self.buttons_layout)

        # Buttons
        self.upload_files_btn = QPushButton("Upload File(s)")
        self.upload_files_btn.clicked.connect(self.upload_files)
        self.buttons_layout.addWidget(self.upload_files_btn)

        self.add_folder_btn = QPushButton("Add Folder")
        self.add_folder_btn.clicked.connect(self.add_folder)
        self.buttons_layout.addWidget(self.add_folder_btn)

        self.save_project_btn = QPushButton("Save Project")
        self.save_project_btn.clicked.connect(self.save_project)
        self.buttons_layout.addWidget(self.save_project_btn)

        self.load_project_btn = QPushButton("Load Project")
        self.load_project_btn.clicked.connect(self.load_project)
        self.buttons_layout.addWidget(self.load_project_btn)

        # Tree widget for files & subitems
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["File / Subitem"])
        self.layout.addWidget(self.tree)

        # Context menu
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self.show_context_menu)
        self.tree.itemDoubleClicked.connect(self.on_item_activated)

        self.model = project_model
        self.model.data_changed.connect(self.refresh_tree)

    def refresh_tree(self):
        self.tree.clear()
        for video in self.model.videos:
            video_item = QTreeWidgetItem([os.path.basename(video["path"])])
            video_item.setData(0, Qt.ItemDataRole.UserRole, video["path"])
            for box in video["boxes"]:
                subitem = QTreeWidgetItem([box.get("name", "Box")])
                subitem.setData(0, Qt.ItemDataRole.UserRole, box)
                video_item.addChild(subitem)
            self.tree.addTopLevelItem(video_item)
        
    def show_context_menu(self, pos):
        item = self.tree.itemAt(pos)
        menu = QMenu(self)

        if item:
            add_subitem_action = menu.addAction("Add Subitem")
            remove_action = menu.addAction("Remove Item")
            chosen = menu.exec(self.tree.mapToGlobal(pos))

            if chosen == add_subitem_action:
                self.add_subitem(item)
            elif chosen == remove_action:
                self.remove_item(item)
        else:
            # right-clicked empty area
            add_file_action = menu.addAction("Add File(s)")
            chosen = menu.exec(self.tree.mapToGlobal(pos))
            if chosen == add_file_action:
                self.upload_files()

    def upload_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select File(s)")
        for path in files:
            self.model.add_video(path)

    def add_folder(self):
        folder_name, ok = QInputDialog.getText(self, "New Folder", "Enter folder name:")
        if ok and folder_name:
            folder_item = QTreeWidgetItem([folder_name])
            # folder_item.setIcon(0, self.icon_folder)
            self.tree.addTopLevelItem(folder_item)

    def add_subitem(self, parent_item):
        name, ok = QInputDialog.getText(self, "Box Name", "Enter box label:")
        if not ok or not name:
            return

        # find which video this belongs to
        video_path = parent_item.data(0, Qt.ItemDataRole.UserRole)
        video_index = next(
            (i for i, v in enumerate(self.model.videos) if v["path"] == video_path),
            None
        )
        if video_index is None:
            return

        box_data = {"name": name, "x": 0, "y": 0, "width": 128, "height": 128}
        self.model.add_box(video_index, box_data)

    def remove_item(self, item):
        parent = item.parent()
        if parent:
            # subitem (box)
            video_path = parent.data(0, Qt.ItemDataRole.UserRole)
            video_index = next(
                (i for i, v in enumerate(self.model.videos) if v["path"] == video_path),
                None
            )
            box_index = parent.indexOfChild(item)
            self.model.remove_box(video_index, box_index)
        else:
            # top-level video
            index = self.tree.indexOfTopLevelItem(item)
            del self.model.videos[index]
            self.model.data_changed.emit()

    def on_item_activated(self, item, column):
        path = item.data(0, Qt.ItemDataRole.UserRole)
        # check if item data is a string (not a dict like DistortionBox is stored) 
        if isinstance(path, str):
            _, ext = os.path.splitext(path or "")
            if ext.lower() in (".mp4", ".avi", ".mov", ".mkv"):
                video_data = next((v for v in self.model.videos if v["path"] == path), None)
                if video_data:
                    self.model.video_selected.emit(video_data)
        
    def save_project(self):
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Project", "", "JSON Files (*.json)")
        if not file_name:
            return
        with open(file_name, "w") as f:
            json.dump({"videos": self.model.videos}, f, indent=2)

    def load_project(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Project", "", "JSON Files (*.json)")
        if not file_name:
            return
        with open(file_name, "r") as f:
            data = json.load(f)
        self.model.videos = data.get("videos", [])
        self.model.data_changed.emit()