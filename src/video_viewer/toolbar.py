from PyQt6.QtWidgets import (
    QApplication, QLabel, QWidget, QPushButton, QHBoxLayout, QVBoxLayout,
    QFileDialog, QSplitter, QSizePolicy
)
from PyQt6.QtCore import QTimer, Qt, QPointF
from PyQt6.QtGui import QImage, QPixmap

# internal imports
from video_viewer.project_model import ProjectModel
from video_viewer.qtroles import PATH_ROLE, ITEM_TYPE_ROLE, DATA_ROLE
from video_viewer.ml_modules.patch_pca import PatchPCA

class Toolbar(QWidget):
    def __init__(self, project_model: ProjectModel,parent=None):
        super().__init__(parent)
        # Initialization code for Toolbar

        self.model = project_model

        layout = QHBoxLayout()
        self.setLayout(layout)

        # Add toolbar buttons
        self.temp_button = QPushButton("Run Classifier")
        layout.addWidget(self.temp_button)

        self.temp_button.clicked.connect(self.predict)

    def get_distortion_boxes(self):
        """
        Return a list of all distortion boxes in the project.
        """
        distortions = []

        def _process(item):
            t = item.data(1, ITEM_TYPE_ROLE)
            if t == "box":
                distortions.append(item.data(2, DATA_ROLE))
                
            else:
                for i in range(item.childCount()):
                    _process(item.child(i))

        for i in range(self.model.file_tree.topLevelItemCount()):
            _process(self.model.file_tree.topLevelItem(i))

        return distortions
    
    def predict(self):
        boxes = self.get_distortion_boxes()

        ml_model = PatchPCA(config={
            "xgboost": {
                "n_estimators": 600,
                "max_depth": 4,
                "learning_rate": 0.05,
                "subsample": 0.9,
                "colsample_bytree": 0.8,
                "reg_lambda": 1.0,
                "reg_alpha": 0.0,
                "eval_metric": "logloss",
                "tree_method": "hist",
                "early_stopping_rounds": 50
            }
        })

        ml_model.set_boxes(boxes)
        ml_model.clear_predicted_labels()
        ml_model.split_boxes(train_ratio=0.7)
        ml_model.train()
        ml_model.predict()

        print("Prediction complete.")

    