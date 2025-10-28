from PyQt6.QtWidgets import (
    QApplication, QLabel, QWidget, QPushButton, QHBoxLayout, QVBoxLayout,
    QFileDialog, QSplitter, QSizePolicy
)
from PyQt6.QtCore import QTimer, Qt, QPointF
from PyQt6.QtGui import QImage, QPixmap
from xgboost import XGBClassifier


# internal imports
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
        self.temp_button = QPushButton("Hist Features")
        layout.addWidget(self.temp_button)

        self.temp_button.clicked.connect(self.get_rgb_data)

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

    def get_rgb_data(self):
        # Logic to get RGB data from the video

        distortions = self.get_distortion_boxes()
        for box in distortions:
            print(box)

        X, y = get_hist_features(distortions)

    def evaluate(self, X_train, y_train, X_val, y_val):


        pos = max(1, y_train.sum())
        neg = max(1, (y_train == 0).sum())
        spw = neg / pos

        # XGBoost classifier
        xgb_params = config['xgboost']
        clf = XGBClassifier(
            **xgb_params,
            scale_pos_weight=spw,
            # eval_metric=xgb_params.get("eval_metric", "logloss"),
            # early_stopping_rounds=xgb_params.get("early_stopping_rounds", 50)
        )

        # Train with early stopping
        clf.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # Evaluation
        probs = clf.predict_proba(X_val)[:, 1]
        preds = (probs >= 0.5).astype(int)

    