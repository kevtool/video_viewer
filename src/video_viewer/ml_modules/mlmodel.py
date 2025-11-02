from typing import List
from abc import ABC, abstractmethod

from video_viewer.distortion import Distortion

class MLModel(ABC):
    def __init__(self):
        self.boxes: List[Distortion] = []

    @abstractmethod
    def train(self):
        """Train the model using self.boxes."""
        pass

    @abstractmethod
    def predict(self):
        """Make predictions using the trained model and self.boxes."""
        pass


    def split_boxes(self, train_ratio: float, val_ratio: float | None = None, test=False):
        """Split boxes into training, validation, and test sets."""
        total_boxes = len(self.boxes)
        train_end = int(total_boxes * train_ratio)

        scramble_boxes = self.boxes.copy()
        import random
        random.shuffle(scramble_boxes)

        self.train_boxes = scramble_boxes[:train_end]
        self.train_boxes.sort(key=lambda distortion: distortion.video_path)

        if test:
            if val_ratio is None:
                raise ValueError("val_ratio must be provided when test=True.")
            val_end = train_end + int(total_boxes * val_ratio)

            self.val_boxes = scramble_boxes[train_end:val_end]
            self.val_boxes.sort(key=lambda distortion: distortion.video_path)

            self.test_boxes = scramble_boxes[val_end:]
            self.test_boxes.sort(key=lambda distortion: distortion.video_path)
        else:
            self.val_boxes = scramble_boxes[train_end:]
            self.val_boxes.sort(key=lambda distortion: distortion.video_path)

            self.test_boxes = self.val_boxes

    def set_boxes(self, boxes: List[Distortion]):
        self.boxes = boxes

    def get_boxes(self):
        return self.boxes
    
    def clear_boxes(self):
        self.boxes.clear()
    
    def add_box(self, box: Distortion):
        self.boxes.append(box)

    def remove_box(self, box: Distortion):
        self.boxes.remove(box)
