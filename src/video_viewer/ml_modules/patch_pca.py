from xgboost import XGBClassifier
import cv2
import numpy as np
from sklearn.decomposition import PCA

from video_viewer.ml_modules.mlmodel import MLModel
from video_viewer.video_utils import *

class PatchPCA(MLModel):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = None

    
    def train(self):
        # TODO: Implement feature extraction logic
        X_train, y_train = self.extract_features(mode='train')  # Extracted features and labels for training
        X_val, y_val = self.extract_features(mode='val')      # Extracted features and labels for validation

        pos = max(1, y_train.sum())
        neg = max(1, (y_train == 0).sum())
        spw = neg / pos

        # XGBoost classifier
        xgb_params = self.config['xgboost']
        self.model = XGBClassifier(
            **xgb_params,
            scale_pos_weight=spw,
            # eval_metric=xgb_params.get("eval_metric", "logloss"),
            # early_stopping_rounds=xgb_params.get("early_stopping_rounds", 50)
        )

        # Train with early stopping
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        
    def predict(self):
        if not self.model:
            raise ValueError("Model is not trained yet.")

        X_test, y_test = self.extract_features(mode='val')
        probs = self.model.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)

        for box, pred in zip(self.test_boxes, preds):
            print(box.predicted_label, type(box.predicted_label))
            box.predicted_label = int(pred)


    def extract_features(self, mode='train'):
        if mode == 'train':
            boxes = self.train_boxes
        elif mode == 'val':
            boxes = self.val_boxes
        elif mode == 'test':
            boxes = self.test_boxes
        else:
            raise ValueError("Mode must be 'train', 'val', or 'test'.")

        if not boxes:
            raise RuntimeError("No distortion boxes found.")

        N = len(boxes)

        residues = []
        labels = []
        curr_video_path = None
        frame_num = [0, 1]

        print(f"Found {N} boxes.")
        for i, box in enumerate(boxes):
            print(box)
            print(f"\rProcessing box {i+1} of {N}", end='', flush=True)

            video_path = box.video_path
            if video_path != curr_video_path:
                if curr_video_path is not None:
                    cap.release()
                curr_video_path = video_path
                cap = cv2.VideoCapture(video_path)

            frames_cropped = []
            for fnum in frame_num:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fnum)
                ret, frame = cap.read()
                if ret:
                    frames_cropped.append(frame[box.y:box.y+box.size, box.x:box.x+box.size])
                else:
                    raise IOError(f"Cannot read frame {fnum} from video: {video_path}")
                
            frame_cropped1, frame_cropped2 = frames_cropped

            residue = compute_residue_from_frames(frame_cropped1, frame_cropped2)
            residues.append(residue)
            labels.append(box.ground_truth_label)

        if 'cap' in locals():
            cap.release()         
        print("\n")
        

        n_components = 32
        patch_size = 7

        if mode == 'train':
            all_patches = []
            for img in residues:
                patches = extract_color_patches(img, patch_size).astype(np.float32) / 255.0
                all_patches.append(patches.reshape(-1, 3*patch_size*patch_size))
            all_patches = np.concatenate(all_patches, axis=0)
            self.pca_model = PCA(n_components=n_components, svd_solver="randomized", random_state=42)
            self.pca_model.fit(all_patches)
        elif not hasattr(self, 'pca_model'):
            raise ValueError("PCA model is not fitted. Train the model first.")

        features = []
        for i, img in enumerate(residues):
            print(f"\rProcessing image {i+1} of {N:<80}", end='', flush=True)
            H, W, _ = img.shape
            patches = extract_color_patches(img, patch_size).astype(np.float32) / 255.0
            feats = self.pca_model.transform(patches)  # (num_patches, n_components)
            feats = feats.reshape(H - (patch_size - 1), W - (patch_size - 1), self.pca_model.n_components_).astype(np.float32)
            mu = feats.mean(axis=(0, 1))
            sd = feats.std(axis=(0, 1))
            features.append(np.concatenate([mu, sd], axis=0))

        return features, np.array(labels)
            
