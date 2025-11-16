from xgboost import XGBClassifier
import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import mlflow

from video_viewer.ml_modules.mlmodel import MLModel
from video_viewer.video_utils import *

class PatchPCA(MLModel):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = None

        self.set_parameters()

    def set_parameters(self):
        self.frame_num = [0, 1, 2]
        self.n_components = 32
        self.patch_size = 7

    
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

        mlflow.set_experiment('PatchPCA')
        with mlflow.start_run():
            mlflow.log_param("train_samples", len(self.train_boxes))
            mlflow.log_param("val_samples", len(self.val_boxes))
            mlflow.log_param("frames_used", len(self.frame_num))
            mlflow.log_param("frame_num", self.frame_num)
            mlflow.log_param("n_components", self.n_components)
            mlflow.log_param("patch_size", self.patch_size)

            X_test, y_test = self.extract_features(mode='val')
            probs = self.model.predict_proba(X_test)[:, 1]
            preds = (probs >= 0.5).astype(int)

            # Metrics
            acc = accuracy_score(y_test, preds)
            try:
                auc = roc_auc_score(y_test, probs)
            except ValueError:
                auc = float('nan')
            cm = confusion_matrix(y_test, preds)
            report = classification_report(y_test, preds, output_dict=True, digits=4)

            print("Accuracy:", acc)
            print("ROC-AUC:", auc if not np.isnan(auc) else "ROC-AUC: not defined (only one class present in y_test)")
            print("\nConfusion matrix:\n", cm)
            print("\nClassification report:\n", classification_report(y_test, preds, digits=4))

            mlflow.log_metric("accuracy", acc)
            if not np.isnan(auc):
                mlflow.log_metric("roc_auc", auc)
            else:
                print("ROC-AUC undefined (only one class in validation set). Skipping logging.")
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
            else:
                tn = fp = fn = tp = 0
            mlflow.log_metric("true_positives", tp)
            mlflow.log_metric("false_positives", fp)
            mlflow.log_metric("true_negatives", tn)
            mlflow.log_metric("false_negatives", fn)
            if '1' in report:
                mlflow.log_metric("precision_class_1", report['1']['precision'])
                mlflow.log_metric("recall_class_1", report['1']['recall'])
                mlflow.log_metric("f1_class_1", report['1']['f1-score'])
            else:
                print("Warning: Class '1' not present in validation set; skipping class-1 metrics.")


    # def extract_features(self, mode='train'):
    #     if mode == 'train':
    #         boxes = self.train_boxes
    #     elif mode == 'val':
    #         boxes = self.val_boxes
    #     elif mode == 'test':
    #         boxes = self.test_boxes
    #     else:
    #         raise ValueError("Mode must be 'train', 'val', or 'test'.")

    #     if not boxes:
    #         raise RuntimeError("No distortion boxes found.")

    #     N = len(boxes)

    #     residues = []
    #     labels = []
    #     curr_video_path = None
    #     frame_num = [0, 1]

    #     print(f"Found {N} boxes.")
    #     for i, box in enumerate(boxes):
    #         print(f"\rProcessing box {i+1} of {N}", end='', flush=True)

    #         video_path = box.video_path
    #         if video_path != curr_video_path:
    #             if curr_video_path is not None:
    #                 cap.release()
    #             curr_video_path = video_path
    #             cap = cv2.VideoCapture(video_path)

    #         frames_cropped = []
    #         for fnum in self.frame_num:
    #             cap.set(cv2.CAP_PROP_POS_FRAMES, fnum)
    #             ret, frame = cap.read()
    #             if ret:
    #                 frames_cropped.append(frame[box.y:box.y+box.size, box.x:box.x+box.size])
    #             else:
    #                 raise IOError(f"Cannot read frame {fnum} from video: {video_path}")
                
    #         frame_cropped1, frame_cropped2 = frames_cropped

    #         residue = compute_residue_from_frames(frame_cropped1, frame_cropped2)
    #         residues.append(residue)
    #         labels.append(box.ground_truth_label)

    #     if 'cap' in locals():
    #         cap.release()         
    #     print("\n")
        

    #     if mode == 'train':
    #         all_patches = []
    #         for img in residues:
    #             patches = extract_color_patches(img, self.patch_size).astype(np.float32) / 255.0
    #             all_patches.append(patches.reshape(-1, 3*self.patch_size*self.patch_size))
    #         all_patches = np.concatenate(all_patches, axis=0)
    #         self.pca_model = PCA(n_components=self.n_components, svd_solver="randomized", random_state=42)
    #         self.pca_model.fit(all_patches)
    #     elif not hasattr(self, 'pca_model'):
    #         raise ValueError("PCA model is not fitted. Train the model first.")

    #     features = []
    #     for i, img in enumerate(residues):
    #         print(f"\rProcessing box {i+1} of {N:<80}", end='', flush=True)
    #         H, W, _ = img.shape
    #         patches = extract_color_patches(img, self.patch_size).astype(np.float32) / 255.0
    #         feats = self.pca_model.transform(patches)  # (num_patches, n_components)
    #         feats = feats.reshape(H - (self.patch_size - 1), W - (self.patch_size - 1), self.pca_model.n_components_).astype(np.float32)
    #         mu = feats.mean(axis=(0, 1))
    #         sd = feats.std(axis=(0, 1))
    #         features.append(np.concatenate([mu, sd], axis=0))
    #     print("\n")

    #     return features, np.array(labels)

    def extract_features(self, mode='train'):
        """
        Extract concatenated PCA features from multiple residues computed from
        consecutive frames listed in self.frame_nums.

        For F frames, we produce (F-1) residues:
            residue[i] = frame[i+1] - frame[i]

        For each residue we:
            1. Extract sliding patches
            2. Project patches with PCA
            3. Compute mean & std over PCA features

        The final feature = concatenation of all residue features.
        """
        # ----------------------------------------------------------------------
        # Select list of boxes
        # ----------------------------------------------------------------------
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
        print(f"Found {N} boxes.")

        # ----------------------------------------------------------------------
        # Read all residues for all boxes
        # ----------------------------------------------------------------------
        all_residues_per_box = []   # list of list-of-residue-images
        labels = []

        curr_video_path = None
        cap = None

        F = len(self.frame_num)
        if F < 2:
            raise ValueError("You need at least 2 frame numbers to form a residue.")

        for i, box in enumerate(boxes):
            print(f"\rProcessing box {i+1} of {N}", end='', flush=True)

            # Open video if path changed
            if box.video_path != curr_video_path:
                if cap is not None:
                    cap.release()
                curr_video_path = box.video_path
                cap = cv2.VideoCapture(curr_video_path)

            # ----------------------------------------------------------
            # Load all frames specified in frame_nums
            # ----------------------------------------------------------
            frames = []
            for fnum in self.frame_num:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fnum)
                ret, frame = cap.read()
                if not ret:
                    raise IOError(f"Cannot read frame {fnum} from video: {curr_video_path}")

                crop = frame[box.y : box.y + box.size,
                            box.x : box.x + box.size]
                frames.append(crop)

            # ----------------------------------------------------------
            # Compute residues: frame[i+1] - frame[i]
            # ----------------------------------------------------------
            residues = []
            for j in range(F-1):
                r = compute_residue_from_frames(frames[j], frames[j+1])
                residues.append(r)

            all_residues_per_box.append(residues)
            labels.append(box.ground_truth_label)

        if cap is not None:
            cap.release()

        print("\n")

        # ----------------------------------------------------------------------
        # Train PCA (only using train residues)
        # ----------------------------------------------------------------------
        if mode == 'train':
            all_patches = []

            print("Collecting patches for PCA training...")
            for residues in all_residues_per_box:  # list of residues for one box
                for img in residues:
                    patches = extract_color_patches(img, self.patch_size).astype(np.float32) / 255.0
                    patches = patches.reshape(-1, 3 * self.patch_size * self.patch_size)
                    all_patches.append(patches)

            all_patches = np.concatenate(all_patches, axis=0)

            self.pca_model = PCA(
                n_components=self.n_components,
                svd_solver="randomized",
                random_state=42
            )
            self.pca_model.fit(all_patches)

        elif not hasattr(self, 'pca_model'):
            raise ValueError("PCA model is not fitted. Train the model first.")

        # ----------------------------------------------------------------------
        # Extract PCA features for each residue for each box
        # ----------------------------------------------------------------------
        all_features = []

        print("Extracting PCA features...")
        for i, residues in enumerate(all_residues_per_box):
            print(f"\rProcessing box {i+1} of {N:<50}", end='', flush=True)

            residue_features = []

            for img in residues:
                H, W, _ = img.shape

                patches = extract_color_patches(img, self.patch_size).astype(np.float32) / 255.0
                feats = self.pca_model.transform(patches)

                # reshape back to spatial feature map
                h2 = H - (self.patch_size - 1)
                w2 = W - (self.patch_size - 1)
                feats = feats.reshape(h2, w2, self.pca_model.n_components_)

                mu = feats.mean(axis=(0, 1))
                sd = feats.std(axis=(0, 1))

                residue_features.append(np.concatenate([mu, sd], axis=0))

            # Concatenate all residues to one feature vector
            box_feature = np.concatenate(residue_features, axis=0)
            all_features.append(box_feature)

        print("\n")

        return np.array(all_features), np.array(labels)
            
