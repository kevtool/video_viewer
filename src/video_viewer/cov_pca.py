import cv2
import numpy as np
from sklearn.decomposition import PCA

from video_viewer.distortion import Distortion

def apply_pca(roi, component=1):
    """
    Apply PCA to the ROI and return the specified principal component channel.
    
    roi: (H, W, 3) array
    component: 1, 2, or 3 (1 = first PC)
    
    Returns: (H, W) array of the specified PCA component
    """
    H, W, C = roi.shape
    assert C == 3, "Expected 3 channels"

    # Reshape to (N_pixels, 3)
    pixels = roi.reshape(-1, 3).astype(np.float32)

    # Fit PCA
    pca = PCA(n_components=3)
    pixels_pca = pca.fit_transform(pixels)  # Shape: (H*W, 3)

    # Get the specified component (1-based index)
    pc_channel = pixels_pca[:, component - 1]

    # Reshape back to (H, W)
    pc_image = pc_channel.reshape(H, W)

    return pc_image

def residue_to_histogram_feature(residue, n_bins=64):
    """
    Convert a single (H, W, 3) residue array into a (3*n_bins)-dimensional feature vector.
    
    Steps:
    1. Reshape to (H*W, 3) — each pixel is a 3D point in channel space.
    2. Compute covariance matrix (3x3) and perform PCA (equivalent to fitting PCA on pixels).
    3. Project all pixels onto the 3 PCs → (H*W, 3).
    4. Reshape back to (H, W, 3).
    5. Compute 64-bin histogram for each PC channel (clipped to [0, 255]).
    6. Concatenate → (3*n_bins,) vector.
    """
    H, W, C = residue.shape
    assert C == 3, "Expected 3 channels"

    # Step 1: Flatten to (N_pixels, 3)
    pixels = residue.reshape(-1, 3).astype(np.float32)

    # Step 2: Fit PCA on the pixel distribution (centered by default)
    pca = PCA(n_components=3)
    pixels_pca = pca.fit_transform(pixels)  # Shape: (H*W, 3)

    # Step 3: Reshape to (H, W, 3)
    pc_channels = pixels_pca.reshape(H, W, 3)

    # Step 4: Compute histogram for each PC channel
    # Note: PCA output is not bounded in [0,255]; we must choose a reasonable range.
    # Option: use percentiles or fixed range. Here we use global min/max per channel,
    # but for consistency across train/test, it's better to use a fixed range.
    # Since raw residues are in [0,255], PCA components are typically in [-255, 255].
    # We'll clip to [-255, 255] and shift to [0, 510] for histogramming.
    features = []
    for i in range(3):
        channel = pc_channels[:, :, i]
        # Clip to a reasonable range (e.g., 99th percentile or fixed)
        # For reproducibility across train/test, use fixed range:
        hist, _ = np.histogram(channel, bins=n_bins, range=(-255, 255), density=False)
        features.append(hist)
    
    return np.concatenate(features)  # Shape: (3*n_bins,)

def extract_histogram_features(residues, n_bins):
    """
    residues: list or object array of (H, W, 3) arrays
    Returns: np.ndarray of shape (N, 3*n_bins)
    """
    features = []
    for residue in residues:
        feat = residue_to_histogram_feature(residue, n_bins)
        features.append(feat)
    return np.stack(features) 

def compute_residue_from_frames(frame1, frame2):
    if frame1.shape != frame2.shape:
        raise ValueError("Input frames must have the same shape.")
    # Use int16 to avoid overflow in subtraction
    residue = np.abs(frame1.astype(np.int16) - frame2.astype(np.int16))
    return residue.astype(np.uint8)

def get_hist_features(distortions: list[Distortion], dtype=object):

    video_path = None
    residues = []
    labels = []
    label_threshold = 1600

    distortions.sort(key=lambda d: d.video_path)
    for distortion in distortions:

        if distortion.video_path != video_path:
            video_path = distortion.video_path


            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise IOError(f"Cannot open video: {video_path}")

        frame_num = [0, 1]
        fnum1, fnum2 = frame_num
        frames = {}
        # Only read frame 0 and 1

        residues = []
        labels = []

        x, y, size = int(distortion.x), int(distortion.y), int(distortion.size)
        
        for fnum in frame_num:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fnum)
            ret, frame = cap.read()
            if ret:
                frames[fnum] = frame[y:y+size, x:x+size]
            else:
                # If frame 30 missing, use last available (or frame 0)
                # frames[fnum] = frames.get(0, np.zeros((H, W, 3), dtype=np.uint8))
                raise IOError(f"Cannot read frame {fnum} from video: {video_path}")

        residue = compute_residue_from_frames(frames[fnum1], frames[fnum2])
        residues.append(residue)
        labels.append(0 if distortion.setting > label_threshold else 1)

    residues, labels = np.array(residues, dtype=dtype), np.array(labels)

    n_bins = 64

    X = extract_histogram_features(residues, n_bins)
    y = labels.astype(int)
    return X, y