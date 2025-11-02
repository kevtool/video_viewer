import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def compute_residue_from_frames(frame1, frame2, abs=False):
    if frame1.shape != frame2.shape:
        raise ValueError("Input frames must have the same shape.")
    # Use int16 to avoid overflow in subtraction
    if abs:
        residue = np.abs(frame1.astype(np.int16) - frame2.astype(np.int16))
    else:
        residue = frame1.astype(np.int16) - frame2.astype(np.int16)
    return residue.astype(np.uint8)

def extract_color_patches(img: np.ndarray, patch_size: int) -> np.ndarray:
    """
    Extract all patch_size*patch_size*3 patches (stride=1) from an image of shape (H, W, 3).
    Returns array of shape (num_patches, patch_size*patch_size*3).
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Each residue must have shape (H, W, 3).")

    # sliding_window_view over (H, W, C) with window (patch_size,patch_size,3)
    win = sliding_window_view(img, (patch_size, patch_size, 3))
    # win shape: (H-(patch_size-1), W-(patch_size-1), 1, patch_size, patch_size, 3) -> squeeze the singleton channel window dim
    win = win.reshape(win.shape[0], win.shape[1], patch_size, patch_size, 3)
    patches = win.reshape(-1, patch_size * patch_size * 3)
    return patches