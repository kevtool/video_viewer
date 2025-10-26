import cv2
import numpy as np

def compute_residue_from_frames(frame1, frame2):
    if frame1.shape != frame2.shape:
        raise ValueError("Input frames must have the same shape.")
    # Use int16 to avoid overflow in subtraction
    residue = np.abs(frame1.astype(np.int16) - frame2.astype(np.int16))
    return residue.astype(np.uint8)

def get_hist_features(video_path, rois, dtype=object):
    """
    Extract RGB data from multiple ROIs in the first frame of a video.

    Args:
        video_path (str): Absolute path to the video.
        rois (list of tuples): Each tuple is (x, y, size).
    
    Returns:
        list of numpy.ndarray: List of RGB ROI arrays.
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    frame_num = [0, 1]
    fnum1, fnum2 = frame_num
    frames = {}
    # Only read frame 0 and 1

    residues=[]
    labels=[]
    
    for (x, y, size) in rois:
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
        labels.append(0 if setting > label_threshold else 1)

    return np.array(residues, dtype=dtype), np.array(labels)