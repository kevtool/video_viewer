import numpy as np

def patch_pca(frame_t, frame_tn1, topleft, patch_size=32, entire_frame=False):

    # extract roi
    x0, y0 = topleft
    roi_t = frame_t[y0:y0+patch_size, x0:x0+patch_size]
    roi_tn1 = frame_tn1[y0:y0+patch_size, x0:x0+patch_size]

    # color pca
    data = np.concatenate((roi_t, roi_tn1), axis=0)
    data = data.reshape(-1, 3).T
    mean_color = np.mean(data, axis=1, keepdims=True)

    data_centered = data - mean_color
    covariance_matrix = data_centered @ data_centered.T / (3 - 1)

    # get eigenvectors
    eigenvals, eigenvecs = np.linalg.eigh(covariance_matrix)  # eigh for symmetric matrices

    # Sort by eigenvalues in descending order
    idx = np.argsort(eigenvals)[::-1]  # descending
    eigenvals = eigenvals[idx]

    eigenvectors = eigenvecs[:, idx]
    eigenvector = eigenvectors[0, :]

    # pixelwise subtraction
    if entire_frame:
        print(frame_t.shape, eigenvector.shape)
        pca_t = frame_t @ eigenvector
        pca_tn1 = frame_tn1 @ eigenvector
        diff = pca_t - pca_tn1
        diff = np.abs(diff)

    else:
        print(roi_t.shape, eigenvector.shape)
        pca_t = roi_t @ eigenvector
        pca_tn1 = roi_tn1 @ eigenvector
        diff = pca_t - pca_tn1
        diff = np.abs(diff)