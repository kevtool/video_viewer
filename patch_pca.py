import numpy as np

def patch_pca(frame_t, frame_tn1):
    # color pca
    data = np.concatenate((frame_t, frame_tn1), axis=0)
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
    pca_t = frame_t.T @ eigenvector
    pca_tn1 = frame_tn1.T @ eigenvector
    diff = pca_t - pca_tn1