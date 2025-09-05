import numpy as np
import matplotlib.pyplot as plt

class FramePCA:
    def __init__(self):
        self.eigenvector = None
        self.freeze_eigenvectors = False
        
    def set_freeze_eigenvectors(self, freeze: bool):
        self.freeze_eigenvectors = freeze

    def initialize_histogram(self, n_bins=64):
        self.fig, self.ax = plt.subplots()
        self.ax.set_title('Live Updating Histogram')
        self.ax.set_xlabel('Value')
        self.ax.set_ylabel('Frequency')
        self.n_bins = n_bins

        # sets the x and y axis maximum values
        self.ax.set_xlim(0, 256)
        self.ax.set_ylim(0, 12000)

        plt.ion()
        plt.show()

    def update_histogram(self, data):
        if isinstance(data, np.ndarray) and data.ndim > 1:
            data = data.flatten()

        self.ax.clear()
        
        # Plot histogram
        self.ax.hist(data, bins=self.n_bins, color='skyblue', edgecolor='black', alpha=0.7)
        self.ax.set_title('Live Updating Histogram')
        self.ax.set_xlabel('Value')
        self.ax.set_ylabel('Frequency')

        # sets the x and y axis maximum values
        self.ax.set_xlim(0, 256)
        self.ax.set_ylim(0, 12000)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def patch_pca(self, frame_t, frame_tn1, topleft, patch_size=32, entire_frame=False, get_histogram_data=False):

        x0, y0 = topleft
        roi_t = frame_t[y0:y0+patch_size, x0:x0+patch_size]
        roi_tn1 = frame_tn1[y0:y0+patch_size, x0:x0+patch_size]

        # color pca
        data = np.concatenate((roi_t, roi_tn1), axis=0)
        data = data.reshape(-1, 3).T
        mean_color = np.mean(data, axis=1, keepdims=True)

        if self.freeze_eigenvectors and self.eigenvector is not None:
            eigenvector = self.eigenvector

        else:            
            data_centered = data - mean_color
            covariance_matrix = data_centered @ data_centered.T / (3 - 1)

            # get eigenvectors
            eigenvals, eigenvecs = np.linalg.eigh(covariance_matrix)  # eigh for symmetric matrices

            # Sort by eigenvalues in descending order
            idx = np.argsort(eigenvals)[::-1]  # descending
            eigenvals = eigenvals[idx]

            eigenvectors = eigenvecs[:, idx]
            eigenvector = eigenvectors[0, :]

            self.eigenvector = eigenvector

        # print(frame_t.shape, roi_t.shape)

        # pixelwise subtraction
        if entire_frame:
            pca_t = (frame_t.reshape(-1, 3).T - mean_color).T @ eigenvector
            pca_t = pca_t.reshape(frame_t.shape[0], frame_t.shape[1])

            pca_tn1 = (frame_tn1.reshape(-1, 3).T - mean_color).T @ eigenvector
            pca_tn1 = pca_tn1.reshape(frame_tn1.shape[0], frame_tn1.shape[1])

            pca_t, pca_tn1 = np.clip(pca_t, 0, 255), np.clip(pca_tn1, 0, 255)
            pca_t, pca_tn1 = (pca_t - pca_t.min()) / (pca_t.max() - pca_t.min()) * 200, (pca_tn1 - pca_tn1.min()) / (pca_tn1.max() - pca_tn1.min()) * 200
            diff = pca_t - pca_tn1
            diff = np.abs(diff)

            arr = np.stack([diff, diff, diff], axis=-1)
            arr = arr.astype(np.uint8)
        
        else:
            arr = None
            

        pca_t = (roi_t.reshape(-1, 3).T - mean_color).T @ eigenvector
        pca_t = pca_t.reshape(patch_size, patch_size)

        pca_tn1 = (roi_tn1.reshape(-1, 3).T - mean_color).T @ eigenvector
        pca_tn1 = pca_tn1.reshape(patch_size, patch_size)

        pca_t, pca_tn1 = np.clip(pca_t, 0, 255), np.clip(pca_tn1, 0, 255)
        pca_t, pca_tn1 = (pca_t - pca_t.min()) / (pca_t.max() - pca_t.min()) * 200, (pca_tn1 - pca_tn1.min()) / (pca_tn1.max() - pca_tn1.min()) * 200
        diff = pca_t - pca_tn1
        diff = np.abs(diff)

        # doubt in normalization here
        diff = (diff - diff.min()) / (diff.max() - diff.min()) * 255

        if get_histogram_data:
            bin_edges = np.linspace(0, 256, 65)
            counts, _ = np.histogram(diff, bin_edges)
            return counts

        self.update_histogram(diff)

        return arr