import numpy as np
import matplotlib
matplotlib.use("Agg")  # avoid GUI
import matplotlib.pyplot as plt

def save_histogram_from_bins(bin_counts: np.ndarray, out_path="histogram.png"):
    if bin_counts.shape[0] != 64:
        raise ValueError("Array must have 64 values (one per bin).")
    
    # Bin edges from 0–256, so each bin covers 4 units
    bin_edges = np.linspace(0, 256, 65)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fig, ax = plt.subplots()
    ax.bar(bin_centers, bin_counts, width=4, color="skyblue", edgecolor="black", alpha=0.8)
    ax.set_title("ROI PCA-Diff Histogram")
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.set_xlim(0, 256)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)