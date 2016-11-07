import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# based on https://github.com/mrahtz/sanger-machine-learning-workshop/blob/master/learn_utils.py
def segment(data, segment_len, step_size):
    segments = []

    for start in range(0, len(data), step_size):
        end = start + segment_len
        # numpy arrays are mutable, so we need to copy so we don't alter the original data
        segment = np.copy(data[start:end])
        if len(segment) != segment_len:
            continue
        segments.append(segment)

    return segments

# based on https://github.com/mrahtz/sanger-machine-learning-workshop/blob/master/learn_utils.py
def plot_segments(segments, step):
    plt.figure(figsize=(12,12))
    num_cols = 4
    num_rows = 4
    graph_num = 1
    seg_num = 1

    for _ in range(num_rows):
        for _ in range(num_cols):
            axes = plt.subplot(num_rows, num_cols, graph_num)
            plt.plot(segments[seg_num])
            graph_num += 1
            seg_num += step

    plt.tight_layout()
    plt.show()

def window_segments(segments, segment_len):
    rad = np.linspace(0, np.pi, segment_len)
    window = np.sin(rad)**2
    windowed_segments = [np.copy(segment) * window for segment in segments]
           
    return windowed_segments

def cluster(segments, num_clusters):
    clusterer = KMeans(n_clusters=num_clusters)
    clusterer.fit(segments)

    return clusterer
    
def reconstruct(data, segment_len, clusterer):
    slide_len = int(segment_len / 2)
    segments = segment(data, segment_len, slide_len)
    windowed_segments = window_segments(segments, segment_len)
    reconstructed = np.zeros(len(data))
    for segment_num, seg in enumerate(windowed_segments):
        # calling seg.reshape(1,-1) is done to avoid a DeprecationWarning from sklearn
        nearest_centroid_idx = clusterer.predict(seg.reshape(1,-1))[0]
        nearest_centroid = np.copy(clusterer.cluster_centers_[nearest_centroid_idx])
        pos = segment_num * slide_len
        reconstructed[pos:pos+segment_len] += nearest_centroid

    return reconstructed

def init_pyplot():
    from IPython.display import set_matplotlib_formats
    set_matplotlib_formats('pdf', 'png')
    plt.rcParams['savefig.dpi'] = 75
    plt.rcParams['figure.autolayout'] = False
    plt.rcParams['figure.figsize'] = 10, 6
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['font.size'] = 12
    plt.rcParams['lines.linewidth'] = 2.0
    plt.rcParams['lines.markersize'] = 8
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = "sans serif"
    plt.rcParams['font.serif'] = "cm"
    plt.rcParams['text.latex.preamble'] = r"\usepackage{subdepth}, \usepackage{type1cm}"
