import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from src.geometry.objects import Interval


def filter_relevant_dicts(data: list[Interval], n_clusters=3, scale=1000):
    """
    Filters dictionaries based on clustering of angle values using Gaussian Mixture.


    Args:
    data (list of dict): Input data with angle values in [-180, 180].
    key (str): Key to extract values from dictionaries.
    min_cluster_size (int): Minimum number of points to mark cluster as relevant.
    n_clusters (int): Number of Gaussian mixture components to fit.


    Returns:
    list of dict: Filtered dictionaries belonging to relevant clusters.
    """

    # Extract angles and convert to radians for circular clustering
    angles_deg = np.array([d.angle if d.angle >= 0 else d.angle + 180 for d in data])
    angles_rad = np.deg2rad(angles_deg)

    # Represent angles on the unit circle (x, y)
    coords = np.column_stack((np.cos(angles_rad), np.sin(angles_rad)))

    # Fit Gaussian Mixture Model
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    labels = gmm.fit_predict(coords)

    # Compute cluster scores based on "length"
    cluster_scores = {}
    for label in set(labels):
        cluster_lengths = [d.length for d, lbl in zip(data, labels) if lbl == label]
        # Score: max length in cluster (prioritize big values)
        cluster_scores[label] = max(cluster_lengths) if cluster_lengths else 0

    # print(cluster_scores)

    # Define relevant clusters
    # threshold = np.max(list(cluster_scores.values())) * .05
    threshold = scale * 0.05
    relevant_clusters = {
        label for label, score in cluster_scores.items() if score >= threshold
    }

    # Filter dictionaries that belong to relevant clusters
    for d, label in zip(data, labels):
        d.angle_label = label

    filtered_data = [d for d, label in zip(data, labels) if label in relevant_clusters]

    return filtered_data, labels


def select_relevant_lines(intervals, scale, tolerance_degree=2):
    selected_intervals, _ = filter_relevant_dicts(intervals, n_clusters=90, scale=scale)
    selected_intervals, _ = filter_relevant_dicts(
        selected_intervals, n_clusters=30, scale=scale
    )
    selected_intervals = [
        interval for interval in intervals if any([
            abs(sel_int.angle - interval.angle) < tolerance_degree for sel_int in selected_intervals])
    ]
    return selected_intervals
