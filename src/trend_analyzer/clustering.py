from typing import List, Tuple, Optional
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
from loguru import logger


class TextClustering:
    """Clusterization of texts"""

    def __init__(self, min_cluster_size: int = 5, min_samples: int = 2):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.last_quality_score = 0.0
        logger.info(f"Clusterer initialized (min_size={min_cluster_size})")


    def find_optimal_clusters(self, embeddings: np.ndarray, max_clusters: int = 10) -> int:
        """Finds the optimal number of clusters based on the minimum size"""

        if len(embeddings) <= 2:
            return max(1, len(embeddings))

        # Maximum number of clusters = total number of posts / minimum cluster size
        max_possible_clusters = len(embeddings) // self.min_cluster_size
        max_clusters = min(max_clusters, max_possible_clusters, 15)

        if max_clusters < 2:
            return 2

        silhouette_scores = []
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)

            # Checking for clusters that are too small
            unique, counts = np.unique(labels, return_counts=True)
            min_cluster_count = counts.min()

            if min_cluster_count >= self.min_cluster_size:
                score = silhouette_score(embeddings, labels)
                silhouette_scores.append(score)
            else:
                silhouette_scores.append(-1)

        # Finding the best score considering the size of the clusters
        if silhouette_scores and max(silhouette_scores) > 0:
            best_k = np.argmax(silhouette_scores) + 2
        else:
            #use heuristics
            best_k = max(2, min(5, len(embeddings) // 10))

        logger.info(f"Optimal number of clusters: {best_k} (min_size={self.min_cluster_size})")
        return best_k


    def _find_elbow(self, inertias: List[float]) -> int:
        """Находит 'локоть' в графике inertia"""
        if len(inertias) < 3:
            return 2

        # We calculate changes of changes
        deltas = np.diff(inertias)
        double_deltas = np.diff(deltas)

        # We find a point where the change slows down dramatically
        if len(double_deltas) > 0:
            elbow_point = np.argmin(double_deltas) + 2
        else:
            elbow_point = 2

        return min(max(2, elbow_point), len(inertias))


    def cluster_texts(self, texts: List[str], embeddings: np.ndarray) -> Tuple[np.ndarray, int]:
        """Clustering with filtering of small clusters"""
        if len(embeddings) <= 2:
            return np.arange(len(embeddings)), min(3, len(embeddings))

        # Reducing the dimension
        reduced_embeddings = self._reduce_dimensionality(embeddings)

        best_score = -1
        best_labels = None
        best_n_clusters = 1

        # trying different algorithms
        algorithms = [
            self._kmeans_clustering,
            self._hdbscan_clustering,
            self._gmm_clustering
        ]

        for algorithm in algorithms:
            try:
                labels, n_clusters = algorithm(reduced_embeddings)

                # Filtering small clusters
                filtered_labels, filtered_n_clusters = self._filter_small_clusters(labels)

                if filtered_n_clusters > 1:
                    # calculate score only valid clusters
                    valid_mask = filtered_labels != -1
                    if np.sum(valid_mask) > 1 and len(np.unique(filtered_labels[valid_mask])) > 1:
                        score = silhouette_score(
                            reduced_embeddings[valid_mask],
                            filtered_labels[valid_mask]
                        )

                        if score > best_score:
                            best_score = score
                            best_labels = filtered_labels
                            best_n_clusters = filtered_n_clusters

            except Exception as e:
                logger.debug(f"Algorithm {algorithm.__name__} not work: {e}")

        # if not find anything, we use a single cluster
        if best_labels is None:
            best_labels = np.zeros(len(embeddings))
            best_n_clusters = 1
            best_score = 0

        self.last_quality_score = best_score

        logger.success(f" Better clustering: {best_n_clusters} clusters, quality: {best_score:.3f}")
        return best_labels, best_n_clusters


    def _filter_small_clusters(self, labels: np.ndarray) -> Tuple[np.ndarray, int]:
        """Filters clusters with a small number of elements"""

        if len(labels) == 0:
            return labels, 0

        unique_labels, counts = np.unique(labels, return_counts=True)

        # We only keep clusters with a sufficient size
        valid_clusters = [label for label, count in zip(unique_labels, counts)
                          if count >= self.min_cluster_size]

        if len(valid_clusters) == 0:
            # If all the clusters are too small, return one large one
            return np.zeros_like(labels), 1

        # We create new labels only for valid clusters
        new_labels = np.full_like(labels, -1)  # -1 for noise

        for new_label_idx, old_label in enumerate(valid_clusters):
            new_labels[labels == old_label] = new_label_idx

        # We only count valid clusters (ignoring noise -1)
        valid_labels = new_labels[new_labels != -1]
        n_valid_clusters = len(np.unique(valid_labels)) if len(valid_labels) > 0 else 0

        return new_labels, n_valid_clusters


    def _reduce_dimensionality(self, embeddings: np.ndarray, n_components: int = 50) -> np.ndarray:
        """Reduces the dimensionality of embeddings"""

        if embeddings.shape[1] <= n_components:
            return embeddings

        try:
            n_components = min(n_components, len(embeddings) - 1, embeddings.shape[1] - 1)
            if n_components < 2:
                return embeddings

            pca = PCA(n_components=n_components, random_state=42)
            reduced = pca.fit_transform(embeddings)
            logger.debug(f"Reduced PCA dimension: {embeddings.shape[1]} → {reduced.shape[1]}")
            return reduced

        except Exception as e:
            logger.warning(f"PCA didn't work, so we're using the original embeddings: {e}")
            return embeddings


    def _kmeans_clustering(self, embeddings: np.ndarray) -> Tuple[np.ndarray, int]:
        """KMeans clustering"""

        n_clusters = self.find_optimal_clusters(embeddings)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        return labels, n_clusters


    def _hdbscan_clustering(self, embeddings: np.ndarray) -> Tuple[np.ndarray, int]:
        """HDBSCAN clustering is better at dealing with noise"""

        hdbscan = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_epsilon=0.5
        )
        labels = hdbscan.fit_predict(embeddings)

        # HDBSCAN returns -1 for noise
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

        if n_clusters > 1:
            return labels, n_clusters
        raise ValueError("HDBSCAN found only one cluster")

    def _gmm_clustering(self, embeddings: np.ndarray) -> Tuple[np.ndarray, int]:
        """Gaussian Mixture Model clustering"""

        n_clusters = self.find_optimal_clusters(embeddings)
        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        labels = gmm.fit_predict(embeddings)
        return labels, n_clusters
