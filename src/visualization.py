"""
Professional visualization utilities for comparing Kernel PCA and t-SNE.

This module provides reusable functions for dimensionality reduction comparison,
following sklearn API conventions and best practices.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, Any
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA, PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
import time


def compute_neighbor_preservation(
    X_original: np.ndarray,
    X_embedded: np.ndarray,
    k: int = 10
) -> float:
    """
    Compute neighbor preservation score between original and embedded spaces.
    
    Measures how well local neighborhoods are preserved in the embedding.
    Returns the mean fraction of k-nearest neighbors preserved.
    
    Parameters
    ----------
    X_original : np.ndarray
        Original high-dimensional data
    X_embedded : np.ndarray
        Low-dimensional embedding
    k : int, default=10
        Number of nearest neighbors to consider
        
    Returns
    -------
    float
        Mean neighbor preservation score (0-1, higher is better)
    """
    nbrs_orig = NearestNeighbors(n_neighbors=k + 1).fit(X_original)
    orig_neighbors = nbrs_orig.kneighbors(return_distance=False)[:, 1:]  # Exclude self
    
    nbrs_emb = NearestNeighbors(n_neighbors=k + 1).fit(X_embedded)
    emb_neighbors = nbrs_emb.kneighbors(return_distance=False)[:, 1:]  # Exclude self
    
    shared = [
        len(set(orig_neighbors[i]) & set(emb_neighbors[i])) / k 
        for i in range(len(X_embedded))
    ]
    return np.mean(shared)


def compute_trustworthiness(
    X_original: np.ndarray,
    X_embedded: np.ndarray,
    k: int = 10
) -> float:
    """
    Compute trustworthiness metric for embedding quality.
    
    Measures how well the embedding preserves local structure by checking
    if points that are neighbors in the embedding were also neighbors
    in the original space.
    
    Parameters
    ----------
    X_original : np.ndarray
        Original high-dimensional data
    X_embedded : np.ndarray
        Low-dimensional embedding
    k : int, default=10
        Number of nearest neighbors to consider
        
    Returns
    -------
    float
        Trustworthiness score (0-1, higher is better)
    """
    n = len(X_original)
    if n <= k:
        return 0.0
    
    # Find k-nearest neighbors in embedded space
    nbrs_emb = NearestNeighbors(n_neighbors=k + 1).fit(X_embedded)
    emb_neighbors = nbrs_emb.kneighbors(return_distance=False)[:, 1:]
    
    # Compute distances in original space
    dist_orig = pairwise_distances(X_original, metric='euclidean')
    
    trust = 0.0
    for i in range(n):
        # For each neighbor in embedding that wasn't a neighbor in original
        orig_ranks = np.argsort(dist_orig[i])
        for j_idx, j in enumerate(emb_neighbors[i]):
            if j not in orig_ranks[:k]:
                # Find rank in original space
                rank = np.where(orig_ranks == j)[0][0]
                trust += (rank - k)
    
    trust = 1.0 - (2.0 / (n * k * (2 * n - 3 * k - 1))) * trust
    return max(0.0, min(1.0, trust))


def compute_variance_explained(
    X_original: np.ndarray,
    X_embedded: np.ndarray
) -> float:
    """
    Compute variance explained ratio for Kernel PCA embeddings.
    
    For Kernel PCA, this approximates the variance preservation.
    For t-SNE, this metric is less meaningful but still computed.
    
    Parameters
    ----------
    X_original : np.ndarray
        Original high-dimensional data
    X_embedded : np.ndarray
        Low-dimensional embedding
        
    Returns
    -------
    float
        Ratio of variance in embedding to variance in original space
    """
    var_orig = np.var(X_original, axis=0).sum()
    var_emb = np.var(X_embedded, axis=0).sum()
    return var_emb / var_orig if var_orig > 0 else 0.0


def measure_computation_time(
    model: Any,
    X: np.ndarray,
    fit_transform: bool = True
) -> float:
    """
    Measure computation time for model fitting and transformation.
    
    Parameters
    ----------
    model : sklearn transformer
        The dimensionality reduction model
    X : np.ndarray
        Input data
    fit_transform : bool, default=True
        If True, measure fit_transform time; else measure transform only
        
    Returns
    -------
    float
        Computation time in seconds
    """
    start_time = time.time()
    if fit_transform:
        _ = model.fit_transform(X)
    else:
        _ = model.transform(X)
    return time.time() - start_time


def plot_embeddings_comparison(
    embeddings: Dict[str, np.ndarray],
    colors: np.ndarray,
    title: str = "Embedding Comparison",
    figsize: Tuple[int, int] = (15, 4),
    cmap: str = "Spectral"
) -> plt.Figure:
    """
    Create a side-by-side comparison plot of multiple embeddings.
    
    Parameters
    ----------
    embeddings : dict
        Dictionary mapping method names to embedding arrays
    colors : np.ndarray
        Color values for each point
    title : str, default="Embedding Comparison"
        Overall figure title
    figsize : tuple, default=(15, 4)
        Figure size
    cmap : str, default="Spectral"
        Colormap name
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    n_methods = len(embeddings)
    fig, axes = plt.subplots(1, n_methods, figsize=figsize)
    
    if n_methods == 1:
        axes = [axes]
    
    for idx, (method_name, embedding) in enumerate(embeddings.items()):
        scatter = axes[idx].scatter(
            embedding[:, 0], 
            embedding[:, 1], 
            c=colors, 
            cmap=cmap, 
            s=10,
            alpha=0.7
        )
        axes[idx].set_title(method_name, fontsize=12, fontweight='bold')
        axes[idx].set_xticks([])
        axes[idx].set_yticks([])
        axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def compare_reproducibility(
    model_class: Any,
    X: np.ndarray,
    seeds: list,
    model_params: Dict[str, Any],
    method_name: str
) -> Tuple[np.ndarray, float]:
    """
    Compare reproducibility across different random seeds.
    
    Parameters
    ----------
    model_class : sklearn class
        The model class to instantiate
    X : np.ndarray
        Input data
    seeds : list
        List of random seeds to test
    model_params : dict
        Parameters to pass to model (excluding random_state)
    method_name : str
        Name of the method for display
        
    Returns
    -------
    tuple
        (embeddings_array, std_deviation) where embeddings_array contains
        all embeddings and std_deviation is the mean standard deviation
        across runs
    """
    embeddings = []
    for seed in seeds:
        model = model_class(random_state=seed, **model_params)
        embedding = model.fit_transform(X)
        embeddings.append(embedding)
    
    embeddings = np.array(embeddings)
    std_dev = np.mean(np.std(embeddings, axis=0))
    
    return embeddings, std_dev


def create_summary_table(
    metrics: Dict[str, Dict[str, float]],
    methods: list
) -> None:
    """
    Print a formatted summary table of comparison metrics.
    
    Parameters
    ----------
    metrics : dict
        Dictionary mapping metric names to method scores
    methods : list
        List of method names to compare
    """
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    for metric_name, method_scores in metrics.items():
        print(f"\n{metric_name}:")
        print("-" * 40)
        for method in methods:
            score = method_scores.get(method, "N/A")
            if isinstance(score, float):
                print(f"  {method:20s}: {score:8.4f}")
            else:
                print(f"  {method:20s}: {score}")
    
    print("="*60 + "\n")

