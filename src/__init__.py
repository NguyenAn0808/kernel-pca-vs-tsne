"""
Professional visualization utilities for comparing Kernel PCA and t-SNE.

This package provides reusable functions for dimensionality reduction comparison,
following sklearn API conventions and best practices.
"""

from .visualization import (
    compute_neighbor_preservation,
    compute_trustworthiness,
    compute_variance_explained,
    measure_computation_time,
    plot_embeddings_comparison,
    compare_reproducibility,
    create_summary_table
)

__version__ = "1.0.0"
__all__ = [
    "compute_neighbor_preservation",
    "compute_trustworthiness",
    "compute_variance_explained",
    "measure_computation_time",
    "plot_embeddings_comparison",
    "compare_reproducibility",
    "create_summary_table",
]

