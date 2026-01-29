"""
INES - Imputation Noise Evaluation Score

A tool for evaluating scRNA-seq imputation reliability using Earth Mover's Distance (EMD)
and designing optimized gene panels for spatial transcriptomics (MERFISH, Xenium).
"""

__version__ = "0.1.0"
__author__ = "INES Development Team"

from .metrics import earth_movers_distance, calculate_ines_score
from .design import greedy_probe_selection, optimize_gene_panel
from .models import ImputationSimulator
from .utils import parallelize, setup_hpc_environment

__all__ = [
    "earth_movers_distance",
    "calculate_ines_score",
    "greedy_probe_selection",
    "optimize_gene_panel",
    "ImputationSimulator",
    "parallelize",
    "setup_hpc_environment",
]
