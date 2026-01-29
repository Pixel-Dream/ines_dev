# INES - Imputation Noise Evaluation Score

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**INES** (Imputation Noise Evaluation Score) is a bioinformatics tool that uses Earth Mover's Distance (EMD) to evaluate scRNA-seq imputation reliability and design optimized gene panels for spatial transcriptomics experiments (MERFISH, Xenium).

## Overview

INES addresses a critical challenge in spatial transcriptomics: which genes should be measured directly versus inferred through imputation? By quantifying imputation noise using EMD, INES identifies "unimputable" genes that must be included in spatial probe panels.

### Key Features

- **EMD-based Imputation Quality Assessment**: Vectorized Earth Mover's Distance calculations for evaluating imputation reliability
- **Intelligent Probe Selection**: Greedy selection algorithm that identifies unimputable genes for optimal probe panel design
- **Imputation Simulation**: Lightweight PyTorch wrappers for testing various imputation scenarios
- **HPC Optimization**: Multiprocessing decorators and utilities for handling large genomic datasets
- **Comprehensive Benchmarking**: Scripts for comparing INES against traditional methods (HVG, co-expression networks)

## Installation

### From Source

```bash
git clone https://github.com/Pixel-Dream/ines_dev.git
cd ines_dev
pip install -e .
```

### With Optional Dependencies

```bash
# For development
pip install -e ".[dev]"

# For HPC environments
pip install -e ".[hpc]"

# Install all optional dependencies
pip install -e ".[all]"
```

### Dependencies

Core dependencies:
- `numpy>=1.20.0`
- `scipy>=1.7.0`
- `pandas>=1.3.0`
- `scanpy>=1.9.0`
- `torch>=1.10.0`
- `seaborn>=0.11.0`
- `matplotlib>=3.4.0`
- `scikit-learn>=1.0.0`
- `h5py>=3.0.0`
- `anndata>=0.8.0`

## Quick Start

### Basic Usage

```python
import scanpy as sc
from ines import calculate_ines_score, optimize_gene_panel
from ines.metrics import compute_gene_reliability_matrix

# Load your scRNA-seq data
adata = sc.read_h5ad("your_data.h5ad")

# Assume you have imputed data in adata.layers['imputed']
# Calculate INES scores for all genes
ines_scores, gene_names = compute_gene_reliability_matrix(
    adata,
    imputed_layer="imputed"
)

# Design optimal probe panel (e.g., 300 genes for MERFISH)
panel_result = optimize_gene_panel(
    adata,
    n_probes=300,
    strategy="ines"
)

print(f"Selected {len(panel_result['genes'])} genes for spatial panel")
print(f"Top genes: {panel_result['genes'][:10]}")
```

### Running Benchmarks

```bash
# Compare INES vs HVG/Co-expression methods
python benchmarks/reconstruction_benchmark.py \
    --data-path ./data \
    --dataset pbmc \
    --probes 100 200 300 500 \
    --output-dir results/

# Test EMD scalability
python benchmarks/runtime_test.py \
    --test-type scaling \
    --max-cells 50000 \
    --max-genes 5000 \
    --output-dir results/
```

## Project Structure

```
ines_dev/
├── src/ines/              # Main package source code
│   ├── __init__.py       # Package initialization
│   ├── metrics.py        # EMD calculations and INES scoring
│   ├── design.py         # Probe selection algorithms
│   ├── models.py         # Imputation simulators
│   └── utils.py          # HPC utilities and helpers
├── benchmarks/           # Reproducibility scripts
│   ├── reconstruction_benchmark.py  # Compare methods
│   └── runtime_test.py             # Scalability tests
├── tests/                # Unit tests (pytest)
│   ├── test_metrics.py
│   ├── test_design.py
│   ├── test_models.py
│   └── test_utils.py
├── notebooks/            # Tutorials and examples
├── docs/                 # Documentation
├── pyproject.toml        # Modern build configuration
└── README.md            # This file
```

## Documentation

### Core Modules

#### `metrics.py`
- `earth_movers_distance()`: Vectorized EMD computation
- `calculate_ines_score()`: Compute INES scores for genes
- `compute_gene_reliability_matrix()`: Process AnnData objects

#### `design.py`
- `greedy_probe_selection()`: Greedy algorithm for probe selection
- `optimize_gene_panel()`: High-level panel optimization
- `evaluate_panel_coverage()`: Assess panel quality

#### `models.py`
- `ImputationSimulator`: Test various imputation methods
- Support for k-NN, MAGIC, DCA, and scVI-style imputation

#### `utils.py`
- `parallelize()`: Decorator for parallel execution
- `setup_hpc_environment()`: Configure HPC settings
- `batch_process()`: Memory-efficient batch processing

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=ines --cov-report=html

# Run specific test file
pytest tests/test_metrics.py -v
```

## Development

### Setting Up Development Environment

```bash
# Clone repository
git clone https://github.com/Pixel-Dream/ines_dev.git
cd ines_dev

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

### Code Style

This project uses:
- `black` for code formatting (line length: 100)
- `isort` for import sorting
- `flake8` for linting
- `mypy` for type checking

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint
flake8 src/ tests/

# Type check
mypy src/
```

## Citation

If you use INES in your research, please cite:

```
[Citation information will be added upon publication]
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions, issues, or collaborations:
- **GitHub Issues**: [Create an issue](https://github.com/Pixel-Dream/ines_dev/issues)
- **Email**: ines@example.com

## Acknowledgments

- Earth Mover's Distance implementation based on SciPy's Wasserstein distance
- Inspired by spatial transcriptomics technologies (MERFISH, Xenium)
- Built on the excellent Scanpy ecosystem for scRNA-seq analysis

---

**Note**: INES is under active development. Features and APIs may change. For the latest stable version, please refer to the releases page.
