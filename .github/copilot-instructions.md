# Copilot Instructions for ines_dev

## Project Overview

This is the **INES (Imputation Noise Evaluation Score)** development repository. INES is a Python-based tool designed for evaluating imputation quality in single-cell RNA sequencing (scRNA-seq) data analysis.

### Repository Information
- **Type**: Python development project
- **Domain**: Bioinformatics / Single-cell genomics
- **Target**: scRNA-seq data imputation evaluation
- **Status**: Early development stage

## Project Structure

Currently, this is a minimal repository in the early stages of development. The repository structure will expand as the project grows.

### Current Layout
```
ines_dev/
├── README.md              # Project description
└── .github/
    └── copilot-instructions.md  # This file
```

### Expected Future Structure
As this project develops, expect to see:
- Python source code in a package directory (e.g., `ines/` or `src/ines/`)
- Test files (likely in `tests/` directory)
- Requirements files for dependencies
- Configuration files for Python tooling (e.g., `setup.py`, `pyproject.toml`)
- Documentation directory
- Example notebooks or scripts

## Development Guidelines

### Python Environment
When the project structure is established, development will likely follow standard Python practices:
- Use virtual environments for dependency isolation
- Follow PEP 8 style guidelines for Python code
- Include docstrings for functions and classes
- Write unit tests for new functionality

### Expected Workflow (to be updated as project develops)

**Installation** (future):
```bash
# When setup.py or pyproject.toml exists:
pip install -e .
# Or if requirements.txt exists:
pip install -r requirements.txt
```

**Testing** (future):
```bash
# When test infrastructure is added:
pytest
# or
python -m pytest tests/
```

**Code Quality** (future):
```bash
# Linting (if configured):
flake8 .
# or
pylint ines/
```

## Key Considerations for AI Agents

1. **Early Stage Project**: This repository is in early development. Most standard Python project files (setup.py, requirements.txt, tests, etc.) may not exist yet.

2. **Bioinformatics Domain**: Code in this repository deals with:
   - scRNA-seq data analysis
   - Imputation quality metrics
   - Statistical evaluation methods
   - Potentially large genomic datasets

3. **Dependencies**: When adding dependencies, expect common bioinformatics/data science libraries such as:
   - numpy, scipy for numerical computations
   - pandas for data manipulation
   - scanpy for scRNA-seq analysis
   - matplotlib/seaborn for visualization

4. **File Exploration**: Since the project structure is still forming:
   - Always check for existing files before creating new ones
   - Follow Python best practices for package structure
   - Keep the codebase organized and well-documented

5. **Testing**: When implementing new features:
   - Add appropriate unit tests
   - Consider edge cases specific to genomic data (e.g., missing values, sparse matrices)
   - Validate against typical scRNA-seq data characteristics

## Notes for Agents

- This repository is being actively developed. Check the current file structure before making assumptions.
- When build/test commands are added to the project, update this file with specific instructions.
- Follow scientific computing best practices for numerical stability and performance.
- Document any statistical methods or algorithms clearly.
- Use these instructions as guidance while verifying the current repository state, as it's evolving rapidly in early development.
