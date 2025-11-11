"""
Kaggle 5K Notebook Extractor
Main package initialization
"""

__version__ = "1.0.0"
__author__ = "Kaggle 5K Project"
__description__ = "Legal extraction of 5000 high-quality Kaggle competition notebooks"

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Key directories
CONFIG_DIR = PROJECT_ROOT / "config"
METADATA_DIR = PROJECT_ROOT / "metadata"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
CURATED_DIR = PROJECT_ROOT / "notebooks_curated"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
for directory in [METADATA_DIR, NOTEBOOKS_DIR, CURATED_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
