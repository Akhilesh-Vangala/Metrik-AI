#!/bin/bash
set -e

echo "=== Metrik AI Setup ==="

if ! command -v python &> /dev/null; then
    echo "Python not found. Please install Python 3.10+."
    exit 1
fi

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Building Cython extensions..."
python setup.py build_ext --inplace 2>/dev/null || echo "Cython build skipped (run manually if needed)"

echo "Downloading ASHRAE dataset..."
python scripts/download_data.py

echo "Running tests..."
python -m pytest tests/ -v --tb=short

echo "=== Setup complete ==="
