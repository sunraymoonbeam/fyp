#!/bin/bash

echo "🚀 Starting Food Safety EDA..."
echo ""

# Check if venv exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Run 'uv venv' first."
    exit 1
fi

# Activate venv
source .venv/bin/activate

# Check if jupyter is installed
if ! command -v jupyter &> /dev/null; then
    echo "📦 Installing Jupyter with UV..."
    uv pip install jupyter
fi

# Launch Jupyter Lab
echo "✅ Launching Jupyter Lab..."
echo "📊 Opening: notebooks/00_comprehensive_eda.ipynb"
echo ""
jupyter lab notebooks/00_comprehensive_eda.ipynb
