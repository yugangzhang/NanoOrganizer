#!/bin/bash
# Upgrade Plotly and Kaleido to compatible versions

echo "🔧 Upgrading Plotly and Kaleido..."
echo ""

# Uninstall old versions
pip uninstall -y plotly kaleido

# Install compatible versions
pip install "plotly>=6.1.1" "kaleido>=0.2.1"

echo ""
echo "✅ Done! Versions installed:"
pip show plotly | grep Version
pip show kaleido | grep Version

echo ""
echo "🚀 Restart nanoorganizer to use the updated versions"
