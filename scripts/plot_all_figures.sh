#!/usr/bin/env bash
# Copyright (c) 2024-2025, SO-ARM101 Project
# SPDX-License-Identifier: BSD-3-Clause
#
# 依次运行三个可视化脚本，生成 PDF 图：
#   loss_landscape_compare.pdf
#   input_sensitivity.pdf
#   frequency_analysis.pdf, jerk_comparison.pdf
#
# 用法: 在仓库根目录执行  bash scripts/plot_all_figures.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "=========================================="
echo "  Frequency Analysis (spectrum + ratio + jerk)"
echo "=========================================="
python scripts/visualize_frequency_analysis.py

echo ""
echo "=========================================="
echo "  Done. PDFs in repo root:"
echo "    frequency_spectrum.pdf"
echo "    highfreq_ratio.pdf"
echo "    jerk_comparison.pdf"
echo "=========================================="
