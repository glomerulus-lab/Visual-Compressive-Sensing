#!/usr/bin/env python
"""Regenerate the paper figures from the LASSO sweeps -- DCT only.

Reads result/lasso/dct/ and writes figures/lasso/paper_new by default. The
figure building itself lives in src/sweep_figures.py, shared with
bp_sweep_figures.py so the two cannot drift apart.

Figures produced:
  varying_n.{svg,jpg}           error vs number of measurements
  varying_alpha.{svg,jpg}       error vs LASSO penalty
  varying_filter_dim.{svg,jpg}  error vs patch dimension
  dct_reconstructions.{svg,jpg} 4 images x [original, pixel, gaussian, V1]
  zoomed.{svg,jpg}              one image, full view plus zoomed crops

Usage:
    python lasso_sweep_figures.py
    python lasso_sweep_figures.py --outdir figures/lasso/paper_new
    python lasso_sweep_figures.py --only varying_n varying_alpha
    python lasso_sweep_figures.py --num-cell 128        # faster reconstructions
"""

from src import sweep_figures

if __name__ == '__main__':
    sweep_figures.main('lasso', __doc__)
