#!/usr/bin/env python
"""Regenerate the paper figures from the basis-pursuit sweeps -- DCT only.

The BP counterpart to lasso_sweep_figures.py: reads result/bp/dct/ (produced by
bp_dct_sweep.sh) and writes figures/bp/paper_new by default. The figure
building itself lives in sweep_figures.py, shared with lasso_sweep_figures.py
so the two cannot drift apart.

NO ALPHA FIGURE
    Basis pursuit minimises ||s||_1 subject to an exact constraint, so it has
    no penalty to sweep and its `alp` column is empty. varying_alpha is
    therefore not produced and is not a valid --only choice here; the
    reconstructions pick their best hyperparameters without it.

Figures produced:
  varying_n.{svg,jpg}           error vs number of measurements
  varying_filter_dim.{svg,jpg}  error vs patch dimension
  dct_reconstructions.{svg,jpg} 4 images x [original, pixel, gaussian, V1]
  zoomed.{svg,jpg}              one image, full view plus zoomed crops

Usage:
    python bp_sweep_figures.py
    python bp_sweep_figures.py --outdir figures/bp/paper_new
    python bp_sweep_figures.py --only varying_n
    python bp_sweep_figures.py --num-cell 128        # faster reconstructions

Requires the BP sweeps to have been run and consolidated first:
    ./bp_dct_sweep.sh
    python consolidate_results.py --algorithm bp
"""

import sweep_figures

if __name__ == '__main__':
    sweep_figures.main('bp', __doc__)
