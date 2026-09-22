#!/usr/bin/env python
"""Build the paper figures from swept data, for one solver -- DCT only.

This is the shared body behind the repo-root drivers lasso_sweep_figures.py and
bp_sweep_figures.py, which are what you run (`python bp_sweep_figures.py`).
Everything here is parameterised by `algorithm`, which selects both the
result/<algorithm>/dct/ tree the sweeps are read from and the
figures/<algorithm>/ tree the output is written to.

The figures under figures/paper were assembled by hand in Inkscape: the
varying_* panels are matplotlib SVGs pasted side by side with DCT/DWT labels
added by hand, and the reconstruction grids are composites of individually
rendered images. Nothing recorded which sources went where, so this rebuilds
them from the swept CSVs instead of trying to recover that layout.

Differences from the hand-made originals, by design:
  * DCT only. The DWT halves of varying_* are dropped, and dwt_reconstructions
    is not produced. figure.py keeps its dwt branches; they are just not called.
  * One panel per image (1x4) for the varying_* figures, rather than the
    DCT/DWT pairing, since the panel count and image choice of the originals
    are not recoverable from the Inkscape files.

Figures produced:
  varying_n.{svg,jpg}           error vs number of measurements
  varying_alpha.{svg,jpg}       error vs penalty      (penalised solvers only)
  varying_filter_dim.{svg,jpg}  error vs patch dimension
  dct_reconstructions.{svg,jpg} 4 images x [original, pixel, gaussian, V1]
  zoomed.{svg,jpg}              one image, full view plus zoomed crops
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd

from src.compress_sensing import large_img_experiment
from src.utility import process_image, search_root
from src.figure import error_vs_num_cell, error_vs_alpha, error_vs_filter_dim

METHOD = 'dct'
# image file -> whether the sweep for it was run in colour (-color)
IMAGES = {'baboon.png': True, 'barbara.bmp': False,
          'boat.png': False, 'fruits.png': True}
OBSERVATIONS = ['pixel', 'gaussian', 'V1']
CSV = {'pixel': 'Pixel.csv', 'gaussian': 'Gaussian.csv', 'V1': 'V1.csv'}
LABEL = {'pixel': 'Pixel', 'gaussian': 'Gaussian white noise', 'V1': 'V1-Inspired'}

# Solvers with an alpha penalty. The others sweep no alpha (their `alp` column
# is empty), so varying_alpha has nothing to plot against.
PENALISED = ('lasso', 'ridge')

ZOOM_IMAGE = 'barbara.bmp'
ZOOM_BOX = (180, 220, 140, 140)      # x, y, w, h -- the scarf detail

ALL_FIGURES = ['varying_n', 'varying_alpha', 'varying_filter_dim',
               'dct_reconstructions', 'zoomed']


def figures_for(algorithm):
    """The figures that make sense for one solver."""
    return [f for f in ALL_FIGURES
            if f != 'varying_alpha' or algorithm in PENALISED]


def default_outdir(algorithm):
    # Anchored at the repo root, not the cwd: the drivers can be run from
    # anywhere, and an explicit --outdir is still taken as given.
    return os.path.join(search_root(), 'figures', algorithm, 'paper_new')


# ---------------------------------------------------------------- helpers
def save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    for ext in ('svg', 'jpg'):
        path = os.path.join(outdir, f'{name}.{ext}')
        fig.savefig(path, format=ext, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {name}.svg / .jpg")


def best_params(img, observation, algorithm, num_cell=None, filter_dim=None):
    """Hyperparameters with the lowest mean error for one image/observation.

    Averages over repetitions first, so a single lucky run cannot win. When
    num_cell is given the search is restricted to it, which is the knob for
    trading reconstruction quality against runtime. filter_dim restricts it the
    same way, as the patch side length: the sweeps only ever use square
    patches, so 32 means the (32, 32) rows. The stored value is the repr of a
    tuple, so it is parsed rather than string-matched -- nothing guarantees the
    spacing is uniform across files written at different times.
    """
    img_nm = img.split('.')[0]
    path = os.path.join(search_root(), 'result', algorithm, METHOD, img_nm,
                        observation, CSV[observation])
    df = pd.read_csv(path)
    if num_cell is not None:
        df = df[df['num_cell'] == num_cell]
        if df.empty:
            raise SystemExit(f"no rows with num_cell={num_cell} in {path}")
    if filter_dim is not None:
        df = df[df['filter_dim'].map(eval) == (filter_dim, filter_dim)]
        if df.empty:
            raise SystemExit(
                f"no rows with filter_dim=({filter_dim}, {filter_dim}) in {path}")
    par = [c for c in df.columns if c not in ('error', 'rep')]
    # dropna=False keeps the all-empty `alp` column of an unpenalised sweep
    # from discarding every row.
    means = df.groupby(par, dropna=False)['error'].mean().reset_index()
    return means.loc[means['error'].idxmin()].to_dict()


def reconstruct(img, observation, algorithm, num_cell=None, filter_dim=None,
                cache={}):
    """Reconstruct one image with its best hyperparameters (memoised)."""
    key = (img, observation, algorithm, num_cell, filter_dim)
    if key in cache:
        return cache[key]
    p = best_params(img, observation, algorithm, num_cell, filter_dim)
    color = IMAGES[img]
    arr = process_image(img, color)
    kw = {}
    if observation == 'V1':
        kw = dict(cell_size=int(p['cell_size']), blob_size=int(p['sparse_freq']))
    # An unpenalised solver has no alpha to pass on; alpha=None also keeps
    # large_img_experiment from choking on the empty `alp` cell.
    alpha = float(p['alp']) if algorithm in PENALISED else None
    print(f"    reconstructing {img:<12} {observation:<9} "
          f"n={int(p['num_cell'])} filter_dim={p['filter_dim']} "
          f"alpha={alpha} {kw if kw else ''}")
    rec = large_img_experiment(
        arr, num_cell=int(p['num_cell']), alpha=alpha,
        method=METHOD, observation=observation, color=color,
        fixed_weights=False, filter_dim=eval(p['filter_dim']),
        algorithm=algorithm, **kw)
    cache[key] = rec
    return rec


def show(ax, arr):
    ax.imshow(arr.astype('uint8') if arr.ndim == 3 else arr,
              cmap=None if arr.ndim == 3 else 'gray')
    ax.set_xticks([]); ax.set_yticks([])


# ---------------------------------------------------------------- figures
def varying(fn, name, outdir, images, algorithm):
    """One panel per image, side by side."""
    fig, axes = plt.subplots(1, len(images), squeeze=False,
                             figsize=(6 * len(images), 5.5))
    axes = list(axes[0])
    for ax, img in zip(axes, images):
        fn(img, METHOD, CSV['pixel'], CSV['gaussian'], CSV['V1'], ax=ax,
           algorithm=algorithm)
    fig.tight_layout()
    save(fig, outdir, name)


def reconstructions(outdir, images, algorithm, num_cell, filter_dim=None):
    cols = ['Original'] + [LABEL[o] for o in OBSERVATIONS]
    # squeeze=False keeps axes 2-D even for a single image
    fig, axes = plt.subplots(len(images), 4, squeeze=False,
                             figsize=(4 * 3.2, len(images) * 3.2))
    for r, img in enumerate(images):
        show(axes[r][0], process_image(img, IMAGES[img]))
        axes[r][0].set_ylabel(img.split('.')[0].capitalize(), fontsize=16)
        for c, obs in enumerate(OBSERVATIONS, start=1):
            show(axes[r][c], reconstruct(img, obs, algorithm, num_cell,
                                         filter_dim))
        if r == 0:
            for c, name in enumerate(cols):
                axes[r][c].set_title(name, fontsize=16)
    fig.tight_layout()
    save(fig, outdir, 'dct_reconstructions')


def zoomed(outdir, algorithm, num_cell, filter_dim=None):
    img = ZOOM_IMAGE
    x, y, w, h = ZOOM_BOX
    full = process_image(img, IMAGES[img])
    fig, axes = plt.subplots(1, 5, figsize=(5 * 3.2, 3.6))

    show(axes[0], full)
    axes[0].add_patch(patches.Rectangle((x, y), w, h, fill=False,
                                        edgecolor='red', linewidth=2))
    axes[0].set_title('Original', fontsize=15)

    crop = lambda a: a[y:y + h, x:x + w]
    show(axes[1], crop(full))
    axes[1].set_title('Original', fontsize=15)
    for i, obs in enumerate(OBSERVATIONS, start=2):
        show(axes[i], crop(reconstruct(img, obs, algorithm, num_cell,
                                       filter_dim)))
        axes[i].set_title(LABEL[obs], fontsize=15)
    fig.tight_layout()
    save(fig, outdir, 'zoomed')


# ---------------------------------------------------------------- driver
def build(algorithm, outdir, images, only, num_cell, filter_dim=None):
    """Produce the requested figures for one solver."""
    print(f"writing to {outdir}  (algorithm={algorithm}, method={METHOD}, "
          f"images={images})")
    if 'varying_n' in only:
        varying(error_vs_num_cell, 'varying_n', outdir, images, algorithm)
    if 'varying_alpha' in only:
        varying(error_vs_alpha, 'varying_alpha', outdir, images, algorithm)
    if 'varying_filter_dim' in only:
        varying(error_vs_filter_dim, 'varying_filter_dim', outdir, images,
                algorithm)
    if 'dct_reconstructions' in only:
        reconstructions(outdir, images, algorithm, num_cell, filter_dim)
    if 'zoomed' in only:
        zoomed(outdir, algorithm, num_cell, filter_dim)
    print("done")


def main(algorithm, doc=None):
    """CLI shared by the per-solver drivers."""
    choices = figures_for(algorithm)
    ap = argparse.ArgumentParser(
        description=doc or __doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--outdir', default=default_outdir(algorithm))
    ap.add_argument('--images', nargs='+', default=list(IMAGES))
    ap.add_argument('--only', nargs='+', choices=choices, default=choices)
    ap.add_argument('--num-cell', type=int, default=None,
                    help='restrict reconstructions to this num_cell (faster)')
    ap.add_argument('--filter-dim', type=int, default=None,
                    help='restrict reconstructions to this patch side length, '
                         'e.g. 32 for the (32, 32) patches')
    args = ap.parse_args()
    build(algorithm, args.outdir, args.images, args.only, args.num_cell,
          args.filter_dim)
