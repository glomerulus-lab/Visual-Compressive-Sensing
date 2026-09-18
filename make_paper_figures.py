#!/usr/bin/env python
"""Regenerate the paper figures from the swept data -- DCT only.

The figures under figures/paper were assembled by hand in Inkscape: the
varying_* panels are matplotlib SVGs pasted side by side with DCT/DWT labels
added by hand, and the reconstruction grids are composites of individually
rendered images. Nothing recorded which sources went where, so this rebuilds
them from result/dct/ instead of trying to recover that layout.

Differences from the hand-made originals, by design:
  * DCT only. The DWT halves of varying_* are dropped, and dwt_reconstructions
    is not produced. figure.py keeps its dwt branches; they are just not called.
  * One panel per image (1x4) for the varying_* figures, rather than the
    DCT/DWT pairing, since the panel count and image choice of the originals
    are not recoverable from the Inkscape files.

Figures produced:
  varying_n.{svg,jpg}           error vs number of measurements
  varying_alpha.{svg,jpg}       error vs LASSO penalty
  varying_filter_dim.{svg,jpg}  error vs patch dimension
  dct_reconstructions.{svg,jpg} 4 images x [original, pixel, gaussian, V1]
  zoomed.{svg,jpg}              one image, full view plus zoomed crops

Usage:
    python make_paper_figures.py --outdir figures/paper_new
    python make_paper_figures.py --only varying_n varying_alpha
    python make_paper_figures.py --num-cell 128        # faster reconstructions
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

ZOOM_IMAGE = 'barbara.bmp'
ZOOM_BOX = (180, 220, 140, 140)      # x, y, w, h -- the scarf detail


# ---------------------------------------------------------------- helpers
def save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    for ext in ('svg', 'jpg'):
        path = os.path.join(outdir, f'{name}.{ext}')
        fig.savefig(path, format=ext, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {name}.svg / .jpg")


def best_params(img, observation, num_cell=None):
    """Hyperparameters with the lowest mean error for one image/observation.

    Averages over repetitions first, so a single lucky run cannot win. When
    num_cell is given the search is restricted to it, which is the knob for
    trading reconstruction quality against runtime.
    """
    img_nm = img.split('.')[0]
    path = os.path.join(search_root(), 'result', METHOD, img_nm,
                        observation, CSV[observation])
    df = pd.read_csv(path)
    if num_cell is not None:
        df = df[df['num_cell'] == num_cell]
        if df.empty:
            raise SystemExit(f"no rows with num_cell={num_cell} in {path}")
    par = [c for c in df.columns if c not in ('error', 'rep')]
    means = df.groupby(par, dropna=False)['error'].mean().reset_index()
    return means.loc[means['error'].idxmin()].to_dict()


def reconstruct(img, observation, num_cell=None, cache={}):
    """Reconstruct one image with its best hyperparameters (memoised)."""
    key = (img, observation, num_cell)
    if key in cache:
        return cache[key]
    p = best_params(img, observation, num_cell)
    color = IMAGES[img]
    arr = process_image(img, color)
    kw = {}
    if observation == 'V1':
        kw = dict(cell_size=int(p['cell_size']), blob_size=int(p['sparse_freq']))
    print(f"    reconstructing {img:<12} {observation:<9} "
          f"n={int(p['num_cell'])} alpha={p['alp']} {kw if kw else ''}")
    rec = large_img_experiment(
        arr, num_cell=int(p['num_cell']), alpha=float(p['alp']),
        method=METHOD, observation=observation, color=color,
        fixed_weights=False, filter_dim=eval(p['filter_dim']), **kw)
    cache[key] = rec
    return rec


def show(ax, arr):
    ax.imshow(arr.astype('uint8') if arr.ndim == 3 else arr,
              cmap=None if arr.ndim == 3 else 'gray')
    ax.set_xticks([]); ax.set_yticks([])


# ---------------------------------------------------------------- figures
def varying(fn, name, outdir, images, xlabel_only_first=True):
    """One panel per image, side by side."""
    fig, axes = plt.subplots(1, len(images), squeeze=False,
                             figsize=(6 * len(images), 5.5))
    axes = list(axes[0])
    for ax, img in zip(axes, images):
        fn(img, METHOD, CSV['pixel'], CSV['gaussian'], CSV['V1'], ax=ax)
    fig.tight_layout()
    save(fig, outdir, name)


def reconstructions(outdir, images, num_cell):
    cols = ['Original'] + [LABEL[o] for o in OBSERVATIONS]
    # squeeze=False keeps axes 2-D even for a single image
    fig, axes = plt.subplots(len(images), 4, squeeze=False,
                             figsize=(4 * 3.2, len(images) * 3.2))
    for r, img in enumerate(images):
        show(axes[r][0], process_image(img, IMAGES[img]))
        axes[r][0].set_ylabel(img.split('.')[0].capitalize(), fontsize=16)
        for c, obs in enumerate(OBSERVATIONS, start=1):
            show(axes[r][c], reconstruct(img, obs, num_cell))
        if r == 0:
            for c, name in enumerate(cols):
                axes[r][c].set_title(name, fontsize=16)
    fig.tight_layout()
    save(fig, outdir, 'dct_reconstructions')


def zoomed(outdir, num_cell):
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
        show(axes[i], crop(reconstruct(img, obs, num_cell)))
        axes[i].set_title(LABEL[obs], fontsize=15)
    fig.tight_layout()
    save(fig, outdir, 'zoomed')


# ---------------------------------------------------------------- driver
FIGURES = ['varying_n', 'varying_alpha', 'varying_filter_dim',
           'dct_reconstructions', 'zoomed']


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--outdir', default='figures/paper_new')
    ap.add_argument('--images', nargs='+', default=list(IMAGES))
    ap.add_argument('--only', nargs='+', choices=FIGURES, default=FIGURES)
    ap.add_argument('--num-cell', type=int, default=None,
                    help='restrict reconstructions to this num_cell (faster)')
    args = ap.parse_args()

    print(f"writing to {args.outdir}  (method={METHOD}, images={args.images})")
    if 'varying_n' in args.only:
        varying(error_vs_num_cell, 'varying_n', args.outdir, args.images)
    if 'varying_alpha' in args.only:
        varying(error_vs_alpha, 'varying_alpha', args.outdir, args.images)
    if 'varying_filter_dim' in args.only:
        varying(error_vs_filter_dim, 'varying_filter_dim', args.outdir, args.images)
    if 'dct_reconstructions' in args.only:
        reconstructions(args.outdir, args.images, args.num_cell)
    if 'zoomed' in args.only:
        zoomed(args.outdir, args.num_cell)
    print("done")


if __name__ == '__main__':
    main()
