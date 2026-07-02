# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Research codebase for compressed sensing (CS) image reconstruction. It compares three "observation" (sampling) techniques — random pixel selection, Gaussian random projections, and a biologically-inspired V1 (primary visual cortex) receptive-field model — to see which produces the best sparse-coding reconstruction of an image from a small number of samples. Reconstruction is done by fitting a LASSO model in a sparsifying basis (DCT or DWT) and inverting the transform.

## Setup

```
pip install -r requirements.txt
pip install -e .
```

Installing with `-e .` makes `src` importable as a package (see `setup.py`), which all modules assume (`from src.compress_sensing import *`, etc.) — always run scripts from the repository root.

There is no test suite (`pytest` is listed in `requirements.txt` but no test files exist) and no lint config. Do not invent test/lint commands.

## Architecture

Core pipeline (all in `src/`):

- **`src/compress_sensing.py`** — the math core. For any observation method, an experiment generates a weight/sampling matrix `W` and observed values `y = W @ img`, then reconstructs the image by fitting `Lasso` on `y` against `W` transformed into a sparse basis, and inverting that transform.
  - `generate_pixel_observation` / `generate_gaussian_observation` / `generate_V1_observation` — build `W` (and derived `y`) for the three observation types. V1 weights come from the vendored `structured_random_features` package (see below).
  - `fourier_reconstruct` (DCT basis) / `wavelet_reconstruct` (DWT basis, via `pywt`) — fit LASSO in the transform domain and invert.
  - `reconstruct(W, y, ...)` — dispatches to one of the above based on `method='dct'|'dwt'`. Works on a single 2D (grayscale) patch.
  - `color_experiment` — runs `reconstruct` independently per RGB channel (reusing one `W` across channels when passed in).
  - `large_img_experiment` — the entry point for real (non-toy-sized) images. `reconstruct`/LASSO only works well on small patches, so this tiles the (optionally zero-padded) image into `filter_dim` blocks and reconstructs each block independently, optionally reusing the same weights (`fixed_weights=True`) across all blocks instead of drawing fresh random weights per block.
- **`src/utility.py`** — path/IO helpers shared by everything else. `search_root()` walks up parent directories looking for one named `Visual-Compressive-Sensing`, so all save/load path helpers only work correctly when the repo directory keeps that name. `data_save_path`/`fig_save_path` build the canonical `result/<method>/<image>/<observation>/...` and `figures/<method>/<image>/<observation>/...` layout (see below) and create directories as needed. `process_image` loads a file from `images/`.
- **`src/hyperparam_sweep_filter.py`** — CLI (`python -m src.hyperparam_sweep_filter ...` style, see `src/args.py` for flags) that runs a Dask-parallelized grid sweep of `large_img_experiment` over hyperparameters (alpha, num_cells, cell_size, sparse_freq, filter_dim, dwt level/type, repetitions) for one observation+method combo, and writes results as CSV under `result/` plus a matching hyperparameter-tracking `.txt` file.
- **`src/figure.py`** — turns swept CSV result data (or a single live reconstruction) into plots: `colorbar_live_reconst` (side-by-side reconstruction + pixel error heatmap for one parameter set), `error_vs_num_cell`, `error_vs_alpha`, `error_vs_filter_dim` (compare pixel/gaussian/V1 curves from swept CSVs, picking the best hyperparameters per x-value). Also runnable as a CLI (see `src/args.py::parse_figure_args`).
- **`src/args.py`** — all `argparse` wiring for the sweep and figure CLIs; the two CLIs share many flags but require different subsets depending on `method`/`observation`/`fig_type`.

### Data flow / on-disk layout

```
images/                                  # source images (input)
result/<method>/<image>/<observation>/   # hyperparam sweep CSVs + hyperparameter .txt logs (output of hyperparam_sweep_filter.py)
figures/<method>/<image>/<observation>/  # generated plots (output of figure.py)
```
`method` ∈ {`dct`, `dwt`}, `observation` ∈ {`pixel`, `gaussian`, `V1`} (V1 is upper-cased in paths, others lowercased).

### `structured_random_features/`

A vendored (not a git submodule) copy of an external research package providing the biological receptive-field model. `src/compress_sensing.py` only depends on `structured_random_features.src.models.weights.V1_weights`. Treat the rest of this directory as third-party code — avoid modifying it unless the task specifically concerns the V1 weight generation.

### `experiment/`, `notebooks/`, `matrix_experiments/`, `src/prototype_functions/`

Jupyter notebooks and standalone scripts used for exploratory analysis and generating paper figures. These are not part of the importable package, often duplicate/fork logic from `src/` at various points in time, and are not kept in sync with it — don't assume functions here match current `src/` signatures.

### `compress_sensing_example.md`

Stale usage walkthrough referencing an older API (`filter_reconstruct`, `color_reconstruct`, `generate_pixel_variables`, a `mode` string param) that no longer matches `src/compress_sensing.py`. Useful only for understanding intent/history, not as a working reference — prefer reading `src/compress_sensing.py` directly for current function names and signatures.
