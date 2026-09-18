# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Research codebase for compressed sensing (CS) image reconstruction. It compares three "observation" (sampling) techniques — random pixel selection, Gaussian random projections, and a biologically-inspired V1 (primary visual cortex) receptive-field model — to see which produces the best sparse-coding reconstruction of an image from a small number of samples. Reconstruction is done by fitting a LASSO model in a sparsifying basis (DCT or DWT) and inverting the transform.

## Setup

- All Python dependencies **must be installed, synchronized, and locked** using uv
- Never use pip, pip-tools, poetry, or conda directly for dependency management

Use these commands:

- Install dependencies: `uv add <package>`
- Remove dependencies: `uv remove <package>`
- Sync environment: `uv sync`
- Lock dependencies: `uv lock`

The development environment is already installed within `.venv/`.
This was setup by running `uv venv; uv pip install -r requirements.txt`.

## Architecture

Core pipeline (all in `src/`):

- **`src/compress_sensing.py`** — the math core. For any observation method, an experiment generates a weight/sampling matrix `W` and observed values `y = W @ img`, then reconstructs the image by fitting `Lasso` on `y` against `W` transformed into a sparse basis, and inverting that transform.
  - `generate_pixel_observation` / `generate_gaussian_observation` / `generate_V1_observation` — build `W` (and derived `y`) for the three observation types. V1 weights come from the vendored `structured_random_features` package (see below).
  - `fourier_reconstruct` (DCT basis) / `wavelet_reconstruct` (DWT basis, via `pywt`) — fit a sparse solver in the transform domain and invert. `fourier_reconstruct` takes `algorithm` ∈ {`lasso`, `ridge`, `omp`, `bp`} (`bp` = exact basis pursuit via `scipy.optimize.linprog`, no alpha); `wavelet_reconstruct` is LASSO-only.
  - `reconstruct(W, y, ...)` — dispatches to one of the above based on `method='dct'|'dwt'`. Works on a single 2D (grayscale) patch.
  - `color_experiment` — runs `reconstruct` independently per RGB channel (reusing one `W` across channels when passed in).
  - `large_img_experiment` — the entry point for real (non-toy-sized) images. `reconstruct`/LASSO only works well on small patches, so this tiles the (optionally zero-padded) image into `filter_dim` blocks and reconstructs each block independently, optionally reusing the same weights (`fixed_weights=True`) across all blocks instead of drawing fresh random weights per block.
- **`src/utility.py`** — path/IO helpers shared by everything else. `search_root()` walks up parent directories looking for one named `Visual-Compressive-Sensing`, so all save/load path helpers only work correctly when the repo directory keeps that name. `data_save_path`/`fig_save_path` build the canonical `result/<method>/<image>/<observation>/...` and `figures/<method>/<image>/<observation>/...` layout (see below) and create directories as needed. `process_image` loads a file from `images/`.
- **`src/hyperparam_sweep_filter.py`** — CLI (`python -m src.hyperparam_sweep_filter ...` style, see `src/args.py` for flags) that runs a Dask-parallelized grid sweep of `large_img_experiment` over hyperparameters (alpha, num_cells, cell_size, sparse_freq, filter_dim, dwt level/type, repetitions) for one observation+method+algorithm combo, and writes results as CSV under `result/` plus a matching hyperparameter-tracking `.txt` file. `-algorithm` selects the solver (`lasso` default, plus `ridge`/`omp`/`bp`); `bp`/`omp` have no alpha penalty, so they reject `-alpha_list` and record `alp` as empty, and only `lasso` is wired into the `dwt` method.
- **`src/figure.py`** — turns swept CSV result data (or a single live reconstruction) into plots: `colorbar_live_reconst` (side-by-side reconstruction + pixel error heatmap for one parameter set), `error_vs_num_cell`, `error_vs_alpha`, `error_vs_filter_dim` (compare pixel/gaussian/V1 curves from swept CSVs, picking the best hyperparameters per x-value). Every one takes `algorithm` (default `lasso`) to pick the `result/<algorithm>/` tree. Note the groupbys pass `dropna=False` deliberately: an unpenalised solver's `alp` column is empty, and the default would silently discard every such row. Also runnable as a CLI (see `src/args.py::parse_figure_args`).

- **`sweep_figures.py`** (repo root) — builds the paper figures for one solver from its consolidated sweep CSVs; `lasso_sweep_figures.py` and `bp_sweep_figures.py` are thin drivers over it (`python lasso_sweep_figures.py`, `python bp_sweep_figures.py`). `varying_alpha` is offered only for penalised solvers. Output defaults to `figures/<algorithm>/paper_new/`.

- **`lasso_dct_sweep.sh` / `bp_dct_sweep.sh`** (repo root) — the 36-run DCT sweep for each solver, same grids apart from alpha. `consolidate_results.py` (`--algorithm`) merges the per-run CSVs each produces into the consolidated `{V1,Pixel,Gaussian}.csv` the figure code reads.
- **`src/args.py`** — all `argparse` wiring for the sweep and figure CLIs; the two CLIs share many flags but require different subsets depending on `method`/`observation`/`fig_type`.

### Data flow / on-disk layout

```
images/                                               # source images (input)
result/<algorithm>/<method>/<image>/<observation>/    # hyperparam sweep CSVs + hyperparameter .txt logs (output of hyperparam_sweep_filter.py)
figures/<algorithm>/<method>/<image>/<observation>/   # generated plots (output of figure.py)
figures/<algorithm>/paper_new/                        # paper figures (output of <algorithm>_sweep_figures.py)
```
`algorithm` ∈ {`lasso`, `bp`, ...}, `method` ∈ {`dct`, `dwt`}, `observation` ∈ {`pixel`, `gaussian`, `V1`} (V1 is upper-cased in paths, others lowercased).

The `<algorithm>` level keeps each solver's sweeps and figures in their own subtree. `data_save_path`/`fig_save_path`/`load_dataframe*` and the `-algorithm` flag on both CLIs all default it to `lasso`, which is what every result predating basis pursuit was produced with. Other top-level `figures/` directories (`paper/`, `paper_plots_256/`, `plot_pics/`, ...) predate the split and are not algorithm-scoped.

### `structured_random_features/`

A vendored (not a git submodule) copy of an external research package providing the biological receptive-field model. `src/compress_sensing.py` only depends on `structured_random_features.src.models.weights.V1_weights`. Treat the rest of this directory as third-party code — avoid modifying it unless the task specifically concerns the V1 weight generation.

### `experiment/`, `notebooks/`, `matrix_experiments/`, `src/prototype_functions/`

Jupyter notebooks and standalone scripts used for exploratory analysis and generating paper figures. These are not part of the importable package, often duplicate/fork logic from `src/` at various points in time, and are not kept in sync with it — don't assume functions here match current `src/` signatures.

`matrix_experiments/` and `matrix_experiments/plots/` are real packages (each has an `__init__.py`) that import `src` absolutely and their own siblings via single-dot relative imports (e.g. `from .plots.theta_exp_improved import *`, `from .extract_patches import *`). Because of the relative imports, these scripts must be run as modules from the repo root — e.g. `python -m matrix_experiments.add_epsilon`, `python -m matrix_experiments.plots.theta_exp_improved` — not as `python matrix_experiments/add_epsilon.py`, which fails with `ModuleNotFoundError` since direct script execution never puts the repo root on `sys.path` or gives the module a package context. This mirrors how `src/hyperparam_sweep_filter.py` and `src/figure.py` are already documented to run (`python -m src.hyperparam_sweep_filter`).

### `compress_sensing_example.md`

Stale usage walkthrough referencing an older API (`filter_reconstruct`, `color_reconstruct`, `generate_pixel_variables`, a `mode` string param) that no longer matches `src/compress_sensing.py`. Useful only for understanding intent/history, not as a working reference — prefer reading `src/compress_sensing.py` directly for current function names and signatures.
