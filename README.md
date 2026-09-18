# Signal Reconstruction Research

Description: Compressed sensing [(CS)](https://github.com/qkstngus000/Compress-Sensing) is a widely accepted theory for signal-based data recovery using trained samples, which enables the reconstruction of a full signal at a low cost. Typically, pixel-based sampling techniques are utilized, but in this study, we propose a new observation technique using a biological model, the Visual Cortex (V1), to acquire more robust and improved signal reconstruction. The V1 is a biological model that has been shown to be effective in processing visual information in the brain. We hypothesize that this biological approach to observation combined with LASSO sparse coding prediction could lead to a better minimization of reconstruction errors compared to the traditional mathematical approach. Thus, we compare the effectiveness of the signal reconstruction between the V1 model and two classical models, namely pixel selection and Gaussian, to determine which method performs best. This study aims to highlight the potential of utilizing biological models in CS and to provide a better understanding of the performance of different observation techniques.

## Table of Contents
1. [Installation](#installation)
2. [Organization](#organization)
3. [Data flow / on-disk layout](#data-flow--on-disk-layout)
4. [Usage](#usage)
5. [Reproducing the paper figures](#reproducing-the-paper-figures)

## Installation

The development environment lives in `.venv/` and was created with
[uv](https://docs.astral.sh/uv/):

    uv venv
    uv pip install -r requirements.txt

Dependencies must be managed with uv (`uv add` / `uv remove` / `uv sync` /
`uv lock`) -- not pip, poetry or conda.

To use the libraries in `src/` as a package, run this in the root directory of
the project:

    pip install -e .

Note that `src/utility.py::search_root()` locates the project by walking up
parent directories looking for one named `Visual-Compressive-Sensing`, so every
save/load path helper only works when the checkout keeps that directory name.

## Organization

### src
All importable source code lives here.

- **`compress_sensing.py`** -- the math core. For a given observation method it
  builds a weight/sampling matrix `W`, takes the observations `y = W @ img`, and
  reconstructs the image by fitting a sparse solver on `y` against `W`
  transformed into a sparsifying basis, then inverting that transform.
  - `generate_pixel_observation` / `generate_gaussian_observation` /
    `generate_V1_observation` -- build `W` (and `y`) for the three observation
    types. V1 weights come from the vendored `structured_random_features`
    package.
  - `fourier_reconstruct` (DCT basis) / `wavelet_reconstruct` (DWT basis, via
    `pywt`) -- fit in the transform domain and invert. `fourier_reconstruct`
    takes `algorithm` in {`lasso`, `ridge`, `omp`, `bp`} (`bp` = exact basis
    pursuit via `scipy.optimize.linprog`, which has no alpha);
    `wavelet_reconstruct` is LASSO-only.
  - `reconstruct(W, y, ...)` -- dispatches on `method='dct'|'dwt'` for a single
    2D (grayscale) patch.
  - `color_experiment` -- runs `reconstruct` per RGB channel, optionally reusing
    one `W` across channels.
  - `large_img_experiment` -- the entry point for real (non-toy-sized) images.
    Sparse recovery only works well on small patches, so this tiles the
    (optionally zero-padded) image into `filter_dim` blocks and reconstructs each
    block independently, optionally reusing the same weights across all blocks
    (`fixed_weights=True`).
- **`hyperparam_sweep_filter.py`** -- CLI that runs a Dask-parallelized grid
  sweep of `large_img_experiment` over hyperparameters (alpha, num_cells,
  cell_size, sparse_freq, filter_dim, DWT level/type, repetitions) for one
  observation + method + algorithm combination, writing results as CSV under
  `result/` plus a matching hyperparameter-tracking `.txt` log. This is one file
  now, covering both DCT and DWT (selected with `-method`).
- **`utility.py`** -- path/IO helpers shared by everything else: `search_root`,
  `data_save_path` / `fig_save_path` (which build the canonical directory
  layout and create directories as needed), `process_image`, and the dataframe
  loading/aggregation helpers used by the figure code.
- **`figure.py`** -- turns swept CSV data (or a single live reconstruction) into
  plots: `colorbar_live_reconst` (reconstruction plus pixel-error heatmap for
  one parameter set), `error_vs_num_cell`, `error_vs_alpha`,
  `error_vs_filter_dim` (compare pixel/gaussian/V1 curves from swept CSVs,
  picking the best hyperparameters per x-value). Each takes `algorithm`
  (default `lasso`) to select the `result/<algorithm>/` tree. Also runnable as a
  CLI.
- **`sweep_figures.py`** -- builds the paper figures for one solver from its
  consolidated sweep CSVs; the root-level `lasso_sweep_figures.py` and
  `bp_sweep_figures.py` are thin drivers over it.
- **`args.py`** -- all `argparse` wiring for the sweep and figure CLIs.

### result
Where all hyperparameter-swept data is stored. Each sweep invocation writes a
timestamped per-run CSV plus a hyperparameter `.txt` log; `consolidate_results.py`
merges those into the consolidated `{V1,Pixel,Gaussian}.csv` files the figure
code reads.

### figures
Where figures generated from the result data are stored, mirroring the `result/`
layout, plus `figures/<algorithm>/paper_new/` for the paper figures.

### structured_random_features
A vendored (not a submodule) copy of an external research package providing the
biological receptive-field model. Only
`structured_random_features.src.models.weights.V1_weights` is used; treat the
rest as third-party code.

### experiment, notebooks, matrix_experiments, src/prototype_functions
Jupyter notebooks and standalone scripts used for exploratory analysis. They are
not part of the importable package, often fork logic from `src/` at various
points in time, and are not kept in sync with it.

`matrix_experiments/` and `matrix_experiments/plots/` are real packages that
import `src` absolutely and their siblings relatively, so they must be run as
modules from the repo root:

    python -m matrix_experiments.add_epsilon
    python -m matrix_experiments.plots.theta_exp_improved

## Data flow / on-disk layout

```
images/                                               # source images (input)
result/<algorithm>/<method>/<image>/<observation>/    # sweep CSVs + hyperparameter .txt logs
figures/<algorithm>/<method>/<image>/<observation>/   # generated plots
figures/<algorithm>/paper_new/                        # paper figures
```

`algorithm` in {`lasso`, `bp`, `ridge`, `omp`}, `method` in {`dct`, `dwt`},
`observation` in {`pixel`, `gaussian`, `V1`} (V1 is upper-cased in paths, the
others lowercased).

The `<algorithm>` level keeps each solver's sweeps and figures in their own
subtree. The path helpers and the `-algorithm` flag on both CLIs default it to
`lasso`, which is what every result predating basis pursuit was produced with.
Other top-level `figures/` directories (`paper/`, `paper_plots_256/`,
`plot_pics/`, ...) predate the split and are not algorithm-scoped.

Note on filenames: the `True_`/`False_` prefix on per-run CSVs is `color`, not
`fixed_weights`.

## Usage

### Hyperparameter sweep

    python -m src.hyperparam_sweep_filter \
        -img_name boat.png -method dct -observation V1 -num_reps 10 \
        -alpha_list 0.01 0.1 1 10 -num_cells 32 56 80 104 128 -filter_dim 16 \
        -cell_size 50 100 200 -sparse_freq 2 4 6 8

`-algorithm` selects the solver (`lasso` by default, plus `ridge`, `omp`, `bp`).
`bp` and `omp` have no alpha penalty, so they reject `-alpha_list` and record an
empty `alp` column; only `lasso` is wired into the `dwt` method. Add `-color` for
RGB images. See `src/args.py` for the full flag list.

### Consolidating sweep results

    python consolidate_results.py --dry-run
    python consolidate_results.py --merge
    python consolidate_results.py --algorithm bp --merge

By default this only builds consolidated files where none exist yet; `--merge`
unions existing and per-run rows, and `--rebuild` is lossy (the consolidated
files already in the repo are not reproducible from the per-run CSVs beside
them). A timestamped `.bak` is always written before overwriting.

### Figures

    python -m src.figure -fig_type num_cell -img_name boat.png -method dct \
        -pixel_file Pixel.csv -gaussian_file Gaussian.csv -v1_file V1.csv -save

    python -m src.figure -fig_type colorbar -img_name boat.png -method dct \
        -observation V1 -num_cells 128 -filter_dim 32 -cell_size 100 \
        -sparse_freq 4 -alpha 0.1 -algorithm lasso

`-fig_type` is one of `colorbar`, `num_cell`, `alpha`, `filter_dim`. The three
curve plots need the consolidated `-pixel_file`/`-gaussian_file`/`-v1_file`
CSVs; `colorbar` runs a live reconstruction instead and needs the parameters for
that one run (V1 additionally requires `-cell_size` and `-sparse_freq`, and
`dwt` requires `-dwt_type` and `-level`). Without `-save` the figure is shown
rather than written.

### compress_sensing library examples

`compress_sensing_example.md` is a stale walkthrough referencing an older API
(`filter_reconstruct`, `color_reconstruct`, `generate_pixel_variables`, a `mode`
string parameter) that no longer matches `src/compress_sensing.py`. It is useful
for intent and history only -- read `src/compress_sensing.py` for the current
function names and signatures.

## Reproducing the paper figures

The 36-run DCT sweep for each solver is committed as a driver script (4 images x
3 observations x 3 num_cells waves):

    ./lasso_dct_sweep.sh          # DRY_RUN=1 to print without running
    ./bp_dct_sweep.sh             # NUM_REPS=2 for a quick trial

The two scripts use the same grids apart from alpha. They append to `result/`
rather than overwriting it. Then consolidate and build the figures:

    python consolidate_results.py --algorithm lasso --merge
    python lasso_sweep_figures.py

    python consolidate_results.py --algorithm bp --merge
    python bp_sweep_figures.py

Output defaults to `figures/<algorithm>/paper_new/`. `varying_alpha` is produced
only for penalised solvers, so it exists for LASSO but not for basis pursuit.
