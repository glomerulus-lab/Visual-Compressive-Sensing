# matrix_experiments

Exploratory scripts and paper-figure generators investigating *why* V1
receptive-field sampling tends to produce better sparse-coding reconstructions
than random pixel or Gaussian sampling. Where `src/` is the stable,
importable pipeline (generate observations → LASSO in a sparse basis →
reconstruct), everything here digs into the intermediate math — the design
matrix `theta` (`W` transformed into the DCT basis), its singular value
spectrum/principal components, column mutual coherence, and the sparsity of
the resulting DCT coefficient vectors — to compare V1 against pixel/Gaussian
sampling.

These are research/analysis scripts, not a library API: several duplicate
logic across files, most have hardcoded output filenames (re-running
overwrites previous figures), and many still contain commented-out
alternative calls left from earlier experiments. As with `src/`, everything
here imports `src` absolutely and uses single-dot relative imports to reach
its siblings, so scripts must be run as modules from the repo root, e.g.:

```
python -m matrix_experiments.plots.plot_metrics
python -m matrix_experiments.plots.paper_plots
```

## Shared library (`core.py` + `plots/`)

### `core.py`
The shared building block almost every other file in this directory imports
from (`from .core import *`, or `from ..core import *` for files inside
`plots/`). Formerly `plots/theta_exp_improved.py`. Provides:

- `generate_design_matrix` — projects a measurement matrix `W` into the DCT
  basis to get the design matrix `theta`.
- `compute_mutual_coherence` / `mutual_coherence_matrix` — max normalized
  dot product between design-matrix columns (mutual coherence), and running
  that over `n` repeated random draws for a given observation type.
- `dot_product_matrix` / `sort_design_matrix` / `high_freq_table` — build
  and inspect the full normalized column dot-product matrix, including
  mapping matrix entries back to `(kx, ky)` DCT frequency coordinates.
- `generate_coeff_vector` — DCT coefficients of an image/patch (the "true"
  sparse code).
- `generate_ctDc` / `generate_Dc` — coefficient-vector-weighted summaries of
  the dot-product matrix (`cᵀDc`, `Dc`), used to see how much column
  confusion actually affects reconstruction given the true coefficients.
- `MC_box_plot*`, `dot_product_matrix_from_patch`,
  `plot_dot_products_for_patch`, `dot_product_histogram*_for_patch(es)` —
  patch-level variants of the above, operating on patches from
  `barbara.bmp` via `extract_patches`.

## Single-image analysis scripts

These operate on the small, fixed 30×30 grayscale image
`images/tree_part1.jpg` at 100/300 observations. Each script now loads the
image itself via `process_image("tree_part1.jpg", color=False)` and defines
its own `SMALL_IMG`/`NUM_CELL_300`/`CELL_SIZE`/`BLOB_SIZE` constants inside
`main()` — `core.py` no longer exposes the image or these constants as
module-level globals.

- **`SVD_coeffs.py`** — a more ad hoc, script-style version of the same
  analysis at 300 observations only: compares true DCT coefficients against
  the design matrix's singular vectors/values, with scatter plots
  (`make_scatter`, `make_combined_scatter`), per-component error plots,
  reconstructed-image and PC-image panels, and coefficient sparsity
  histograms/CDFs. Largely superseded by `pc_plots.py` but kept as a
  standalone, more heavily-annotated run.
- **`SVD_dim.py`** — computes the "effective dimension" of `theta`
  (participation ratio `1 / Σpᵢ²` from the normalized singular value
  spectrum) for V1/Pixel/Gaussian over 20 runs and boxplots the result.
  Whole-image counterpart to `plots/SVD_dim_patches.py`.
- **`compare_reconstructions.py`** — reconstructs the image from V1
  observations with a fixed vs. a random receptive-field center (both at
  300 cells) and displays them side by side.
- **`coeffs_table.py`** — builds a pandas table mapping each dot-product
  matrix entry back to its `(kx, ky)` frequency pair and coherence value,
  to look for a relationship between low-coefficient frequencies and high
  dot products. Marked `Abandoned` in its own docstring.
- **`col_norms.py`** — `dot_product_matrix_mod` is a copy of `core`'s
  `dot_product_matrix` that also returns/plots the column norms of the
  design matrix, to test the hypothesis that V1's poor mutual coherence
  comes from near-zero-norm columns. (Corresponds to the untracked
  `Norms Hist.png` / `Dot_Hist.svg` outputs at the repo root.)
- **`add_epsilon.py`** — redefines `compute_mutual_coherence`/
  `mutual_coherence_matrix` (shadowing `core`'s versions of the same name)
  to add an `epsilon` fudge factor to column norms before normalizing,
  testing whether near-zero column norms are inflating V1's mutual
  coherence.
- **`less_than_epsilon.py`** — the complementary experiment: its own
  `compute_mutual_coherence(..., epsilon)` and `mutual_coherence_matrix_mod`
  instead filter out (or count) columns whose norm falls below `epsilon`,
  rather than regularizing them.

## Plotting scripts (`plots/`)

### Utilities
- **`plots/exp_constants.py`** — Constants shared by the patch-based scripts 
  (`IMAGE_FILE`, `PATCH_SIZE`,`PATCH_IDXS`, `N_OBS`, `ALPHA`, `CELL_SIZE`, 
  `BLOB_SIZE`, `NUM_RUNS`).
- **`plots/extract_patches.py`**:
  - `extract_patches(img, patch_size)` — tiles an image into non-overlapping
    square patches.
  - `show_patches_grid(patches, cols=16)` — renders a grid of patches and
    saves it to `grid_patches.svg`.
- **`plots/util.py`** — plotting helpers shared by `pc_plots.py` (whole-image)
  and `paper_plots.py` (per-patch): `pc_scatter_plots`,
  `plot_smoothed_error`, `cumsum_err`, `compare_smoothed_errors`,
  `plot_first_pc`, `plot_top_pcs`, `coeff_vectors_hist`, `coeff_vectors_cdf`.
  Each takes an optional `patch_idx` (default `None`) that adds patch
  labeling/filenames when set; PC-image reshaping is derived from the data
  itself (`sqrt` of vector length) rather than a hardcoded 30/32 patch size.

### Image-based
Works with full images or single patches

- **`plots/plot_coeffs.py`** — plots true vs. estimated DCT coefficients
  (log-scale) for V1, Pixel, and Gaussian reconstructions at 300 cells. It
  operates on the whole small image (not on patches).
- **`plots/plot_metrics.py`** — uses `core.generate_ctDc`/`generate_Dc` to
  boxplot coefficient-weighted dot-product metrics (`cᵀDc` with/without the
  diagonal replaced, `Dc` under the 1-norm and 2-norm) across V1/Pixel/
  Gaussian, on the whole small image. (Formerly `matrix_experiments/metrics.py`;
  it now loads its own image and constants rather than reading `core.py` globals.)
- **`plots/pc_plots.py`** — `compute_results(num_obs)` builds, for V1/Pixel/
  Gaussian, the design matrix, its SVD, a LASSO reconstruction, and the
  estimated vs. true coefficients projected onto the design matrix's
  principal components, at `num_obs` ∈ {100, 300}. Uses `plots/util.py`'s
  plotting helpers (`pc_scatter_plots`, `compare_smoothed_errors`,
  `plot_first_pc`, `plot_top_pcs`, `coeff_vectors_hist`, `coeff_vectors_cdf`)
  to turn that into PC scatter plots, per-component error curves, PCs
  rendered as images, and coefficient sparsity histograms/CDFs. Whole-image
  counterpart to `plots/paper_plots.py`, which shares the same
  plotting helpers.

### Patch-based
These tile `images/barbara.bmp` into 32×32 patches via `extract_patches`
and `exp_constants.PATCH_SIZE`/`PATCH_IDXS`, and repeat the single-image
analyses per patch.

- **`plots/paper_plots.py`** — the patch-based counterpart to
  `pc_plots.py`, and the most complete/current module in this directory
  (its name suggests it's what generates figures actually used in the
  paper). `compute_patch_results`/`get_results`/`run_selected_patches`
  compute the same SVD/PC/coefficient results as `pc_plots.py` but per
  patch, using the shared `plots/util.py` helpers (passing `patch_idx`) for
  the single-patch plots, plus its own `_all_patches` grid variants of every
  plot (PC scatter, PC-rank plots, cumulative error, PCs as images,
  coefficient histograms/CDFs) that only make sense across multiple patches.
- **`plots/SVD_dim_patches.py`** — patch-based version of `SVD_dim.py`:
  effective dimension of `theta` per patch over `NUM_RUNS` runs, boxplotted
  both across the 4 selected patches (grid) and for a single patch.
- **`plots/SVDplots.py`** — plots the raw singular value spectrum (not the
  participation-ratio dimension) of `theta` at 256 observations for each
  selected patch, again with grid and single-patch variants.

## Known quirks

- `add_epsilon.py` and `less_than_epsilon.py` locally redefine
  `compute_mutual_coherence` (and `add_epsilon.py` also redefines
  `mutual_coherence_matrix`). Because a module-level `def` always wins over
  a preceding `from .core import *`, these files use their *own* versions,
  not `core.py`'s, despite the wildcard import.
- `pc_plots.py` and `plots/paper_plots.py` used to independently
  define several identically-named functions (`pc_scatter_plots`,
  `plot_top_pcs`, `coeff_vectors_hist`, `coeff_vectors_cdf`, `plot_first_pc`,
  `compare_smoothed_errors`, `plot_smoothed_error`) with subtly different
  behavior. These are now consolidated in `plots/util.py`; `pc_plots.py`'s
  histogram/CDF output changed slightly as a result (percentile-based bins
  and log-scaled axes, matching what `paper_plots.py` already did,
  instead of `pc_plots.py`'s old max-value bins and linear axes).
