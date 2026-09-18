#!/usr/bin/env bash
# Reproduce every hyperparameter sweep behind result/dct/.
#
# These 36 invocations were INFERRED from the files in result/dct/ -- the
# original sweep was driven by hand from a shell, so no driver was ever
# committed.  Reconstructed from:
#   * the directory layout   result/dct/<image>/<observation>/
#   * the grids recorded in  {True,False}_hyperparam.txt
#   * the num_cell/filter_dim pairing in the consolidated {V1,Pixel,Gaussian}.csv
#   * the save-name logic in src/hyperparam_sweep_filter.py (run_sweep) and
#     src/utility.py (data_save_path)
#
# 4 images x 3 observations x 3 num_cells waves = 36 runs.
# Every command below was verified to parse through src/args.py.
#
# READING THE FILENAMES
#   The True_/False_ prefix on the outputs is `color`, NOT `fixed_weights`:
#   run_sweep builds the name as f'{color}_param_'.  baboon/fruits are RGB and
#   were swept with -color; barbara/boat are grayscale and were not.
#
#   `rep` is not a flag -- run_sweep hardcodes rep = np.arange(10), which is why
#   every log shows [0 1 2 3 4 5 6 7 8 9].
#
#   -filter_dim scaled with num_cells:  8-32 -> 8,  32-128 -> 16,  128-512 -> 32.
#   The .txt logs only record filter_dim for the third wave (the parameter was
#   added partway through), but the consolidated CSVs pair num_cell with
#   filter_dim exactly this way in all 12 image/observation directories.
#
# ONE FLAG IS UNRECOVERABLE
#   -fixed_weights is not a swept column, so it never reaches the hyperparam log,
#   and it does not affect the filenames.  It is omitted here (argparse default
#   False).  If the original V1 runs used fixed weights, that flag is missing
#   from all 12 V1 commands.
#
# WARNING: running this APPENDS to result/dct/.  Each run writes a new
# timestamped {color}_param_<ctime>.csv and appends to {color}_hyperparam.txt,
# so existing per-run data is not overwritten, but the directories accumulate.
# It does NOT touch the consolidated {V1,Pixel,Gaussian}.csv files -- run
# consolidate_results.py --merge for that.
# Set DRY_RUN=1 to print the commands without executing them.
#
# Usage:   ./reproduce_dct_sweep.sh
#          DRY_RUN=1 ./reproduce_dct_sweep.sh
#          PYTHON=.venv/bin/python ./reproduce_dct_sweep.sh

set -euo pipefail
cd "$(dirname "$0")"

run() {
    echo "+ $*"
    if [ "${DRY_RUN:-0}" != "1" ]; then
        "$@"
    fi
}

PY="${PYTHON:-python}"


# ---------------------------------------------------------------- baboon.png  [color (RGB)]
# pixel, num_cells 8-32, filter_dim 8
run "$PY" -m src.hyperparam_sweep_filter \
    -img_name baboon.png -method dct -observation pixel -alpha_list 0.01 0.1 1 10 -num_cells 8 14 20 26 32 -filter_dim 8 -color
# pixel, num_cells 32-128, filter_dim 16
run "$PY" -m src.hyperparam_sweep_filter \
    -img_name baboon.png -method dct -observation pixel -alpha_list 0.01 0.1 1 10 -num_cells 32 56 80 104 128 -filter_dim 16 -color
# pixel, num_cells 128-512, filter_dim 32
run "$PY" -m src.hyperparam_sweep_filter \
    -img_name baboon.png -method dct -observation pixel -alpha_list 0.01 0.1 1 10 -num_cells 128 224 320 416 512 -filter_dim 32 -color
# gaussian, num_cells 8-32, filter_dim 8
run "$PY" -m src.hyperparam_sweep_filter \
    -img_name baboon.png -method dct -observation gaussian -alpha_list 0.01 0.1 1 10 -num_cells 8 14 20 26 32 -filter_dim 8 -color
# gaussian, num_cells 32-128, filter_dim 16
run "$PY" -m src.hyperparam_sweep_filter \
    -img_name baboon.png -method dct -observation gaussian -alpha_list 0.01 0.1 1 10 -num_cells 32 56 80 104 128 -filter_dim 16 -color
# gaussian, num_cells 128-512, filter_dim 32
run "$PY" -m src.hyperparam_sweep_filter \
    -img_name baboon.png -method dct -observation gaussian -alpha_list 0.01 0.1 1 10 -num_cells 128 224 320 416 512 -filter_dim 32 -color
# V1, num_cells 8-32, filter_dim 8
run "$PY" -m src.hyperparam_sweep_filter \
    -img_name baboon.png -method dct -observation V1 -alpha_list 0.01 0.1 1 10 -num_cells 8 14 20 26 32 -filter_dim 8 -cell_size 50 100 200 -sparse_freq 2 4 6 8 -color
# V1, num_cells 32-128, filter_dim 16
run "$PY" -m src.hyperparam_sweep_filter \
    -img_name baboon.png -method dct -observation V1 -alpha_list 0.01 0.1 1 10 -num_cells 32 56 80 104 128 -filter_dim 16 -cell_size 50 100 200 -sparse_freq 2 4 6 8 -color
# V1, num_cells 128-512, filter_dim 32
run "$PY" -m src.hyperparam_sweep_filter \
    -img_name baboon.png -method dct -observation V1 -alpha_list 0.01 0.1 1 10 -num_cells 128 224 320 416 512 -filter_dim 32 -cell_size 50 100 200 -sparse_freq 2 4 6 8 -color

# ---------------------------------------------------------------- barbara.bmp  [grayscale]
# pixel, num_cells 8-32, filter_dim 8
run "$PY" -m src.hyperparam_sweep_filter \
    -img_name barbara.bmp -method dct -observation pixel -alpha_list 0.01 0.1 1 10 -num_cells 8 14 20 26 32 -filter_dim 8
# pixel, num_cells 32-128, filter_dim 16
run "$PY" -m src.hyperparam_sweep_filter \
    -img_name barbara.bmp -method dct -observation pixel -alpha_list 0.01 0.1 1 10 -num_cells 32 56 80 104 128 -filter_dim 16
# pixel, num_cells 128-512, filter_dim 32
run "$PY" -m src.hyperparam_sweep_filter \
    -img_name barbara.bmp -method dct -observation pixel -alpha_list 0.01 0.1 1 10 -num_cells 128 224 320 416 512 -filter_dim 32
# gaussian, num_cells 8-32, filter_dim 8
run "$PY" -m src.hyperparam_sweep_filter \
    -img_name barbara.bmp -method dct -observation gaussian -alpha_list 0.01 0.1 1 10 -num_cells 8 14 20 26 32 -filter_dim 8
# gaussian, num_cells 32-128, filter_dim 16
run "$PY" -m src.hyperparam_sweep_filter \
    -img_name barbara.bmp -method dct -observation gaussian -alpha_list 0.01 0.1 1 10 -num_cells 32 56 80 104 128 -filter_dim 16
# gaussian, num_cells 128-512, filter_dim 32
run "$PY" -m src.hyperparam_sweep_filter \
    -img_name barbara.bmp -method dct -observation gaussian -alpha_list 0.01 0.1 1 10 -num_cells 128 224 320 416 512 -filter_dim 32
# V1, num_cells 8-32, filter_dim 8
run "$PY" -m src.hyperparam_sweep_filter \
    -img_name barbara.bmp -method dct -observation V1 -alpha_list 0.01 0.1 1 10 -num_cells 8 14 20 26 32 -filter_dim 8 -cell_size 50 100 200 -sparse_freq 2 4 6 8
# V1, num_cells 32-128, filter_dim 16
run "$PY" -m src.hyperparam_sweep_filter \
    -img_name barbara.bmp -method dct -observation V1 -alpha_list 0.01 0.1 1 10 -num_cells 32 56 80 104 128 -filter_dim 16 -cell_size 50 100 200 -sparse_freq 2 4 6 8
# V1, num_cells 128-512, filter_dim 32
run "$PY" -m src.hyperparam_sweep_filter \
    -img_name barbara.bmp -method dct -observation V1 -alpha_list 0.01 0.1 1 10 -num_cells 128 224 320 416 512 -filter_dim 32 -cell_size 50 100 200 -sparse_freq 2 4 6 8

# ---------------------------------------------------------------- boat.png  [grayscale]
# pixel, num_cells 8-32, filter_dim 8
run "$PY" -m src.hyperparam_sweep_filter \
    -img_name boat.png -method dct -observation pixel -alpha_list 0.01 0.1 1 10 -num_cells 8 14 20 26 32 -filter_dim 8
# pixel, num_cells 32-128, filter_dim 16
run "$PY" -m src.hyperparam_sweep_filter \
    -img_name boat.png -method dct -observation pixel -alpha_list 0.01 0.1 1 10 -num_cells 32 56 80 104 128 -filter_dim 16
# pixel, num_cells 128-512, filter_dim 32
run "$PY" -m src.hyperparam_sweep_filter \
    -img_name boat.png -method dct -observation pixel -alpha_list 0.01 0.1 1 10 -num_cells 128 224 320 416 512 -filter_dim 32
# gaussian, num_cells 8-32, filter_dim 8
run "$PY" -m src.hyperparam_sweep_filter \
    -img_name boat.png -method dct -observation gaussian -alpha_list 0.01 0.1 1 10 -num_cells 8 14 20 26 32 -filter_dim 8
# gaussian, num_cells 32-128, filter_dim 16
run "$PY" -m src.hyperparam_sweep_filter \
    -img_name boat.png -method dct -observation gaussian -alpha_list 0.01 0.1 1 10 -num_cells 32 56 80 104 128 -filter_dim 16
# gaussian, num_cells 128-512, filter_dim 32
run "$PY" -m src.hyperparam_sweep_filter \
    -img_name boat.png -method dct -observation gaussian -alpha_list 0.01 0.1 1 10 -num_cells 128 224 320 416 512 -filter_dim 32
# V1, num_cells 8-32, filter_dim 8
run "$PY" -m src.hyperparam_sweep_filter \
    -img_name boat.png -method dct -observation V1 -alpha_list 0.01 0.1 1 10 -num_cells 8 14 20 26 32 -filter_dim 8 -cell_size 50 100 200 -sparse_freq 2 4 6 8
# V1, num_cells 32-128, filter_dim 16
run "$PY" -m src.hyperparam_sweep_filter \
    -img_name boat.png -method dct -observation V1 -alpha_list 0.01 0.1 1 10 -num_cells 32 56 80 104 128 -filter_dim 16 -cell_size 50 100 200 -sparse_freq 2 4 6 8
# V1, num_cells 128-512, filter_dim 32
run "$PY" -m src.hyperparam_sweep_filter \
    -img_name boat.png -method dct -observation V1 -alpha_list 0.01 0.1 1 10 -num_cells 128 224 320 416 512 -filter_dim 32 -cell_size 50 100 200 -sparse_freq 2 4 6 8

# ---------------------------------------------------------------- fruits.png  [color (RGB)]
# pixel, num_cells 8-32, filter_dim 8
run "$PY" -m src.hyperparam_sweep_filter \
    -img_name fruits.png -method dct -observation pixel -alpha_list 0.01 0.1 1 10 -num_cells 8 14 20 26 32 -filter_dim 8 -color
# pixel, num_cells 32-128, filter_dim 16
run "$PY" -m src.hyperparam_sweep_filter \
    -img_name fruits.png -method dct -observation pixel -alpha_list 0.01 0.1 1 10 -num_cells 32 56 80 104 128 -filter_dim 16 -color
# pixel, num_cells 128-512, filter_dim 32
run "$PY" -m src.hyperparam_sweep_filter \
    -img_name fruits.png -method dct -observation pixel -alpha_list 0.01 0.1 1 10 -num_cells 128 224 320 416 512 -filter_dim 32 -color
# gaussian, num_cells 8-32, filter_dim 8
run "$PY" -m src.hyperparam_sweep_filter \
    -img_name fruits.png -method dct -observation gaussian -alpha_list 0.01 0.1 1 10 -num_cells 8 14 20 26 32 -filter_dim 8 -color
# gaussian, num_cells 32-128, filter_dim 16
run "$PY" -m src.hyperparam_sweep_filter \
    -img_name fruits.png -method dct -observation gaussian -alpha_list 0.01 0.1 1 10 -num_cells 32 56 80 104 128 -filter_dim 16 -color
# gaussian, num_cells 128-512, filter_dim 32
run "$PY" -m src.hyperparam_sweep_filter \
    -img_name fruits.png -method dct -observation gaussian -alpha_list 0.01 0.1 1 10 -num_cells 128 224 320 416 512 -filter_dim 32 -color
# V1, num_cells 8-32, filter_dim 8
run "$PY" -m src.hyperparam_sweep_filter \
    -img_name fruits.png -method dct -observation V1 -alpha_list 0.01 0.1 1 10 -num_cells 8 14 20 26 32 -filter_dim 8 -cell_size 50 100 200 -sparse_freq 2 4 6 8 -color
# V1, num_cells 32-128, filter_dim 16
run "$PY" -m src.hyperparam_sweep_filter \
    -img_name fruits.png -method dct -observation V1 -alpha_list 0.01 0.1 1 10 -num_cells 32 56 80 104 128 -filter_dim 16 -cell_size 50 100 200 -sparse_freq 2 4 6 8 -color
# V1, num_cells 128-512, filter_dim 32
run "$PY" -m src.hyperparam_sweep_filter \
    -img_name fruits.png -method dct -observation V1 -alpha_list 0.01 0.1 1 10 -num_cells 128 224 320 416 512 -filter_dim 32 -cell_size 50 100 200 -sparse_freq 2 4 6 8 -color

echo "done: 36 sweeps"
