#!/usr/bin/env bash
# Basis-pursuit counterpart to lasso_dct_sweep.sh.
#
# Same 36 invocations (4 images x 3 observations x 3 num_cells waves), same
# num_cells/filter_dim pairing, same color flags -- the one difference is the
# solver:  -algorithm bp  instead of the default lasso.
#
# NO ALPHA SWEEP
#   Basis pursuit minimises ||s||_1 subject to theta @ s == y exactly; there is
#   no penalty to trade off, so there is no alpha.  The CLI rejects
#   -alpha_list together with -algorithm bp and the grid holds a single
#   alpha of None, so the output CSVs keep their `alp` column but leave it
#   blank.  That makes each non-V1 run 4x smaller than the lasso equivalent
#   (which swept 0.01 0.1 1 10) and each V1 run 48 -> 12 combinations per
#   num_cell.
#
# OUTPUT PATHS
#   Each solver owns a subtree, so these land in
#   result/bp/dct/<image>/<observation>/ and cannot collide with the LASSO
#   sweeps in result/lasso/dct/.  Filenames are the same shape in both:
#   {color}_param_<ctime>.csv plus a {color}_hyperparam.txt log.
#
#   The True_/False_ part of the name is `color`, NOT `fixed_weights`:
#   baboon/peppers are RGB and are swept with -color; barbara/boat are not.
#
# COST
#   BP solves a linear program with 2 * filter_dim^2 variables per patch, which
#   is far slower than a LASSO fit -- the filter_dim 32 waves dominate the
#   runtime.  Use NUM_REPS=1 (or DRY_RUN=1) before committing to the full 10.
#
# WARNING: running this APPENDS to result/bp/dct/.  Each run writes a new
# timestamped {color}_param_<ctime>.csv and appends to
# {color}_hyperparam.txt.
# Set DRY_RUN=1 to print the commands without executing them.
#
# Usage:   ./bp_dct_sweep.sh
#          DRY_RUN=1 ./bp_dct_sweep.sh
#          NUM_REPS=2 ./bp_dct_sweep.sh          # quick trial, not the real sweep
#          PYTHON=.venv/bin/python ./bp_dct_sweep.sh

set -euo pipefail
cd "$(dirname "$0")"

run() {
    echo "+ $*"
    if [ "${DRY_RUN:-0}" != "1" ]; then
        "$@"
    fi
}

PY="${PYTHON:-python}"
NUM_REPS="${NUM_REPS:-10}"   # 10 = what the lasso sweep used

# One image x observation x num_cells wave.  Any extra args (-color, the V1
# -cell_size/-sparse_freq grids) are forwarded verbatim.
sweep() {
    local img="$1" obs="$2" cells="$3" filter_dim="$4"; shift 4
    run "$PY" -m src.hyperparam_sweep_filter \
        -img_name "$img" -method dct -algorithm bp -num_reps "$NUM_REPS" \
        -observation "$obs" -num_cells $cells -filter_dim "$filter_dim" "$@"
}

# num_cells waves, paired with the filter_dim each was swept at in the lasso run
WAVE1="8 14 20 26 32";        DIM1=8
WAVE2="32 56 80 104 128";     DIM2=16
WAVE3="128 224 320 416 512";  DIM3=32

# V1's own grid, unchanged from the lasso sweep
#V1_GRID=(-cell_size 50 100 200 -sparse_freq 2 4 6 8)
V1_GRID=(-cell_size 50 -sparse_freq 2 4 6 8)

# ---------------------------------------------------------------- mandril.bmp  [color (RGB)]
sweep mandril.bmp pixel    "$WAVE1" $DIM1 -color
sweep mandril.bmp pixel    "$WAVE2" $DIM2 -color
sweep mandril.bmp pixel    "$WAVE3" $DIM3 -color
sweep mandril.bmp gaussian "$WAVE1" $DIM1 -color
sweep mandril.bmp gaussian "$WAVE2" $DIM2 -color
sweep mandril.bmp gaussian "$WAVE3" $DIM3 -color
sweep mandril.bmp V1       "$WAVE1" $DIM1 "${V1_GRID[@]}" -color
sweep mandril.bmp V1       "$WAVE2" $DIM2 "${V1_GRID[@]}" -color

# ---------------------------------------------------------------- barbara.bmp  [grayscale]
sweep barbara.bmp pixel    "$WAVE1" $DIM1
sweep barbara.bmp pixel    "$WAVE2" $DIM2
sweep barbara.bmp pixel    "$WAVE3" $DIM3
sweep barbara.bmp gaussian "$WAVE1" $DIM1
sweep barbara.bmp gaussian "$WAVE2" $DIM2
sweep barbara.bmp gaussian "$WAVE3" $DIM3
sweep barbara.bmp V1       "$WAVE1" $DIM1 "${V1_GRID[@]}"
sweep barbara.bmp V1       "$WAVE2" $DIM2 "${V1_GRID[@]}"

# ---------------------------------------------------------------- boat.bmp  [grayscale]
sweep boat.bmp pixel    "$WAVE1" $DIM1
sweep boat.bmp pixel    "$WAVE2" $DIM2
sweep boat.bmp pixel    "$WAVE3" $DIM3
sweep boat.bmp gaussian "$WAVE1" $DIM1
sweep boat.bmp gaussian "$WAVE2" $DIM2
sweep boat.bmp gaussian "$WAVE3" $DIM3
sweep boat.bmp V1       "$WAVE1" $DIM1 "${V1_GRID[@]}"
sweep boat.bmp V1       "$WAVE2" $DIM2 "${V1_GRID[@]}"

# ---------------------------------------------------------------- peppers.bmp  [color (RGB)]
sweep peppers.bmp pixel    "$WAVE1" $DIM1 -color
sweep peppers.bmp pixel    "$WAVE2" $DIM2 -color
sweep peppers.bmp pixel    "$WAVE3" $DIM3 -color
sweep peppers.bmp gaussian "$WAVE1" $DIM1 -color
sweep peppers.bmp gaussian "$WAVE2" $DIM2 -color
sweep peppers.bmp gaussian "$WAVE3" $DIM3 -color
sweep peppers.bmp V1       "$WAVE1" $DIM1 "${V1_GRID[@]}" -color
sweep peppers.bmp V1       "$WAVE2" $DIM2 "${V1_GRID[@]}" -color


# 32 x 32, V1 - run last
NUM_REPS=1
for i in {1..10}; do
    echo "V1 run $i of 10"
    sweep mandril.bmp V1 "$WAVE3" $DIM3 "${V1_GRID[@]}" -color
    sweep barbara.bmp V1 "$WAVE3" $DIM3 "${V1_GRID[@]}"
    sweep boat.bmp V1    "$WAVE3" $DIM3 "${V1_GRID[@]}"
    sweep peppers.bmp V1 "$WAVE3" $DIM3 "${V1_GRID[@]}" -color
done

echo "done: 36 sweeps"
