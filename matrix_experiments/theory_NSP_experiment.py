"""
Does the Gaussian-width statistical dimension predict the BP recovery threshold?

Setup
-----
Take image patch 58 of barbara (the "good" patch used in `test_theory`), keep
only its top-`SPARSITY` DCT coefficients, and push that *exactly sparse* signal
back into pixel space.  That patch is then run through the ordinary compressed
sensing pathway (`run_selected_patches`, basis pursuit) for each of the three
sensing matrices (Pixel / Gaussian / V1) over a sweep of observation counts.

Because the signal is exactly s-sparse, basis pursuit either recovers it to
machine precision or it does not -- there is a sharp phase transition in the
number of observations m.  Conic integral geometry (Amelunxen et al. 2014) says
that transition sits at the statistical dimension of the descent cone

    C = { x : ||x_S||_1 >= rho ||x_Sc||_1 },   rho = 1,

measured under the covariance of the rows of the design matrix Theta:

    delta(C) ~= w(C)^2,    m* ~= delta(C).

We compute w(C) three ways, all with S = the true top-`SPARSITY` support:
  * isotropic     -- identity covariance      (matches the Gaussian sensing matrix)
  * V1 diagonal   -- diag covariance whose spectrum is that of the V1 covariance
  * V1 full       -- the full V1 covariance rotated into the DCT basis
The first two use the analytic statistical-dimension bound (`diag_analytic`),
the last uses the Monte Carlo estimator (`gaussian_width`).

Run with
    python -m matrix_experiments.theory_NSP_experiment            # prototype (fast)
    python -m matrix_experiments.theory_NSP_experiment --full     # full run
"""

import argparse

import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import dctn, idctn

from structured_random_features.src.models.weights import V1_covariance_matrix

from .plots.paper_plots import process_image, extract_patches, run_selected_patches
from .plots.exp_constants import CELL_SIZE, BLOB_SIZE, PATCH_SIZE, IMAGE_FILE
from .gaussian_width import diag_analytic_NSP, full_NSP, statdim_descent_cone

# Local copies of the small helpers in `test_theory`; that module is a script
# that runs its whole experiment at import time, so it cannot be imported here.
WIDTH = 74
LABEL_W = 34


def top_k_approx(vector, s):
    vector_s = np.zeros_like(vector)
    topS = np.argsort(np.abs(vector))[::-1][:s]
    vector_s[topS] = vector[topS]
    return vector_s


def header(title):
    print()
    print("=" * WIDTH)
    print(f"  {title}")
    print("=" * WIDTH)


def section(title):
    print()
    print(f"-- {title} " + "-" * max(0, WIDTH - len(title) - 4))


def kv(label, value, note=""):
    note_str = f"   ({note})" if note else ""
    print(f"  {label:<{LABEL_W}}{value:>16}{note_str}")


def num(x, fmt=".4e"):
    return format(x, fmt)


PATCH_IDX = 58
SPARSITY = 30
N_OBS_LIST = [40, 50, 60, 70, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260]
RHO = 1.0                 # l1 descent cone / NSP constant for exact recovery
ALG = 'bp'
METHODS = ['Pixel', 'Gaussian', 'V1']
REL_ERR_TOL = 1e-4        # ||z - z*||_2 / ||z*||_2 below this counts as recovered


# ---------------------------------------------------------------- data
def make_sparse_patch(patch, s):
    """Top-s DCT approximation of `patch`, returned as (pixel patch, DCT coeffs)."""
    coeffs = dctn(patch, norm='ortho', axes=[0, 1])
    coeffs_s = top_k_approx(coeffs.flatten(), s).reshape(coeffs.shape)
    return idctn(coeffs_s, norm='ortho', axes=[0, 1]), coeffs_s


# ---------------------------------------------------------------- empirical
def run_sweep(sparse_patch, zstar, n_obs_list, trials, center, algorithm=ALG):
    """Success indicator of BP recovery per method, per m, per trial.

    Returns dict[method] -> (len(n_obs_list), trials) boolean array, plus the
    matching relative errors.
    """
    znorm = np.linalg.norm(zstar)
    success = {m: np.zeros((len(n_obs_list), trials), dtype=bool) for m in METHODS}
    rel_err = {m: np.zeros((len(n_obs_list), trials)) for m in METHODS}

    for i, n in enumerate(n_obs_list):
        for t in range(trials):
            # run_selected_patches expects a list of patches; ours has exactly one
            res = run_selected_patches([sparse_patch], [0], center=center,
                                       algorithm=algorithm, n=n)[0]
            for method in METHODS:
                z = res[method]['est_coeffs'].flatten()
                e = np.linalg.norm(z - zstar) / znorm
                rel_err[method][i, t] = e
                success[method][i, t] = e < REL_ERR_TOL
        rates = "  ".join(f"{m}={success[m][i].mean():.2f}" for m in METHODS)
        print(f"  m={n:>4}   success rate:  {rates}")

    return success, rel_err


def empirical_threshold(n_obs_list, success_rates, level=0.5):
    """Smallest m where the success rate crosses `level`, linearly interpolated."""
    n_obs = np.asarray(n_obs_list, dtype=float)
    r = np.asarray(success_rates, dtype=float)
    above = np.nonzero(r >= level)[0]
    if above.size == 0:
        return np.nan
    k = above[0]
    if k == 0:
        return n_obs[0]
    r0, r1 = r[k - 1], r[k]
    if r1 == r0:
        return n_obs[k]
    return n_obs[k - 1] + (level - r0) * (n_obs[k] - n_obs[k - 1]) / (r1 - r0)


# ---------------------------------------------------------------- theory
def v1_covariance_dct(patch_shape, cell_size, blob_size, center):
    """V1 covariance expressed in the DCT basis, plus its spectrum.

    The rows of the V1 measurement matrix W are (approximately) N(0, CV1) in
    pixel space; the design matrix Theta is the row-wise DCT of W, so its rows
    have covariance D CV1 D^T with D the orthonormal DCT.  Applying D to the
    eigenvectors of CV1 gives that rotated covariance directly.
    """
    d = patch_shape[0] * patch_shape[1]
    CV1 = V1_covariance_matrix(patch_shape, cell_size, blob_size, center=center)
    L, Q = np.linalg.eigh(CV1)
    order = np.argsort(L)[::-1]
    L = np.maximum(L[order], 0.0)
    Qpix = Q[:, order]
    Qdct = dctn(Qpix.T.reshape(d, patch_shape[0], patch_shape[1]),
                norm='ortho', axes=[1, 2]).reshape(d, d).T
    cov = (Qdct * L) @ Qdct.T

    # Two diagonal surrogates, BOTH indexed by DCT coordinate so they can be
    # combined with a support S that lives in DCT coordinates:
    #   var  - the per-coordinate variances diag(cov); same marginals as V1,
    #          correlations discarded.
    #   spec - the actual V1 spectrum L, assigned to DCT coordinates in order
    #          of their variance rank, so the diagonal matrix has exactly V1's
    #          eigenvalues.  (L on its own is indexed by eigenvalue rank, NOT
    #          by DCT coordinate -- pairing it directly with a DCT support
    #          mixes two different bases.)
    var = np.diag(cov).copy()
    rank = np.argsort(np.argsort(var)[::-1])   # rank[j] = variance rank of coord j
    spec = L[rank]
    return cov, var, spec

def compute_theory(d, S, rho, cov_var, cov_spec, cov_V1,
                   num_samples, restarts, iters, sign_rounds):
    """Statistical dimension predictions for the covariance models.

    `cov_var` and `cov_spec` must be indexed by DCT coordinate, matching S.
    """
    preds = {}

    w_iso, _ = diag_analytic_NSP(d, S, rho, np.ones(d))
    preds['isotropic'] = {'width': w_iso, 'statdim': w_iso ** 2, 'se': 0.0}

    # diag_analytic_NSP is scale invariant in c, but guard against exact zeros
    w_spec, _ = diag_analytic_NSP(d, S, rho, np.maximum(cov_spec, cov_spec.max() * 1e-12))
    preds['V1 diag (spectrum)'] = {'width': w_spec, 'statdim': w_spec ** 2, 'se': 0.0}

    w_var, _ = diag_analytic_NSP(d, S, rho, np.maximum(cov_var, cov_var.max() * 1e-12))
    preds['V1 diag (variances)'] = {'width': w_var, 'statdim': w_var ** 2, 'se': 0.0}

    w_full, se_full, samples = full_NSP(
        d, S, rho, cov_V1, num_samples=num_samples, restarts=restarts,
        iters=iters, sign_rounds=sign_rounds, verbose=False)
    # E[w^2] is the statistical dimension; w^2 is a (slightly low) proxy
    preds['V1 full'] = {'width': w_full, 'statdim': (samples ** 2).mean(),
                        'se': se_full}
    return preds


# ---------------------------------------------------------------- plotting
def plot_sparse_patch(patch, sparse_patch, s, filename):
    """True patch next to its top-s DCT approximation, plus the difference.

    This is the signal actually fed to the CS pathway: the sweep measures
    recovery of `sparse_patch`, not of the original patch, so it is worth
    seeing how much of the patch survives keeping only s coefficients.
    """
    diff = sparse_patch - patch
    rel = np.linalg.norm(diff) / np.linalg.norm(patch)

    # share a grayscale range between the two patches so they are comparable
    vmin = min(patch.min(), sparse_patch.min())
    vmax = max(patch.max(), sparse_patch.max())

    fig, ax = plt.subplots(1, 3, figsize=(11, 4), constrained_layout=True)
    im0 = ax[0].imshow(patch, cmap='gray', vmin=vmin, vmax=vmax)
    ax[0].set_title(f"true patch {PATCH_IDX}")
    fig.colorbar(im0, ax=ax[0], shrink=0.8)

    im1 = ax[1].imshow(sparse_patch, cmap='gray', vmin=vmin, vmax=vmax)
    ax[1].set_title(f"top-{s} DCT approximation")
    fig.colorbar(im1, ax=ax[1], shrink=0.8)

    lim = np.abs(diff).max()
    im2 = ax[2].imshow(diff, cmap='RdBu_r', vmin=-lim, vmax=lim)
    ax[2].set_title(f"difference (rel. $\\ell_2$ = {rel:.3f})")
    fig.colorbar(im2, ax=ax[2], shrink=0.8)

    for a in ax:
        a.axis('off')
    fig.suptitle(f"Sparse ground truth fed to the CS pathway "
                 f"($s$={s}, $d$={patch.size})")
    fig.savefig(filename, format='svg')
    print(f"saved {filename}")


def plot_results(n_obs_list, success, preds, filename):
    colors = {'Pixel': '#FF6F00', 'Gaussian': '#43A047', 'V1': '#2196F3'}
    theory_style = {'isotropic': ('#43A047', '--'),
                    'V1 diag (spectrum)': ('#1565C0', ':'),
                    'V1 diag (variances)': ('#7E57C2', ':'),
                    'V1 full': ('#2196F3', '-.'),
                    'descent cone': ('#757575', '--')}

    plt.figure(figsize=(8, 5))
    for method in METHODS:
        plt.plot(n_obs_list, success[method].mean(axis=1), 'o-',
                 color=colors[method], label=f"{method} (empirical)")
    for name, (color, ls) in theory_style.items():
        delta = preds[name]['statdim']
        plt.axvline(delta, color=color, linestyle=ls,
                    label=rf"$\delta$ {name} = {delta:.0f}")

    plt.xlabel("number of observations $m$")
    plt.ylabel("P(exact recovery)")
    plt.ylim(-0.05, 1.05)
    plt.title(f"BP recovery vs. statistical dimension "
              f"(patch {PATCH_IDX}, $s$={SPARSITY}, $d$={PATCH_SIZE**2})")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, format='svg')
    print(f"\nsaved {filename}")


# ---------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--full', action='store_true',
                    help="full run (more trials and Monte Carlo samples)")
    ap.add_argument('--trials', type=int, default=None,
                    help="BP trials per observation count")
    ap.add_argument('--num-samples', type=int, default=None,
                    help="Monte Carlo samples for the full-covariance width")
    ap.add_argument('--n-obs', type=int, nargs='+', default=None,
                    help="observation counts to sweep (default: %s)" % N_OBS_LIST)
    ap.add_argument('--output', default='theory_NSP_experiment.svg')
    ap.add_argument('--patch-output', default='theory_NSP_sparse_patch.svg',
                    help="where to save the true vs. sparse patch comparison")
    args = ap.parse_args()

    # Prototype defaults are deliberately cheap; --full is the real run.
    trials = args.trials if args.trials is not None else (20 if args.full else 3)
    num_samples = args.num_samples if args.num_samples is not None else (
        400 if args.full else 40)
    restarts, iters, sign_rounds = (4, 250, 4) if args.full else (2, 120, 2)
    n_obs_list = args.n_obs if args.n_obs is not None else N_OBS_LIST

    dim = (PATCH_SIZE, PATCH_SIZE)
    d = PATCH_SIZE * PATCH_SIZE
    center = (PATCH_SIZE / 2, PATCH_SIZE / 2)

    header("Run configuration")
    kv("mode", "full" if args.full else "prototype")
    kv("algorithm", ALG)
    kv("image", IMAGE_FILE)
    kv("patch index", PATCH_IDX)
    kv("patch size", f"{PATCH_SIZE} x {PATCH_SIZE}")
    kv("sparsity s", SPARSITY)
    kv("observations (N_OBS)", str(n_obs_list))
    kv("trials per m", trials)
    kv("rho", num(RHO))
    kv("MC samples (full cov width)", num_samples)

    # --- sparse ground truth ------------------------------------------------
    img = process_image(IMAGE_FILE, color=False)
    patches = extract_patches(img, PATCH_SIZE)
    sparse_patch, coeffs_s = make_sparse_patch(patches[PATCH_IDX], SPARSITY)
    zstar = coeffs_s.flatten()
    S = np.nonzero(np.abs(zstar) > 0)[0]

    header(f"Sparse ground truth (patch {PATCH_IDX})")
    kv("||z*||_0", f"{S.size} / {d}")
    kv("||z*||_2", num(np.linalg.norm(zstar)))
    kv("||z*||_1", num(np.linalg.norm(zstar, 1)))
    z_full = dctn(patches[PATCH_IDX], norm='ortho', axes=[0, 1]).flatten()
    kv("energy kept", f"{100 * np.linalg.norm(zstar) / np.linalg.norm(z_full):.2f}%")

    plot_sparse_patch(patches[PATCH_IDX], sparse_patch, SPARSITY, args.patch_output)

    # --- theory -------------------------------------------------------------
    header("Gaussian width / statistical dimension")
    cov_V1, cov_var, cov_spec = v1_covariance_dct(dim, CELL_SIZE, BLOB_SIZE, center)
    preds = compute_theory(d, S, RHO, cov_var, cov_spec, cov_V1,
                           num_samples, restarts, iters, sign_rounds)

    preds['descent cone'] = {
        'statdim': statdim_descent_cone(d, S.size), 'width': np.nan, 'se': 0.0}

    unif = 2 * SPARSITY * np.log(d / SPARSITY) + (5 / 4) * SPARSITY
    section("Predicted thresholds")
    for name, p in preds.items():
        note = "" if np.isnan(p['width']) else (
            f"w = {p['width']:.3f}" + (f" +/- {p['se']:.3f}" if p['se'] else ""))
        kv(f"delta ({name})", f"{p['statdim']:.1f}", note)
    kv("uniform bound (Chandrasekaran)", f"{unif:.1f}")

    # --- experiment ---------------------------------------------------------
    header("Basis pursuit sweep")
    success, rel_err = run_sweep(sparse_patch, zstar, n_obs_list, trials, center)

    header("Empirical vs. predicted threshold")
    print(f"  {'method':<12}{'m* (50% recovery)':>20}")
    print("  " + "-" * 32)
    for method in METHODS:
        m_star = empirical_threshold(n_obs_list, success[method].mean(axis=1))
        kv(method, f"{m_star:.1f}" if np.isfinite(m_star) else "not reached")

    section("Median relative coefficient error")
    print(f"  {'m':>6}" + "".join(f"{m:>14}" for m in METHODS))
    for i, n in enumerate(n_obs_list):
        row = "".join(f"{np.median(rel_err[m][i]):>14.3e}" for m in METHODS)
        print(f"  {n:>6}" + row)

    plot_results(n_obs_list, success, preds, args.output)


if __name__ == "__main__":
    main()
