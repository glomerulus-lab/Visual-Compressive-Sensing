# %load_ext autoreload
# %autoreload 2

import numpy as np
import scipy
import cvxpy as cp
from matplotlib import pyplot as plt

from .plots.paper_plots import process_image, extract_patches, run_selected_patches
from .plots.exp_constants import *
from structured_random_features.src.models.weights import V1_covariance_matrix

dim = (PATCH_SIZE, PATCH_SIZE)
d = PATCH_SIZE * PATCH_SIZE
center = (PATCH_SIZE / 2, PATCH_SIZE / 2)

ALG = 'bp'
idx = 58
PATCH_IDXS = [idx]

# T = conv{e_1, ..., e_d}, the standard simplex. sup_{x in T} <a, x> = max_i a_i,
# so a plain vertex-enumeration estimate (np.max) is exact. The CVXPY estimate
# below solves the same maximization as an explicit LP per sample, as a
# cross-check that generalizes to convex sets without a closed-form max.
NUM_SAMPLES_VERTEX = 10000
NUM_SAMPLES_CVX = 1000

print(f"Computing V1 covariance for patch {idx} of {IMAGE_FILE}, cell_size={CELL_SIZE}, blob_size={BLOB_SIZE}")
barbara = process_image("barbara.bmp", color=False)
patches = extract_patches(barbara, PATCH_SIZE)
results = run_selected_patches(patches, PATCH_IDXS, center=center, algorithm=ALG)
zstar = results[idx]['coeffs_true'].flatten()

CV1 = V1_covariance_matrix(patches[idx].shape, CELL_SIZE, BLOB_SIZE, center=center)
L, Q = np.linalg.eigh(CV1)
order = np.argsort(L)[::-1]
L = L[order]
Q = Q[:, order]
Qpix = Q.copy()
Q = scipy.fft.dctn(Qpix.T.reshape(d, dim[0], dim[1]), norm='ortho', axes=[1, 2]).reshape(d, d).T
sqrtCV1 = (Q * np.sqrt(L)) @ Q.T
sqrtinv = (Q * (1 / np.sqrt(L))) @ Q.T


def gaussian_width_vertex(A, num_samples):
    """Estimate w(A T) for T = conv{e_1, ..., e_d} via the closed-form vertex max."""
    g = np.random.randn(A.shape[1], num_samples)
    return np.max(np.abs(A @ g), axis=0) # remove abs for simplex


def gaussian_width_cvx(A, num_samples):
    """Estimate w(A T) for T = conv{e_1, ..., e_d} by solving
    max_{x in T} <g, A x> with CVXPY for each sampled Gaussian vector g."""
    n = A.shape[1]
    x = cp.Variable(n)
    g_param = cp.Parameter(A.shape[0])
    # constraints = [x >= 0, cp.sum(x) == 1] # for simplex
    constraints = [cp.norm1(x) <= 1]
    objective = cp.Maximize(g_param @ (A @ x))
    problem = cp.Problem(objective, constraints)

    widths = np.zeros(num_samples)
    for i in range(num_samples):
        g_param.value = np.random.randn(A.shape[0])
        problem.solve()
        widths[i] = problem.value
    return widths


print("\nEstimating Gaussian width via vertex enumeration...")
w_T_vertex = gaussian_width_vertex(np.eye(d), NUM_SAMPLES_VERTEX)
w_CT_vertex = gaussian_width_vertex(sqrtCV1, NUM_SAMPLES_VERTEX)

print("Estimating Gaussian width via CVXPY...")
w_T_cvx = gaussian_width_cvx(np.eye(d), NUM_SAMPLES_CVX)
w_CT_cvx = gaussian_width_cvx(sqrtCV1, NUM_SAMPLES_CVX)

print(f"\nw(T):       vertex-enum mean = {w_T_vertex.mean():.4f}   CVXPY mean = {w_T_cvx.mean():.4f}")
print(f"w(C^0.5 T): vertex-enum mean = {w_CT_vertex.mean():.4f}   CVXPY mean = {w_CT_cvx.mean():.4f}")


def localized_gaussian_width_cvx(A, r_values, num_samples, inverse=False):
    """Estimate the localized Gaussian width w_r(A T) = w(A T \\cap r B) for a
    range of r, where T is the unit l1 ball and B is the unit l2 ball:
        w_r(A T) = E[sup_{x: ||x||_1 <= 1, ||x||_2 <= r} <g, Ax>]
    Solves one LP per (r, sample) with CVXPY, reusing the same compiled problem
    and the same num_samples draws of g across all r (so the resulting curve is
    monotone non-decreasing in r, since the feasible region only grows)."""
    n = A.shape[1]
    x = cp.Variable(n)
    g_param = cp.Parameter(A.shape[0])
    r_param = cp.Parameter(nonneg=True)
    if inverse:
        constraints = [cp.norm1(A @ x) <= 1, cp.norm2(x) <= r_param]
        objective = cp.Maximize(g_param @ x)
    else:
        constraints = [cp.norm1(x) <= 1, cp.norm2(x) <= r_param]
        objective = cp.Maximize(g_param @ (A @ x))
    problem = cp.Problem(objective, constraints)

    g_samples = np.random.randn(A.shape[0], num_samples)
    widths = np.zeros((len(r_values), num_samples))
    for j in range(num_samples):
        g_param.value = g_samples[:, j]
        for i, r in enumerate(r_values):
            r_param.value = r
            problem.solve()
            widths[i, j] = problem.value
    return widths.mean(axis=1)


R_VALUES = np.linspace(.5, 60.0, 10)
NUM_SAMPLES_LOCAL = 100

print("\nEstimating localized Gaussian width w_r(T) over a range of r...")
w_r_T = localized_gaussian_width_cvx(np.eye(d), R_VALUES, NUM_SAMPLES_LOCAL)
w_r_CT = localized_gaussian_width_cvx(sqrtCV1, R_VALUES, NUM_SAMPLES_LOCAL)
# w_r_T = localized_gaussian_width_cvx(np.eye(d), R_VALUES, NUM_SAMPLES_LOCAL, inverse=True)
# w_r_CT = localized_gaussian_width_cvx(sqrtinv, R_VALUES, NUM_SAMPLES_LOCAL, inverse=True)

plt.figure()
plt.plot(R_VALUES, w_r_T / R_VALUES, marker='o', label=r"$w_r(T) / r$")
plt.plot(R_VALUES, w_r_CT / R_VALUES, marker='o', label=r"$w_r(C^{0.5} T) / r$")
plt.xlabel("r")
plt.yscale('log')
plt.ylabel(r"$w_r(T)/r$")
plt.legend()
plt.title(f"Localized Gaussian width for patch {idx}")
plt.show()

plt.figure()
plt.hist(w_T_vertex, bins=50, alpha=0.5, density=True, label="w(T) vertex-enum")
plt.hist(w_CT_vertex, bins=50, alpha=0.5, density=True, label="w(C^0.5 T) vertex-enum")
plt.axvline(w_T_vertex.mean(), color='k', linestyle=':', label="w(T) vertex-enum mean")
plt.axvline(w_CT_vertex.mean(), color='r', linestyle=':', label="w(C^0.5 T) vertex-enum mean")
plt.axvline(w_T_cvx.mean(), color='k', linestyle='--', label="w(T) CVXPY mean")
plt.axvline(w_CT_cvx.mean(), color='r', linestyle='--', label="w(C^0.5 T) CVXPY mean")
plt.xlabel(r"$\sup_{x \in T} \langle g, x \rangle$")
plt.legend()
plt.title(f"Gaussian width estimates for patch {idx}")
plt.show()
