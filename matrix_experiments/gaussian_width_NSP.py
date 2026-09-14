import numpy as np
from matplotlib import pyplot as plt

from scipy.fft import dctn
from structured_random_features.src.models.weights import V1_covariance_matrix

from .plots.exp_constants import CELL_SIZE, BLOB_SIZE, PATCH_SIZE

from .gaussian_width import *

NUM_SAMPLES = 200
PATCH_SIZE = 32
CELL_SIZE = 50
BLOB_SIZE = 6

s = 40

d = PATCH_SIZE * PATCH_SIZE
S = np.arange(s).astype(int)
rho = 0.2

# Covariance
# CV1 = V1_covariance_matrix([PATCH_SIZE, PATCH_SIZE], CELL_SIZE, BLOB_SIZE, 
#                             center=(PATCH_SIZE // 2, PATCH_SIZE // 2))
# L, Q = np.linalg.eigh(CV1)
# order = np.argsort(L)[::-1]
# L = L[order]
# Q = Q[:, order]
# Qpix = Q.copy()
# Q = dctn(
#     Qpix.T.reshape(d, PATCH_SIZE, PATCH_SIZE), norm='ortho', axes=[1, 2]
#     ).reshape(d, d).T
# cov = Q @ np.diag(L) @ Q.T
# cov_sqrt = Q @ np.diag(np.sqrt(L)) @ Q.T

# d = 128
# s = 16
spectrum = np.ones(d)
#spectrum[:s] = 
spectrum = np.arange(1, d+1) ** -1.5
cov = np.diag(spectrum) + 1e-6 * np.eye(d)

cov *= d / cov.trace()

width_bound = np.sqrt(2 * s * np.log(d / s) + (5/4) * s) # Chandra
print(f"uniform bound:" f" {width_bound:.4f}, dim: {width_bound ** 2:.1f}")

print(f"Estimating Gaussian width of the RNP bad set in R^{d}, S={S}, rho={rho}")
# width_mean, width_se, width_samples = gaussian_width(d, S, rho, cov, NUM_SAMPLES)
width_mean, width_se, width_samples = diag_sampling(d, S, rho, np.diag(cov), NUM_SAMPLES)
print(f"\nw(C) estimate at r = 1: {width_mean:.4f}, dim: {width_mean ** 2:.1f}")
width_analytic, _ = diag_analytic(d, S, rho, np.diag(cov))
print(f"analytic bound: {width_analytic:.4f}, dim: {width_analytic ** 2:.1f}")
print(f"uniform bound:" f" {width_bound:.4f}, dim: {width_bound ** 2:.1f}")

plt.figure()
plt.hist(width_samples, bins=30, density=True)
plt.axvline(width_mean, color='k', linestyle=':', label="mean")
plt.xlabel(r"$\sup_{x \in C,\ \|x\|_2 \leq 1} \langle g, x \rangle$")
plt.legend()
plt.title(rf"Gaussian width of RNP bad set ($|S|={len(S)}$, $\rho$={rho}) in $\mathbb{{R}}^{{{d}}}$")
plt.show()
