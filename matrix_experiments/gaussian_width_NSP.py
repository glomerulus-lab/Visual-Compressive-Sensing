"""Estimate the Gaussian width of the restricted-nullspace-property "bad set"

    C = {x in R^d : ||x_S||_1 >= rho * ||x_S^c||_1}

i.e. the region a nonzero nullspace vector must avoid for the restricted
nullspace property to hold on the support S (see plot_RNP.py for a 3D
picture of C when |S| = 1). C is a cone (positively homogeneous), so its
plain Gaussian width is infinite; we instead estimate the LOCALIZED width
at r = 1 (the unit ball):

    w(C) = E[ sup_{x in C, ||x||_2 <= 1} <g, x> ],   g ~ N(0, I_d)

C is NOT convex in general: ||x_S||_1 = max over sign patterns sigma in
{-1,1}^S of sigma^T x_S, so C is a union of 2^|S| convex "nappes", one per
sign pattern (each nappe {x : sigma^T x_S >= rho ||x_S^c||_1} is convex,
since it is a sublevel set of the convex function rho*||x_S^c||_1 -
sigma^T x_S). Directly enumerating all 2^|S| nappes to compute the sup is
exponential in |S|.

Instead, note that flipping the sign of ANY single x_i (i in S or i in S^c)
alone leaves both ||x_S||_1 and ||x_S^c||_1 -- and hence membership in C --
unchanged, since it only changes the (absolute value of a) single term.
So for a FIXED sampled g, the optimal x has sign(x_i) = sign(g_i) for every
i, since that is always at least as good for the objective <g,x> and never
hurts feasibility. Substituting u_i = |x_i| = sign(g_i) x_i >= 0 for every i
turns the defining inequality ||x_S||_1 >= rho ||x_S^c||_1 into the linear
inequality sum(u_S) >= rho * sum(u_S^c), and ||x||_2 = ||u||_2, so

    sup_{x in C, ||x||_2 <= 1} <g, x>  =  sup_{u in K, ||u||_2 <= 1} <|g|, u>

where |g| is the entrywise absolute value of g and K is the convex cone

    K = {u in R^d : u >= 0, sum(u_S) >= rho * sum(u_S^c)}

(the same family of cone as in plot_cone_3d.py, generalized to arbitrary
|S| and |S^c|). This equivalence was verified against brute-force
enumeration over the 2^|S| sign nappes. Each sample now costs a single LP
(linear objective, linear + one L2-ball constraint) over K, with no
enumeration over sign patterns needed regardless of |S|.
"""

import numpy as np
import scipy
import cvxpy as cp
from matplotlib import pyplot as plt
from structured_random_features.src.models.weights import V1_covariance_matrix
from .plots.exp_constants import CELL_SIZE, BLOB_SIZE, PATCH_SIZE


NUM_SAMPLES = 40
# PATCH_SIZE = 16
# CELL_SIZE = 30
# BLOB_SIZE = 6

s = 8

d = PATCH_SIZE * PATCH_SIZE
S = np.arange(s).astype(int)
rho = 0.5

# Covariance
CV1 = V1_covariance_matrix([PATCH_SIZE, PATCH_SIZE], CELL_SIZE, BLOB_SIZE, 
                            center=(PATCH_SIZE // 2, PATCH_SIZE // 2))
L, Q = np.linalg.eigh(CV1)
order = np.argsort(L)[::-1]
L = L[order]
Q = Q[:, order]
Qpix = Q.copy()
Q = scipy.fft.dctn(
    Qpix.T.reshape(d, PATCH_SIZE, PATCH_SIZE), norm='ortho', axes=[1, 2]
    ).reshape(d, d).T
cov = Q @ np.diag(L) @ Q.T
cov_sqrt = Q @ np.diag(np.sqrt(L)) @ Q.T

# d = 128
# s = 16
# spectrum = np.ones(d)
# #spectrum[:s] = 
# spectrum = np.arange(1, d+1) ** -2.
# cov = np.diag(spectrum) + 1e-6 * np.eye(d)

cov *= d / cov.trace()

width_bound = np.sqrt(2 * s * np.log(np.e * d / s))
print(f"uniform bound:" f" {width_bound:.4f}, dim: {width_bound ** 2:.1f}")



def gaussian_width_NSP_cvx(d, S, rho, num_samples, cov_sqrt):
    """Estimate w(C) = E[sup_{x in C, ||x||_2 <= 1} <g,x>] for
    C = {x : ||x_S||_1 >= rho ||x_S^c||_1} in R^d, by solving the equivalent
    problem sup_{u in K, ||u||_2 <= 1} <|g|, u> over the cone
    K = {u >= 0 : sum(u_S) >= rho * sum(u_S^c)} described above, once per
    sampled g. Returns (mean width, raw per-sample widths)."""
    S = sorted(S)
    Sc = sorted(set(range(d)) - set(S))

    u = cp.Variable(d, nonneg=True)
    absg = cp.Parameter(d, nonneg=True)

    constraints = [cp.sum(u[S]) >= rho * cp.sum(u[Sc]), cp.norm2(cov_sqrt @ u) <= 1]
    objective = cp.Maximize(absg @ u)
    problem = cp.Problem(objective, constraints)

    g_samples = cov_sqrt @ np.random.randn(d, num_samples)
    widths = np.zeros(num_samples)
    for j in range(num_samples):
        absg.value = np.abs(g_samples[:, j])
        problem.solve()
        widths[j] = problem.value
        print(f"Sample {j+1}/{num_samples}: width = {widths[j]:.4f}")
    return widths.mean(), widths


print(f"Estimating Gaussian width of the RNP bad set in R^{d}, S={S}, rho={rho}")
width_mean, width_samples = gaussian_width_NSP_cvx(d, S, rho, NUM_SAMPLES, cov)
print(f"\nw(C) estimate at r = 1: {width_mean:.4f}, dim: {width_mean ** 2:.1f}")
print(f"uniform bound:" f" {width_bound:.4f}, dim: {width_bound ** 2:.1f}")

plt.figure()
plt.hist(width_samples, bins=30, density=True)
plt.axvline(width_mean, color='k', linestyle=':', label="mean")
plt.xlabel(r"$\sup_{x \in C,\ \|x\|_2 \leq 1} \langle g, x \rangle$")
plt.legend()
plt.title(rf"Gaussian width of RNP bad set ($|S|={len(S)}$, $\rho$={rho}) in $\mathbb{{R}}^{{{d}}}$")
plt.show()
