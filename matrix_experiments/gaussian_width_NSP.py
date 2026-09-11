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

"""
Gaussian width of  T = Gamma ∩ B_2^d,   Gamma = { x : C^{-1/2} x in K },
K = { v : ||v_S||_1 >= rho ||v_Sc||_1 }   (the NSP / l1-descent cone).

w(T) = E_g sup_{x in T} <g,x>,  g ~ N(0,I_d).

Method
------
Gamma is a cone, so for each sample the support function is a projection:
    sup_{x in Gamma, ||x||<=1} <g,x> = ||Pi_Gamma(g)||_2.
K is nonconvex; it splits as K = U_eps K_eps over sign patterns eps in {+-1}^s,
with K_eps = { v : rho||v_Sc||_1 <= <eps, v_S> } convex.  For a fixed eps,
    dist(g, Gamma_eps)^2 = min_{v in K_eps} ||C^{1/2} v - g||^2
is a smooth QP over a one-constraint convex cone whose Euclidean projection has
a closed form (soft-threshold off S, shift by mu*eps on S, mu from an exact
O(d log d) breakpoint search).  Solved by batched FISTA over all Monte Carlo
samples at once.  eps is found by a sign fixed-point iteration with restarts.

Every eps visited yields a *certified lower bound*
    <g, x> / ||x||_2  with x = C^{1/2} v, v feasible,
so the estimator is conservative under inexact convergence and the running max
over restarts is always valid.
"""
# ---------------------------------------------------------------- projection
def _proj_K_eps(Y, eps, S, Sc, rho):
    """Batched Euclidean projection of columns of Y (d x N) onto
    K_eps = {v : rho||v_Sc||_1 <= <eps, v_S>}.  eps is (s x N) of +-1."""
    d, N = Y.shape
    s, m = len(S), len(Sc)
    Ys, Yc = Y[S, :], Y[Sc, :]
    c = np.einsum('ij,ij->j', eps, Ys)          # <eps, y_S>
    Tm = np.abs(Yc)
    viol = rho * Tm.sum(0) - c

    mu = np.zeros(N)
    act = viol > 0                               # infeasible columns need mu>0
    if act.any():
        Ta = Tm[:, act]
        c_a = c[act]
        Ts = -np.sort(-Ta, axis=0)               # descending, m x na
        CS = np.vstack([np.zeros(Ts.shape[1]), np.cumsum(Ts, axis=0)])   # (m+1) x na
        k = np.arange(m + 1)[:, None]
        mu_k = (rho * CS - c_a[None, :]) / (k * rho**2 + s)
        lo = np.vstack([Ts, np.zeros(Ts.shape[1])]) / rho                # (m+1) x na
        hi = np.vstack([np.full((1, Ts.shape[1]), np.inf), Ts]) / rho
        ok = (mu_k >= lo - 1e-12) & (mu_k <= hi + 1e-12) & (mu_k >= 0)
        idx = np.argmax(ok, axis=0)                                       # first valid k
        mu[act] = np.maximum(mu_k[idx, np.arange(idx.size)], 0.0)

    V = np.empty_like(Y)
    V[S, :] = Ys + mu[None, :] * eps
    V[Sc, :] = np.sign(Yc) * np.maximum(Tm - rho * mu[None, :], 0.0)
    return V


# ---------------------------------------------------------------- inner solve
def _fista(G, Csqrt, C, eps, S, Sc, rho, Lip, iters, tol=1e-7, check=10):
    """min_{v in K_eps} 0.5||C^{1/2}v - g||^2, batched over columns of G."""
    B = Csqrt @ G                                # C^{1/2} g, the linear term
    V = _proj_K_eps(B, eps, S, Sc, rho)          # warm start
    Z, t = V.copy(), 1.0
    for it in range(iters):
        Vn = _proj_K_eps(Z - (C @ Z - B) / Lip, eps, S, Sc, rho)
        tn = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
        Z = Vn + ((t - 1) / tn) * (Vn - V)
        if it % check == check - 1:
            if np.linalg.norm(Vn - V) <= tol * max(np.linalg.norm(Vn), 1.0):
                return Vn
        V, t = Vn, tn
    return V


def _value(V, G, Csqrt):
    """Certified per-sample lower bound <g,x>/||x||, x = C^{1/2}v, clipped at 0."""
    X = Csqrt @ V
    nx = np.linalg.norm(X, axis=0)
    num = np.einsum('ij,ij->j', G, X)
    return np.where(nx > 1e-12, num / np.maximum(nx, 1e-12), 0.0).clip(min=0.0)


# ---------------------------------------------------------------- driver
def gaussian_width(d, S, rho, cov, num_samples=400, restarts=4,
                   iters=250, sign_rounds=4, batch=None, seed=0, verbose=True,
                   max_enum=256):
    S = np.asarray(sorted(S), dtype=int)
    Sc = np.setdiff1d(np.arange(d), S)
    s = len(S)

    lam, Q = np.linalg.eigh(cov)
    lam = np.maximum(lam, 0.0)
    Csqrt = (Q * np.sqrt(lam)) @ Q.T
    Cisqrt = (Q * np.where(lam > 0, 1/np.sqrt(np.maximum(lam, 1e-300)), 0)) @ Q.T
    Cm = (Q * lam) @ Q.T
    Lip = lam.max()

    rng = np.random.default_rng(seed)
    batch = batch or num_samples
    vals, ub = [], []

    for start in range(0, num_samples, batch):
        n = min(batch, num_samples - start)
        G = rng.standard_normal((d, n))
        best = np.zeros(n)

        if 2**s <= max_enum:                              # exact: enumerate all eps
            for bits in range(2**s):
                pat = np.array([1.0 if (bits >> i) & 1 else -1.0 for i in range(s)])
                E = np.repeat(pat[:, None], n, axis=1)
                V = _fista(G, Csqrt, Cm, E, S, Sc, rho, Lip, iters)
                best = np.maximum(best, _value(V, G, Csqrt))
        else:                                             # heuristic: fixed point
            for r in range(restarts):
                if r == 0:                                # greedy init
                    E = np.sign((Cisqrt @ G)[S, :])
                else:                                     # perturb the greedy init
                    E = np.sign((Cisqrt @ G)[S, :])
                    flip = rng.random((s, n)) < 0.3
                    E = np.where(flip, -E, E)
                E[E == 0] = 1.0
                for _ in range(sign_rounds):              # sign fixed point
                    V = _fista(G, Csqrt, Cm, E, S, Sc, rho, Lip, iters)
                    best = np.maximum(best, _value(V, G, Csqrt))
                    En = np.sign(V[S, :])
                    En[En == 0] = E[En == 0]
                    if np.array_equal(En, E):
                        break
                    E = En
        vals.append(best)

        kap = (1 + 1/rho) * np.sqrt(s) / np.sqrt(max(lam.min(), 1e-300))
        ub.append(np.minimum(np.linalg.norm(G, axis=0),
                             kap * np.abs(Csqrt @ G).max(0)))

    v = np.concatenate(vals)
    se = v.std(ddof=1) / np.sqrt(len(v))
    if verbose:
        print(f"w = {v.mean():.4f} +/- {se:.4f}   "
              f"[ceiling {np.concatenate(ub).mean():.3f}]   "
              f"stat.dim ~ {(v**2).mean():.2f}")
    return v.mean(), se, v


print(f"Estimating Gaussian width of the RNP bad set in R^{d}, S={S}, rho={rho}")
width_mean, width_samples = gaussian_width(d, S, rho, cov, NUM_SAMPLES)
print(f"\nw(C) estimate at r = 1: {width_mean:.4f}, dim: {width_mean ** 2:.1f}")
print(f"uniform bound:" f" {width_bound:.4f}, dim: {width_bound ** 2:.1f}")

plt.figure()
plt.hist(width_samples, bins=30, density=True)
plt.axvline(width_mean, color='k', linestyle=':', label="mean")
plt.xlabel(r"$\sup_{x \in C,\ \|x\|_2 \leq 1} \langle g, x \rangle$")
plt.legend()
plt.title(rf"Gaussian width of RNP bad set ($|S|={len(S)}$, $\rho$={rho}) in $\mathbb{{R}}^{{{d}}}$")
plt.show()
