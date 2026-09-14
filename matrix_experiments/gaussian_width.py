import numpy as np
import cvxpy as cp

from scipy.special import erfc
from scipy.optimize import minimize_scalar

# def gaussian_width_NSP_cvx(d, S, rho, num_samples, cov_sqrt):
#     """Estimate w(C) = E[sup_{x in C, ||x||_2 <= 1} <g,x>] for
#     C = {x : ||x_S||_1 >= rho ||x_S^c||_1} in R^d, by solving the equivalent
#     problem sup_{u in K, ||u||_2 <= 1} <|g|, u> over the cone
#     K = {u >= 0 : sum(u_S) >= rho * sum(u_S^c)} described above, once per
#     sampled g. Returns (mean width, raw per-sample widths)."""
#     S = sorted(S)
#     Sc = sorted(set(range(d)) - set(S))

#     u = cp.Variable(d, nonneg=True)
#     absg = cp.Parameter(d, nonneg=True)

#     constraints = [cp.sum(u[S]) >= rho * cp.sum(u[Sc]), cp.norm2(cov_sqrt @ u) <= 1]
#     objective = cp.Maximize(absg @ u)
#     problem = cp.Problem(objective, constraints)

#     g_samples = cov_sqrt @ np.random.randn(d, num_samples)
#     widths = np.zeros(num_samples)
#     for j in range(num_samples):
#         absg.value = np.abs(g_samples[:, j])
#         problem.solve()
#         widths[j] = problem.value
#         print(f"Sample {j+1}/{num_samples}: width = {widths[j]:.4f}")
#     return widths.mean(), widths

def Psi(c):
    phi = lambda u: np.exp(-u*u/2)/np.sqrt(2*np.pi)
    Q   = lambda u: 0.5*erfc(u/np.sqrt(2))

    c = np.asarray(c, float)
    pos = 1 + 2*c*np.sqrt(2/np.pi) + c**2
    u = np.abs(c)
    neg = 2*((1+c**2)*Q(u) - u*phi(u))
    return np.where(c >= 0, pos, neg)

def statdim_bound(d, S, rho, c):
    S = np.asarray(sorted(S)); Sc = np.setdiff1d(np.arange(d), S)
    a = np.empty(d); a[S] = 1/np.sqrt(c[S]); a[Sc] = -rho/np.sqrt(c[Sc])
    F = lambda mu: Psi(mu*a).sum()
    hi = 1.0
    while F(hi) < F(hi/2): hi *= 2          # bracket
    r = minimize_scalar(F, bounds=(0, max(hi,1)*4), method='bounded',
                        options={'xatol':1e-10})
    return r.fun, r.x

def diag_analytic(d, S, rho, c):
    """Analytic upper bound on w(C) = E[sup_{x in C, ||x||_2 <= 1} <g,x>] for
    C = {x : ||x_S||_1 >= rho ||x_S^c||_1} in R^d, using the statistical dimension
    bound from Amelunxen et al. (2014)."""
    statdim, mu = statdim_bound(d, S, rho, c)
    return np.sqrt(statdim), mu


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

def diag_sampling(d, S, rho, c, num_samples=20000, seed=0, batch=5000):
    """Exact per-sample width for DIAGONAL covariance c (length-d vector).
    No sign enumeration, no iterative solver."""
    S = np.asarray(sorted(S)); Sc = np.setdiff1d(np.arange(d), S)
    a = np.empty(d)
    a[S]  =  1.0/np.sqrt(c[S])
    a[Sc] = -rho/np.sqrt(c[Sc])
    rng = np.random.default_rng(seed); out = []
    for st in range(0, num_samples, batch):
        n = min(batch, num_samples-st)
        B = np.abs(rng.standard_normal((d, n)))          # |g|, g ~ N(0,I)
        phi = lambda mu: (a[:,None]*np.maximum(B + mu[None,:]*a[:,None], 0)).sum(0)
        lo = np.zeros(n)
        hi = np.full(n, 1.0)
        while np.any(phi(hi) < 0):                        # expand bracket
            hi = np.where(phi(hi) < 0, hi*2, hi)
        for _ in range(100):                              # bisection to machine eps
            mid = 0.5*(lo+hi); p = phi(mid)
            lo = np.where(p < 0, mid, lo); hi = np.where(p < 0, hi, mid)
        mu = np.where(phi(np.zeros(n)) >= 0, 0.0, 0.5*(lo+hi))
        T = np.maximum(B + mu[None,:]*a[:,None], 0)
        T /= np.maximum(np.linalg.norm(T, axis=0), 1e-300)
        out.append(np.einsum('ij,ij->j', B, T))
    v = np.concatenate(out)
    return v.mean(), v.std(ddof=1)/np.sqrt(len(v)), v

def descent_cone_statdim(d, s):
    """Exact statistical dimension of the l1 descent cone at an s-sparse vector,
    under an isotropic Gaussian:  min_tau s(1+tau^2) + (d-s) E[(|g|-tau)_+^2].

    This is a *reference*, not one of the three requested covariances.  The NSP
    cone {||x_S||_1 >= rho||x_Sc||_1} is the union of the descent cones over all
    2^s sign patterns on S, so its statistical dimension is strictly larger;
    comparing the two shows how much of any gap to the data is that relaxation.
    """
    from scipy.optimize import minimize_scalar
    F = lambda tau: s * (1 + tau ** 2) + (d - s) * Psi(-tau)
    r = minimize_scalar(F, bounds=(0, np.sqrt(2 * np.log(d / s)) * 4),
                        method='bounded', options={'xatol': 1e-10})
    return r.fun


if __name__ == "__main__":
    rng = np.random.default_rng(7); d, s = 300, 12
    print(f"{'covariance':22s} {'w (MC)':>9s} {'sqrt(dMC)':>10s} {'sqrt(bound)':>12s} {'slack':>7s}")
    for tag, c in [('c = 1 (isotropic)', np.ones(d)),
                   ('c geometric 1..100', np.geomspace(1,100,d)),
                   ('c random lognormal', np.exp(rng.standard_normal(d)))]:
        m, se, v = diag_sampling(d, range(s), 1.0, c, num_samples=40000, seed=4)
        dmc = (v**2).mean()
        bnd, mus = statdim_bound(d, range(s), 1.0, c)
        print(f"{tag:22s} {m:9.4f} {np.sqrt(dmc):10.4f} {np.sqrt(bnd):12.4f} "
              f"{100*(np.sqrt(bnd)/np.sqrt(dmc)-1):6.2f}%")
    print("\nisotropic asymptotic check   2s log(d/s) =", round(2*s*np.log(d/s),2))
    print("analytic bound               =", round(statdim_bound(d,range(s),1.0,np.ones(d))[0],2))
