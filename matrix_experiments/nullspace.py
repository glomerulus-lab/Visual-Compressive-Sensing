from scipy.optimize import linprog
import numpy as np


def _nsp_signed_lp(N, S, Sc, sigma):
    """For a FIXED sign pattern sigma in {-1,+1}^|S|, solve

        maximize   sigma^T v_S
        s.t.       v = N w             (v ranges over null(A))
                   -t_j <= v_j <= t_j,  j in S^c
                   sum_{j in S^c} t_j <= 1

    This is a proper, bounded LP (unlike maximizing ||v_S||_1 directly,
    which is unbounded when phrased as maximizing an upper bound on it).
    An unbounded result here means there is a v in null(A) with v_S^c = 0
    and sigma^T v_S > 0, i.e. the restricted NSP fails on S.
    """
    k = N.shape[1]
    n_Sc = Sc.size
    n_vars = k + n_Sc
    w_idx, t_idx = slice(0, k), slice(k, n_vars)
    N_S, N_Sc = N[S, :], N[Sc, :]

    c = np.zeros(n_vars)
    c[w_idx] = -(sigma @ N_S)  # minimize -sigma^T v_S == maximize sigma^T v_S

    rows, rhs = [], []
    row = np.zeros((n_Sc, n_vars)); row[:, w_idx] = N_Sc; row[np.arange(n_Sc), k + np.arange(n_Sc)] = -1.0
    rows.append(row); rhs.append(np.zeros(n_Sc))          # (N w)_j - t_j <= 0
    row = np.zeros((n_Sc, n_vars)); row[:, w_idx] = -N_Sc; row[np.arange(n_Sc), k + np.arange(n_Sc)] = -1.0
    rows.append(row); rhs.append(np.zeros(n_Sc))          # -(N w)_j - t_j <= 0
    row = np.zeros((1, n_vars)); row[0, t_idx] = 1.0
    rows.append(row); rhs.append(np.array([1.0]))          # sum(t) <= 1

    A_ub, b_ub = np.vstack(rows), np.concatenate(rhs)
    bounds = [(None, None)] * k + [(0, None)] * n_Sc
    return linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")


def nsp_constant(A, S, num_restarts=5, max_iter=30, rcond=None, rng=None):
    """Restricted nullspace property (NSP) constant of A on support S:

        theta_S = max_{v in null(A)}  ||v_S||_1 / ||v_S^c||_1

    Exact computation of theta_S is a maximization of a convex function
    (||v_S||_1) over a polytope, which is NP-hard in general -- so this
    returns a certified LOWER BOUND via an alternating sign-fixing LP
    heuristic (a standard approach for this exact problem):

      1. Fix a sign pattern sigma in {-1,+1}^|S|.
      2. Solve the LP  max sigma^T v_S  s.t. v in null(A), ||v_S^c||_1 <= 1
         via `_nsp_signed_lp` (bounded for fixed sigma; unbounded is itself
         a certificate that the restricted NSP FAILS on S).
      3. Set sigma = sign(v_S) from the new v and repeat.

    Each step's objective ||v_S||_1 is non-decreasing and there are only
    2^|S| sign patterns, so each restart converges in finitely many steps
    to a local maximum -- a valid but not necessarily globally tight value
    of theta_S. Restarting from several random initial sign patterns and
    keeping the best tightens the bound.

    N.B. cost: each restart is up to `max_iter` LP solves over the
    nullspace basis of A (dimension = d - rank(A)), so for the patch-sized
    matrices in this script (d ~ 1024) a single call can take from seconds
    to a few minutes depending on |S| and how many restarts/iters are used.

    Returns (theta_S_lower_bound, v) for the best restart found, where v is
    a certificate vector (||v_S^c||_1 == 1 at convergence), or (np.inf,
    None) if any restart certifies that the restricted NSP fails on S.
    """
    d = A.shape[1]
    S = np.asarray(S)
    Sc = np.setdiff1d(np.arange(d), S)
    if Sc.size == 0:
        raise ValueError("S^c is empty: ||v_S^c||_1 is always 0")

    # Orthonormal basis for null(A): rows of Vh past the numerical rank of A.
    _, sing, Vh = np.linalg.svd(A, full_matrices=True)
    if rcond is None:
        rcond = max(A.shape) * np.finfo(float).eps
    rank = int(np.sum(sing > sing[0] * rcond)) if sing.size else 0
    N = Vh[rank:].T  # (d, nullity), orthonormal columns spanning null(A)
    k = N.shape[1]
    if k == 0:
        return 0.0, np.zeros(d)  # null(A) = {0}

    n_S = S.size
    rng = rng if rng is not None else np.random.default_rng()
    best_theta, best_v = 0.0, np.zeros(d)

    for _ in range(num_restarts):
        sigma = rng.choice([-1.0, 1.0], size=n_S)
        v = np.zeros(d)
        for _ in range(max_iter):
            res = _nsp_signed_lp(N, S, Sc, sigma)
            if res.status == 3:  # unbounded: NSP fails on S
                return np.inf, None
            if not res.success:
                raise RuntimeError(f"NSP LP failed to solve: {res.message}")
            v = N @ res.x[:k]
            new_sigma = np.sign(v[S])
            new_sigma[new_sigma == 0] = sigma[new_sigma == 0]  # keep sign at exact zeros
            if np.array_equal(new_sigma, sigma):
                break
            sigma = new_sigma
        theta = np.linalg.norm(v[S], 1)
        if theta > best_theta:
            best_theta, best_v = theta, v

    return best_theta, best_v
