"""Plot the restricted nullspace property (RNP) "bad set"

    {x in R^3 : ||x_S||_1 >= rho * ||x_S^c||_1}

for S = {1}, S^c = {2, 3} (1-indexed), i.e. {x : |x1| >= rho * (|x2| + |x3|)}.

Unlike plot_cone_3d.py, this set is NOT convex: it is symmetric about the
origin (x in the set iff -x is), so it splits into two convex "nappes" --
one where x1 >= 0 and one where x1 <= 0 -- joined only at the origin, similar
to a double-napped cone. Each nappe is itself a polyhedral cone (cross
section at fixed |x1| is an L1 ball of radius |x1| / rho), so we build and
plot each nappe separately, each capped to within the L-infinity ball of
radius 1 (i.e. the cube [-1, 1]^3).
"""

from itertools import product

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pycvxset import Polytope

rho = 0.7
radius = 2.0  # L-infinity cap: max_i |x_i| <= radius


def nappe_halfspaces(sign, rho):
    """H-rep (A, b) for the convex nappe {x : sign*x1 >= rho*(|x2| + |x3|)}.

    Equivalent to, for all (s2, s3) in {-1,1}^2:  -sign*x1 + rho*s2*x2 + rho*s3*x3 <= 0.
    """
    rows = [[-sign, rho * s2, rho * s3] for s2, s3 in product([-1, 1], repeat=2)]
    return np.array(rows), np.zeros(4)


def linf_ball(dim, radius):
    """L-infinity ball {x : max_i |x_i| <= radius} as a Polytope (a cube)."""
    A = np.vstack([np.eye(dim), -np.eye(dim)])
    b = radius * np.ones(2 * dim)
    return Polytope(A=A, b=b)


pos_A, pos_b = nappe_halfspaces(1, rho)
neg_A, neg_b = nappe_halfspaces(-1, rho)

nappe_pos = Polytope(A=pos_A, b=pos_b)
nappe_neg = Polytope(A=neg_A, b=neg_b)

# Transformation

# theta_z = np.deg2rad(0)
# theta_y = np.deg2rad(0)
theta_z = np.deg2rad(10)
theta_y = np.deg2rad(20)

rotation_z = np.array(
    [[np.cos(theta_z), -np.sin(theta_z), 0], [np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]]
)
rotation_y = np.array(
    [[np.cos(theta_y), 0, np.sin(theta_y)], [0, 1, 0], [-np.sin(theta_y), 0, np.cos(theta_y)]]
)
rotation = rotation_y @ rotation_z
scaling = np.diag([8.0, 1.0, 1.0])

# Transform the shapes
# nappe_pos = nappe_pos.affine_map(scaling @ rotation)
# nappe_neg = nappe_neg.affine_map(scaling @ rotation)


# Both nappes are unbounded along x1; cap each to within the L-infinity ball.
cap = linf_ball(3, radius * 10)
nappe_pos_capped = nappe_pos.intersection(cap)
nappe_neg_capped = nappe_neg.intersection(cap)

nappe_pos_capped = nappe_pos_capped.affine_map(rotation.T @ scaling @ rotation)
nappe_neg_capped = nappe_neg_capped.affine_map(rotation.T @ scaling @ rotation)

cap = linf_ball(3, radius)
nappe_pos_capped = nappe_pos_capped.intersection(cap)
nappe_neg_capped = nappe_neg_capped.intersection(cap)

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(projection="3d")

nappe_pos_capped.plot3d(ax=ax, patch_args={"facecolor": "lightcoral", "edgecolor": "red", "alpha": 0.5})
nappe_neg_capped.plot3d(ax=ax, patch_args={"facecolor": "lightcoral", "edgecolor": "red", "alpha": 0.5})

max_extent = max(np.abs(nappe_pos_capped.V).max(), np.abs(nappe_neg_capped.V).max())
ax.set_xlim(-max_extent, max_extent)
ax.set_ylim(-max_extent, max_extent)
ax.set_zlim(-max_extent, max_extent)
ax.set_box_aspect((1, 1, 1))
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_zlabel("$x_3$")
legend_handles = [
    Line2D([0], [0], color="red", label=rf"$\{{x : |x_1| \geq \rho(|x_2|+|x_3|)\}}$"),
]
ax.legend(handles=legend_handles)
ax.set_title(rf"RNP bad set: $|x_1| \geq \rho(|x_2| + |x_3|)$, $\rho$ = {rho}")

output_path = "RNP_plot_3d.svg"
fig.savefig(output_path)
print(f"Saved plot to {output_path}")
