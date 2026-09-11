"""Plot the cone {u : u >= 0, u1 >= rho * (u2 + u3)} in 3D for a parameter 0 < rho <= 1."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pycvxset import Ellipsoid, Polytope
from pycvxset.common.polytope_approximations import polytopic_outer_approximation

rho = 0.7

CAP_TYPE = "planar"  # "planar" (u1 + u2 + u3 <= radius) or "spherical" (||u|| <= radius)
radius = 1.0
n_sphere_halfspaces = 206  # only used when CAP_TYPE == "spherical" (must be 2*dim + 8*k)


def capped(poly, cap_type, radius, n_halfspaces):
    """Intersect an unbounded polytope with a bounded cap for plotting.

    "planar" cuts with the single halfspace u1 + u2 + u3 <= radius. "spherical"
    intersects with a polytopic outer-approximation of the ball ||u|| <= radius
    (tangent halfspaces at directions spread over the sphere, via
    pycvxset.common.polytope_approximations.polytopic_outer_approximation),
    which renders as a rounded cap.
    """
    if cap_type == "planar":
        return poly.intersection_with_halfspaces(np.array([[1.0, 1.0, 1.0]]), np.array([radius]))
    elif cap_type == "spherical":
        ball = Ellipsoid(c=np.zeros(3), r=radius)
        ball_outer = polytopic_outer_approximation(ball, n_halfspaces=n_halfspaces)
        return poly.intersection(ball_outer)
    else:
        raise ValueError(f"unknown CAP_TYPE: {cap_type!r}")


# Nonnegative orthant: u >= 0, i.e. -u <= 0.
orthant_A = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
orthant_b = np.zeros(3)

# Cone: nonnegative orthant plus u1 >= rho * (u2 + u3), i.e. -u1 + rho*u2 + rho*u3 <= 0.
cone_A = np.vstack([orthant_A, [-1, rho, rho]])
cone_b = np.zeros(4)

orthant = Polytope(A=orthant_A, b=orthant_b)
cone = Polytope(A=cone_A, b=cone_b)

# Both are unbounded; cap them to get a bounded slice to plot.
orthant_capped = capped(orthant, CAP_TYPE, radius, n_sphere_halfspaces)
cone_capped = capped(cone, CAP_TYPE, radius, n_sphere_halfspaces)


# Transformation

theta_z = np.deg2rad(0)
theta_y = np.deg2rad(0)
#theta_z = np.deg2rad(30)
#theta_y = np.deg2rad(20)

rotation_z = np.array(
    [[np.cos(theta_z), -np.sin(theta_z), 0], [np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]]
)
rotation_y = np.array(
    [[np.cos(theta_y), 0, np.sin(theta_y)], [0, 1, 0], [-np.sin(theta_y), 0, np.cos(theta_y)]]
)
rotation = rotation_y @ rotation_z
scaling = np.diag([2.0, 1.0, 1.0])

# Transform the shapes
orthant_capped = orthant_capped.affine_map(scaling @ rotation)
cone_capped = cone_capped.affine_map(scaling @ rotation)


fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(projection="3d")

orthant_capped.plot3d(ax=ax, patch_args={"facecolor": "none", "edgecolor": "gray", "linestyle": "--"})
cone_capped.plot3d(ax=ax, patch_args={"facecolor": "lightblue", "edgecolor": "blue", "alpha": 0.5})

max_extent = np.abs(orthant_capped.V).max()
ax.set_xlim(0, max_extent)
ax.set_ylim(0, max_extent)
ax.set_zlim(0, max_extent)
ax.set_box_aspect((1, 1, 1))
ax.set_xlabel("$u_1$")
ax.set_ylabel("$u_2$")
ax.set_zlabel("$u_3$")
legend_handles = [
    Line2D([0], [0], color="gray", linestyle="--", label="nonnegative orthant (rho = 0)"),
    Line2D([0], [0], color="blue", label=rf"cone ($\rho$ = {rho})"),
]
ax.legend(handles=legend_handles)
ax.set_title(rf"Cone $\{{u \geq 0 : u_1 \geq \rho (u_2 + u_3)\}}$, $\rho$ = {rho}")

output_path = "cone_plot_3d.svg"
fig.savefig(output_path)
print(f"Saved plot to {output_path}")
