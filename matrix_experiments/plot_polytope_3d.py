"""Plot a 3D l1 ball rotated 30/20 degrees and stretched by (0.5, 1, 2) along the coordinate axes."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pycvxset import Polytope

theta_z = np.deg2rad(30)
theta_y = np.deg2rad(20)

rotation_z = np.array(
    [[np.cos(theta_z), -np.sin(theta_z), 0], [np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]]
)
rotation_y = np.array(
    [[np.cos(theta_y), 0, np.sin(theta_y)], [0, 1, 0], [-np.sin(theta_y), 0, np.cos(theta_y)]]
)
rotation = rotation_y @ rotation_z
scaling = np.diag([0.5, 1.0, 1.5])

l1_ball = Polytope(V=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]]))
transformed = l1_ball.affine_map(scaling @ rotation)

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(projection="3d")

l1_ball.plot3d(ax=ax, patch_args={"facecolor": "none", "edgecolor": "gray", "linestyle": "--"})
transformed.plot3d(ax=ax, patch_args={"facecolor": "lightblue", "edgecolor": "blue", "alpha": 0.5})

max_extent = np.abs(transformed.V).max()
ax.set_xlim(-max_extent, max_extent)
ax.set_ylim(-max_extent, max_extent)
ax.set_zlim(-max_extent, max_extent)
ax.set_box_aspect((1, 1, 1))
legend_handles = [
    Line2D([0], [0], color="gray", linestyle="--", label="original l1 ball"),
    Line2D([0], [0], color="blue", label="rotated + stretched"),
]
ax.legend(handles=legend_handles)
ax.set_title(r"$\ell_1$ ball rotated 30°/20° and stretched by (0.5, 1, 1.5)")

output_path = "polytope_plot_3d.svg"
fig.savefig(output_path)
print(f"Saved plot to {output_path}")
