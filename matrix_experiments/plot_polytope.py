"""Plot a 2D l1 ball rotated 30 degrees and stretched by (0.5, 2) along the coordinate axes."""

import numpy as np
import matplotlib.pyplot as plt
from pycvxset import Polytope

theta = np.deg2rad(30)
rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
scaling = np.diag([0.5, 2.0])

l1_ball = Polytope(V=np.array([[1, 0], [0, 1], [-1, 0], [0, -1]]))
transformed = l1_ball.affine_map(scaling @ rotation)

fig, ax = plt.subplots(figsize=(6, 6))
l1_ball.plot2d(ax=ax, patch_args={"facecolor": "none", "edgecolor": "gray", "linestyle": "--", "label": "original l1 ball"})
transformed.plot2d(ax=ax, patch_args={"facecolor": "lightblue", "edgecolor": "blue", "alpha": 0.5, "label": "rotated + stretched"})

ax.set_aspect("equal")
ax.legend()
ax.set_title(r"$\ell_1$ ball rotated 30° and stretched by (0.5, 2)")

output_path = "polytope_plot.svg"
fig.savefig(output_path)
print(f"Saved plot to {output_path}")
