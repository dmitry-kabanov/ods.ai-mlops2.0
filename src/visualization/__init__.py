import os

import matplotlib.pyplot as plt

from .visualize import (
    draw_bounding_box_on_image,
    draw_bounding_boxes_on_image,
    draw_bounding_boxes_on_image_array,
    display_digits_with_boxes,
    plot_metrics,
)

__all__ = [
    "draw_bounding_box_on_image",
    "draw_bounding_boxes_on_image",
    "draw_bounding_boxes_on_image_array",
    "display_digits_with_boxes",
    "plot_metrics",
]


# Matplotlib config
plt.rc("image", cmap="gray")
plt.rc("grid", linewidth=0)
plt.rc("xtick", top=False, bottom=False, labelsize="large")
plt.rc("ytick", left=False, right=False, labelsize="large")
plt.rc("axes", facecolor="F8F8F8", titlesize="large", edgecolor="white")
plt.rc("text", color="a8151a")
plt.rc("figure", facecolor="F0F0F0")  # Matplotlib fonts
MATPLOTLIB_FONT_DIR = os.path.join(os.path.dirname(plt.__file__), "mpl-data/fonts/ttf")
