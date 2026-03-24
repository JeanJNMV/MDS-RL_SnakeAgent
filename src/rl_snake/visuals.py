from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

OBSERVATION_COLORS: Sequence[str] = (
    "#faf7ef",  # empty
    "#6d8b74",  # snake body
    "#1f3b2c",  # snake head
    "#d1495b",  # food
    "#5c677d",  # obstacle
)


def render_observation(
    observation: np.ndarray,
    *,
    ax: Axes | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Render a single grid observation with Matplotlib."""
    observation = np.asarray(observation)
    if observation.ndim != 2:
        raise ValueError("Observation must be a 2D array.")

    if not np.issubdtype(observation.dtype, np.integer):
        raise ValueError("Observation must contain integer cell values.")

    min_value = int(observation.min())
    max_value = int(observation.max())
    if min_value < 0 or max_value >= len(OBSERVATION_COLORS):
        raise ValueError(f"Observation values must be between 0 and {len(OBSERVATION_COLORS) - 1}.")

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    cmap = ListedColormap(OBSERVATION_COLORS)
    ax.imshow(observation, cmap=cmap, vmin=0, vmax=len(OBSERVATION_COLORS) - 1)

    height, width = observation.shape
    ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
    ax.grid(which="minor", color="#d6d0c4", linewidth=1)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

    if title is not None:
        ax.set_title(title)

    ax.set_aspect("equal")
    return ax.figure, ax
