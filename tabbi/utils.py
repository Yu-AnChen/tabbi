import matplotlib.pyplot as plt
import numpy as np


def make_figure(subplot_grid_shape, n_plots):
    if subplot_grid_shape is None:
        subplot_grid_shape = (1, n_plots)
    n_rows, n_cols = subplot_grid_shape
    fig, axes = plt.subplots(n_rows, n_cols, sharex=True, sharey=True)
    axes = np.array(axes)
    return fig, axes


def as_img():
    plt.gca().set_aspect('equal')
    plt.gca().invert_yaxis()