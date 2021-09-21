import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec
from . import utils
from . import move_sns_figure
SeabornFig2Grid = move_sns_figure.SeabornFig2Grid


def plot_pair(
    df, marker_1, marker_2, gates=None,
    kde_class=3,
    transform_func=np.log1p
):
    
    expression_jg = plot_pair_expression(
        df, marker_1, marker_2, gates=gates, transform_func=transform_func
    )

    fig = plt.figure(figsize=(10, 5))
    gs = matplotlib.gridspec.GridSpec(1, 2)
    # this needs to be kept for caller resizing
    move_grid = SeabornFig2Grid(expression_jg, fig, gs[0])
    
    ax = fig.add_subplot(gs[0, 1])
    plot_pair_spatial(
        df, marker_1, marker_2, gates=gates,
        kde_class=kde_class,
        transform_func=transform_func,
        ax=ax
    )
    gs.tight_layout(fig)
    return fig, move_grid

import skimage.filters
def get_gates_and_label(df, marker_1, marker_2, gates):
    d1, d2 = df[marker_1], df[marker_2]
    if gates is None:
        gates = (
            skimage.filters.threshold_triangle(d1),
            skimage.filters.threshold_triangle(d2)
        )
    gated_label = (d1 > gates[0]) + 2 * (d2 > gates[1])
    return gates, gated_label


def plot_pair_expression(
    df, marker_1, marker_2, gates=None, transform_func=np.log1p
):
    gates, gated_label = get_gates_and_label(
        df, marker_1, marker_2, gates
    )
    
    jg = sns.JointGrid(
        x=marker_1, y=marker_2, hue='gated_label',
        data=(df
            .transform(transform_func)
            .assign(**dict(gated_label=gated_label))
            .sample(np.min([10**5, df.shape[0]]))
        )
    )
    jg.plot_joint(
        sns.scatterplot, palette='Dark2', 
        s=5, linewidth=0, alpha=0.1
    )
    # remove hue for the margin plots
    jg.hue = None
    jg.plot_marginals(
        sns.histplot, 
        bins=150, fill=False, element='step', color='#777777'
    )
    jg.ax_marg_x.axvline(transform_func(gates[0]), c='salmon')
    jg.ax_marg_y.axhline(transform_func(gates[1]), c='salmon')

    jg.ax_joint.set_xlim(np.percentile(jg.x, 0.1), np.percentile(jg.x, 99.9))
    jg.ax_joint.set_ylim(np.percentile(jg.y, 0.1), np.percentile(jg.y, 99.9))

    percentages = [
        100 * np.sum(gated_label == i) / gated_label.shape[0]
        for i in np.unique(gated_label)
    ]

    jg.ax_joint.legend([f'{marker_1} / {marker_2}'] + [
        f'{type} ({p:.2f}%)'
        for type, p in
        zip(
            ['- / -', '+ / -', '- / +', '+ / +'],
            percentages
        )
    ])
    return jg


import corner.core as cc
def plot_pair_spatial(
    df, marker_1, marker_2,
    gates=None,
    kde_class=3,
    transform_func=np.log1p,
    ax=None
):
    gates, gated_label = get_gates_and_label(
        df, marker_1, marker_2, gates
    )

    sub_df = (
        df[['X_centroid', 'Y_centroid']]
            .assign(gated_label=gated_label)
            .sample(np.min([10**6, df.shape[0]]))
    )

    ax = plot_tissue_scatter(
        sub_df,
        ax=ax, kwargs=dict(c='gated_label', cmap='Dark2', vmin=0, vmax=7)
    )

    cc.hist2d(
        sub_df.query('gated_label == @kde_class').X_centroid.values,
        sub_df.query('gated_label == @kde_class').Y_centroid.values,
        bins=200, smooth=1, ax=ax,
        plot_datapoints=False, plot_density=False,
        no_fill_contours=True,
        contour_kwargs=dict(cmap='Greys_r', colors=None)
    )
    utils.as_img()
    return ax


def plot_tissue_scatter(df, ax=None, kwargs=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot()
    scatter_kwargs = dict(s=2, linewidths=0)
    if kwargs is not None:
        scatter_kwargs.update(kwargs)
    ax.scatter(
        'X_centroid', 'Y_centroid',
        data=df,
        **scatter_kwargs
    )
    utils.as_img()
    return ax