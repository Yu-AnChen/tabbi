import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec
import matplotlib.patches
from . import utils
from . import move_sns_figure
SeabornFig2Grid = move_sns_figure.SeabornFig2Grid


def plot_pair(
    df, marker_1, marker_2,
    gates=None,
    transform_func=np.log1p,
    expression_plot_kwargs=None,
    spatial_plot_kwargs=None,
):
    
    kwarg_base = {
        'gates': gates,
        'transform_func': transform_func
    }

    if expression_plot_kwargs is None: expression_plot_kwargs = {}
    if spatial_plot_kwargs is None: spatial_plot_kwargs = {}

    expression_jg = plot_pair_expression(
        df, marker_1, marker_2,
        **dict(kwarg_base, **expression_plot_kwargs)
    )


    fig = plt.figure(figsize=(10, 5))
    gs = matplotlib.gridspec.GridSpec(1, 2)
    # this needs to be kept for caller resizing
    move_grid = SeabornFig2Grid(expression_jg, fig, gs[0])
    
    ax = fig.add_subplot(gs[0, 1])
    plot_pair_spatial(
        df, marker_1, marker_2,
        ax=ax,
        **dict(kwarg_base, **spatial_plot_kwargs)
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


import datashader as ds
import datashader.mpl_ext as dsmpl
import matplotlib.colors
import matplotlib.cm
def plot_pair_expression(
    df, marker_1, marker_2,gates=None, transform_func=np.log1p,
    ds_kwargs=None
):
    gates, gated_label = get_gates_and_label(
        df, marker_1, marker_2, gates
    )
    
    df_plot = (df[[marker_1, marker_2]]
        .transform(transform_func)
        .assign(gated_label=gated_label.astype('category'))
    )
    jg = sns.JointGrid(
        x=marker_1, y=marker_2,
        data=df_plot
    )
    
    kwargs = dict(
        aggregator=ds.count_cat('gated_label'),
        aspect='auto',
        color_key=[matplotlib.colors.to_hex(c) for c in matplotlib.cm.Dark2(np.arange(8))]
    )
    if ds_kwargs is None: ds_kwargs = {}
    kwargs.update(ds_kwargs)
    
    artist = dsmpl.dsshow(
        df_plot,
        ds.Point(marker_1, marker_2),
        ax=jg.fig.axes[0],
        **kwargs
    )
    
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

    # legend text, in the form of
    # marker1 / marker2
    # - / - (XX.XX%)
    # + / - (XX.XX%)
    # - / + (XX.XX%)
    # + / + (XX.XX%)
    legend_labels = (
        [f'{marker_1} / {marker_2}'] +
        [
            f'{type} ({p:.2f}%)'
            for type, p in
            zip(
                ['- / -', '+ / -', '- / +', '+ / +'],
                percentages
            )
        ]
    )
    legend_handles = (
        [matplotlib.patches.Patch(alpha=0)] +
        artist.get_legend_elements()
    )
    jg.ax_joint.legend(
        handles=legend_handles,
        labels=legend_labels
    )
    return jg


import corner.core as cc
import functools
import datashader.transfer_functions
def plot_pair_spatial(
    df, marker_1, marker_2,
    gates=None,
    kde_class=3,
    transform_func=np.log1p,
    ax=None,
    plot_contour=False,
    ds_kwargs=None
):
    gates, gated_label = get_gates_and_label(
        df, marker_1, marker_2, gates
    )

    df_plot = (
        df[['X_centroid', 'Y_centroid']]
            .assign(gated_label=gated_label.astype('category'))
    )
    
    kwargs = dict(
        aggregator=ds.count_cat('gated_label'),
        color_key=[matplotlib.colors.to_hex(c) for c in matplotlib.cm.Dark2(np.arange(8))],
        alpha_range=(100, 255),
        shade_hook=functools.partial(
            datashader.transfer_functions.dynspread,
            threshold=0.99, how='over'
        )
    )
    if ds_kwargs is None: ds_kwargs = {}
    kwargs.update(ds_kwargs)
    
    ax = datashader_tissue_scatter(
        df_plot,
        ax=ax,
        kwargs=kwargs
    )
    
    if plot_contour:
        cc.hist2d(
            df_plot.query('gated_label == @kde_class').X_centroid.values,
            df_plot.query('gated_label == @kde_class').Y_centroid.values,
            bins=200, smooth=1, ax=ax,
            plot_datapoints=False, plot_density=False,
            no_fill_contours=True,
            contour_kwargs=dict(cmap='Greys_r', colors=None)
        )
    # utils.as_img()
    return ax


def datashader_tissue_scatter(df, ax=None, kwargs=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot()
    if kwargs is None:
        kwargs = {}
    kwargs_final = dict(
        shade_hook=functools.partial(
            datashader.transfer_functions.dynspread,
            threshold=0.99, how='over'
        )
    )
    kwargs_final.update(kwargs)
    dsmpl.dsshow(
        df,
        ds.Point('X_centroid', 'Y_centroid'),
        ax=ax,
        **kwargs_final
    )
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