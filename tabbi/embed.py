import umap
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tabbi.utils


def umap_embed(df, markers=None, transform_func=np.log1p):
    reducer = umap.UMAP()
    if markers is None:
        markers = list(df.columns)
    data = df[markers].transform(transform_func).values
    reducer.fit(data)
    embedding = reducer.transform(data)
    return df.assign(
        **{'umap-2d-one': embedding[:,0], 'umap-2d-two': embedding[:,1]}
    )


def plot_embedding(
    df, markers, 
    x_col_name="umap-2d-one", y_col_name="umap-2d-two", 
    fig_title=None,
    norm_percentiles=(0, 100),
    subplot_grid_shape=None
):
    fig, axes = tabbi.utils.make_figure(
        subplot_grid_shape, len(markers)
    )
    p1, p2 = norm_percentiles
    if fig_title is not None:
        fig.suptitle(fig_title)
    for m, ax in zip(markers, axes.ravel()):
        plt.sca(ax)           
        sns.scatterplot(
            x=x_col_name, y=y_col_name,
            hue=m,
            hue_norm=(np.percentile(df[m], p1), np.percentile(df[m], p2)),
            data=df,
            palette='magma',
            linewidth=0,
            alpha=0.7,
            marker='.',
            s=1,
            legend=None
        )
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.title.set_text(m)
    return fig