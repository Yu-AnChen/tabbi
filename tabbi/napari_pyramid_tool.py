import napari
import dask.array as da
import zarr
import tifffile


def lazy_view(
    tiff_path, channel_names=None, colormaps=None, viewer=None,
    channel_from_to=(None, None)
):
    target_filepath = tiff_path
    # workaround for Faas pyramid
    tiff = tifffile.TiffFile(target_filepath, is_ome=False)
    n_levels = len(tiff.series[0].levels)
    base_shape = tiff.series[0].shape
    if len(base_shape) == 2:
        n_channels = 1
        channel_axis = None
    elif len(base_shape) == 3:
        n_channels = tiff.series[0].shape[0]
        channel_axis = 0
    else:
        raise NotImplementedError('Only 2D/3D images are currently supported')
    
    tiff.close()

    channel_from, channel_to = channel_from_to
    n_channels = len(range(n_channels)[channel_from:channel_to])

    if channel_names is not None:
        assert n_channels == len(channel_names), (
            f'number of channel names ({len(channel_names)}) must '
            f'match number of channels ({n_channels})'
        )

    if colormaps is not None:
        assert n_channels == len(colormaps), (
            f'number of colormaps ({len(colormaps)}) must '
            f'match number of channels ({n_channels})'
        )

    z = zarr.open(tiff.aszarr(), mode='r')
    # FIXME would this still be the case for single level pyramid?
    assert type(z) == zarr.hierarchy.Group
    pyramid = [
        da.from_zarr(z[i])[channel_from:channel_to] 
        for i in range(n_levels)
    ]

    viewer = viewer if viewer is not None else napari.Viewer()
    viewer.add_image(
        pyramid, multiscale=True, channel_axis=channel_axis,
        visible=False, name=channel_names, colormap=colormaps,
        blending='additive'
    )
    return viewer


def pan_viewer(viewer, center):
    current_zoom = viewer.camera.zoom
    viewer.camera.center = center
    viewer.camera.zoom *= 1.001
    viewer.camera.zoom = current_zoom