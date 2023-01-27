import numpy as np
import rasterio as rio
import rioxarray as rxr
import tempfile
import raster_utils as ru
from shapely.geometry import shape
import pandas as pd
import geopandas as gpd
import plotly.express as px
import geemap.plotlymap as geemap
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from pathlib import Path
import plots_utils as putils


def download_s3(bbox, ds,
                data_path=None, resolution=1000,
                s3_path='GHSL/',
                bucket='tec-expansion-urbana-p'):
    """Downloads a GHSL windowed rasters for each available year.

    Takes a bounding box (bbox) and downloads the corresponding rasters from a
    the global COG stored on Amazon S3. Returns a single multiband raster,
    a band per year.

    Parameters
    ----------
    bbox : Polygon
        Shapely Polygon defining the bounding box.
    ds : str
        Data set to download, can be one of SMOD, BUILT_S, POP, or LAND.
    resolution : int
        Resolution of dataset to download, either 100 or 1000.
    data_path : Path
        Path to directory to store rasters.
        If none, don't write to disk.
    s3_dir : str
        Relative path to COGs on S3.
    bucket : str

    Returns
    -------
    raster : rioxarray.DataArray
        In memory raster.

    """

    assert ds in ['SMOD', 'BUILT_S', 'POP', 'LAND'], 'Data set not available.'

    s3_path = f'{s3_path}/GHS_{ds}/'
    fname = f'GHS_{ds}_E{{}}_GLOBE_R2022A_54009_{resolution}_V1_0.tif'
    year_list = list(range(1975, 2021, 5))

    array_list = []
    for year in year_list:
        subset, profile = ru.np_from_bbox_s3(
            s3_path + fname.format(year),
            bbox, bucket, nodata_to_zero=True)
        array_list.append(subset)
    ghs_full = np.concatenate(array_list)

    # Create rioxarray
    profile['count'] = ghs_full.shape[0]
    with tempfile.NamedTemporaryFile() as tmpfile:
        with rio.open(tmpfile.name, 'w', **profile) as dst:
            dst.write(ghs_full)
        raster = rxr.open_rasterio(tmpfile.name)

    # Rename band dimension to reflect years
    raster.coords['band'] = year_list

    if data_path is not None:
        raster.rio.to_raster(data_path / f'GHS_{ds}_{resolution}.tif')

    return raster


def load_or_download(bbox, ds,
                     data_path=None, resolution=1000,
                     s3_path='GHSL/',
                     bucket='tec-expansion-urbana-p'):
    """Searches for a GHS dataset to load, if not available,
    downloads it from S3 and loads it.

    Parameters
    ----------
    bbox : Polygon
        Shapely Polygon defining the bounding box.
    ds : str
        Data set to download, can be one of SMOD, BUILT_S, POP, or LAND.
    resolution : int
        Resolution of dataset to download, either 100 or 1000.
    data_path : Path
        Path to directory to store rasters.
        If none, don't write to disk.
    s3_dir : str
        Relative path to COGs on S3.
    bucket : str

    Returns
    -------
    raster : rioxarray.DataArray
        In memory raster.

    """
    fpath = data_path / f'GHS_{ds}_{resolution}.tif'
    if fpath.exists():
        raster = rxr.open_rasterio(fpath)
        raster.coords['band'] = list(range(1975, 2021, 5))
    else:
        raster = download_s3(bbox, ds, data_path, resolution, s3_path, bucket)

    return raster


def smod_polygons(smod, centroid):
    """Find SMOD polygons for urban centers and urban clusters.

    Parameters
    ----------
    smod : xarray.DataArray
        DataArray with SMOD raster data.
    centroid : shapely.Point
        Polygons containing centroid will be identified as
        the principle urban center and cluster.
        Must be in Mollweide proyection.

    Returns
    -------
    smod_polygons : GeoDataFrame
        GeoDataFrame with polygons for urban clusters and centers.

"""

    # Get DoU lvl 1 representation (1: rural, 2: cluster, 3: center)
    smod_lvl_1 = (smod // 10)

    smod_centers = (smod_lvl_1 == 3).astype(smod.dtype)
    smod_clusters = (smod_lvl_1 > 1).astype(smod.dtype)

    transform = smod.rio.transform()

    dict_list = []
    for year in range(1975, 2021, 5):
        centers = rio.features.shapes(
            smod_centers.sel(band=year).values,
            connectivity=8,
            transform=transform)
        clusters = rio.features.shapes(
            smod_clusters.sel(band=year).values,
            connectivity=8,
            transform=transform)

        center_list = [shape(f[0]) for f in centers if f[1] > 0]
        cluster_list = [shape(f[0]) for f in clusters if f[1] > 0]

        center_dicts = [
            {
                'class': 3,
                'year': year,
                'is_main': centroid.within(center),
                'geometry': center
            } for center in center_list
        ]
        cluster_dicts = [
            {
                'class': 2,
                'year': year,
                'is_main': centroid.within(cluster),
                'geometry': cluster
            } for cluster in cluster_list
        ]
        dict_list += center_dicts
        dict_list += cluster_dicts

    smod_polygons = gpd.GeoDataFrame(dict_list, crs=smod.rio.crs)

    return smod_polygons


def built_s_polygons(built):
    """ Returns a polygon per pixel for GHS BUILT rasters. """

    resolution = built.rio.resolution()
    pixel_area = abs(np.prod(resolution))

    built_df = built.to_dataframe(name='b_area').reset_index()
    built_df = built_df.rename(columns={'band': 'year'})
    built_df = built_df.drop(columns='spatial_ref')

    built_df = built_df[built_df.b_area > 0].reset_index(drop=True)

    built_df['fraction'] = built_df.b_area / pixel_area
    built_df['geometry'] = built_df.apply(
        ru.row2cell, res_xy=resolution, axis=1)

    built_gdf = gpd.GeoDataFrame(
        built_df, crs=built.rio.crs).drop(columns=['x', 'y'])

    return built_gdf


def plot_built_poly(built_gdf, bbox_latlon, year=2020):
    """ Plots a map with built information for year with polygons.
    May be slow and memory heavy. """

    west, south, east, north = bbox_latlon.bounds

    Map = geemap.Map()

    gdf = built_gdf[built_gdf.year == year].to_crs(4326).reset_index(drop=True)
    gdf['id'] = list(gdf.index)
    fig = px.choropleth_mapbox(
        gdf,
        geojson=gdf.geometry,
        color='fraction',
        locations='id',
        color_continuous_scale='viridis',
        hover_data={'fraction': True, 'id': False},
        opacity=0.5)
    fig.update_traces(marker_line_width=0)
    Map.add_traces(fig.data)

    Map.update_layout(
        mapbox_bounds={"west": west,
                       "east": east,
                       "south": south,
                       "north": north},
        height=600, width=600)

    return Map


def plot_built_agg_img(thresh=0.2):
    """ Plots historic built using an image overlay. """

    # Obtain bboxes and rasters
    bbox_mollweide, uc_mollweide, fua_mollweide = ru.get_bbox(
        'Monterrey', 'Mexico', Path('../../data/output/cities/'),
        proj='ESRI:54009')
    bbox_latlon, uc_latlon, fua_latlon = ru.get_bbox(
        'Monterrey', 'Mexico', Path('../../data/output/cities/'),
        proj='EPSG:4326')

    smod = load_or_download(bbox_mollweide, 'SMOD',
                            data_path=Path('.'), resolution=1000)
    built = load_or_download(bbox_mollweide, 'BUILT_S',
                             data_path=Path('.'), resolution=100)

    years = ['1975', '1980', '1985', '1990', '1995',
             '2000', '2005', '2010', '2015', '2020']
    years_uint8 = np.array(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        dtype='uint8')

    resolution = built.rio.resolution()
    pixel_area = abs(np.prod(resolution))

    # Create a yearly coded binary built array
    built_bin = (built / pixel_area > thresh).astype('uint8')
    built_bin *= years_uint8[:, None, None]
    built_bin.values[built_bin.values == 0] = 200

    # Aggregate yearly binary built data
    # Keep earliest year of observed urbanization
    built_bin_agg = np.min(built_bin, axis=0)
    built_bin_agg.values[built_bin_agg == 200] = 0
    built_bin_agg.rio.set_nodata(0)

    # Create high resolution raster in lat-lon
    built_bin_agg_latlon = built_bin_agg.rio.reproject(
        'EPSG:4623', resolution=0.0001217742672088975)

    # Create array to hold colorized image
    built_img = np.zeros((*built_bin_agg_latlon.shape, 4), dtype='uint8')

    # Set colormap
    colors_rgba = [plt.cm.get_cmap('cividis', 10)(i) for i in range(10)]
    colors = (np.array(colors_rgba)*255).astype('uint8')
    cmap = {y: c for y, c in zip(years_uint8, colors)}
    cmap_cat = {y: mpl.colors.rgb2hex(c)
                for y, c in zip(years, colors_rgba)}

    # Set colors manually on image array
    for year, color in cmap.items():
        mask = built_bin_agg_latlon == year
        built_img[mask] = color

    # Create image bounding box
    lonmin, latmin, lonmax, latmax = built_bin_agg_latlon.rio.bounds()
    coordinates = [[lonmin, latmin],
                   [lonmax, latmin],
                   [lonmax, latmax],
                   [lonmin, latmax]]

    # Create Image object (memory haevy)
    img = ImageOps.flip(Image.fromarray(built_img))

    # Create figure
    west, south, east, north = bbox_latlon.bounds

    dummy_df = pd.DataFrame({'lat': [0]*10, 'lon': [0]*10, 'Año': years})
    fig = px.scatter_mapbox(dummy_df,
                            lat='lat', lon='lon',
                            color='Año', color_discrete_map=cmap_cat,
                            mapbox_style='carto-positron')
    fig.update_layout(
        mapbox_bounds={"west": west,
                       "east": east,
                       "south": south,
                       "north": north},
        height=1000, width=1000)

    # Create polygons of urban clusters and centers
    smod_p = smod_polygons(smod, uc_mollweide.iloc[0].geometry.centroid)
    main_p = smod_p[(smod_p.year == 2020) & (smod_p.is_main)]

    traces = putils.get_line_traces(
        main_p,
        'class',
        {2: 'Cluster urbano', 3: 'Centro urbano'},
        {'Cluster urbano': 'orange', 'Centro urbano': 'maroon'}
    )

    fig.add_traces(traces)

    fig.update_layout(mapbox_layers=[
                          {
                              "sourcetype": "image",
                              "source": img,
                              "coordinates": coordinates,
                              "opacity": 0.7,
                              "below": 'traces',
                          }]
                      )

    fig.write_html('fig_image.html')

    return fig
