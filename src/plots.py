from ast import literal_eval
import base64
import json
import degree_of_urbanization as du
import geemap.plotlymap as geemap
import matplotlib.pyplot as plt
import matplotlib as mpl
import plotly.graph_objects as go
import rpy2.robjects as robjects
import plotly.express as px
import data_sources as dts
from PIL import ImageDraw
from PIL import ImageFont
import contextily as ctx
from pathlib import Path
import geopandas as gpd
import rioxarray as rxr
import rasterio as rio
from PIL import Image, ImageOps
import pandas as pd
import xarray as xr
import stats as sts
import geemap as gp
import numpy as np
import requests
import rasterio
import warnings
import boto3
import sys
import ee
sys.path.append('./sleuth_python')
import spread as spd
import raster_utils as ru
import plots_utils as putils
import ghsl

warnings.filterwarnings('ignore')

with open('./assets/key') as f:
    stringKey = f.read()
dataKey = base64.b64decode(stringKey).decode("utf-8")
python_dict = literal_eval(dataKey)
privateKey = json.dumps(python_dict, sort_keys=True)
service_account = 'pred-exp-deault@predictor-de-expansion.iam.gserviceaccount.com'
credentials = ee.ServiceAccountCredentials(service_account, None, privateKey)
ee.Initialize(credentials)

directory = Path('./data')
cities_fua = gpd.read_file(directory / 'output/cities/cities_fua.gpkg')
cities_uc = gpd.read_file(directory / 'output/cities/cities_uc.gpkg')

s3Client = boto3.client('s3')


def download_data(country, city):
    '''
    This function downloads every data source required to create the dashboard
    -----------------

    Parameters:
        - city (str): city to download the data for
        - country (str): country to download the data for
    -----------------

    Returns:
         None
    '''
    # This variable is used globally for all the other functions
    global datadir

    option = '{}-{}'.format(country, city)
    datadir = directory / f'cache/{option}/'

    bbox, uc, fua = du.get_bbox(city, country, datadir, buff=10)

    dts.download_rasters(country, city, directory, s3Client, buff=10)
    du.download_rasters_s3(bbox, datadir)
    spd.temp_driver(datadir)


def land_graph(country, city):
    '''
    The purpose of this function is to create the stats cards and
    the area chart for the 'cobertura de suelo' in past, as well
    as the horizontal bar chart 'cobertura de suelo' in present
    from the dashboard.

    This works with DynamicWorld data.

    -----------------

    Parameters:
        - city (str): city to create the visuals for
        - country (str): country to create the visuals for
    -----------------

    Returns:
        - area (plotly figure): area chart
        - land_bar (plotly figure): horizontal bar chart
        - card_land_txt (dict): dictionary with the card's text
        - card_land_num (dict): dictionary with the card's stats
    '''
    # Preprocessing of the dataframe
    land_dir = datadir / 'historic_landcover.csv'
    land_data = pd.read_csv(land_dir)

    land_data.drop('Unnamed: 0', inplace=True, axis=1)
    land_data = land_data.rename(
        columns=dts.sources['DynamicWorld']['class_dict'])
    land_data = land_data.rename(columns={'null': 'Desconocido'})
    land_data.set_index('year', inplace=True)
    land_data = land_data*0.0001

    # Area chart of the land use for all the years
    area = px.area(land_data, x=land_data.index, y=land_data.columns,
                   color_discrete_map=dts.sources['DynamicWorld']['colors'],
                   title='Cobertura de suelo por año')

    area.update_traces(mode='markers+lines')

    area.update_layout(xaxis_dtick='M1',
                       yaxis_title='km2',
                       xaxis_title='Año',
                       legend_title='Tipo de suelo:',
                       font_size=17,
                       font_color='rgb(142, 136, 131)',
                       font_family='Franklin Gothic Book',
                       title_font_size=25,
                       title_font_color='rgb(99, 95, 93)',
                       title_font_family='ITC Franklin Gothic Std',
                       title_xanchor='center',
                       title_yanchor='top',
                       title_x=0.45,
                       title_y=.9,
                       paper_bgcolor='white',
                       plot_bgcolor='white',
                       legend=dict(y=.85))

    area.update_xaxes(
        showgrid=True, gridwidth=1, gridcolor='rgb(245, 243, 242)')
    area.update_yaxes(
        showgrid=True, gridwidth=1, gridcolor='rgb(245, 243, 242)')

    area.add_annotation(
        x=1.2,
        y=1.2,
        xref="paper",
        yref="paper",
        text="Doble click en el <br> tipo de suelo <br> para verlo",
        showarrow=False,
        font=dict(family="ITC Franklin Gothic Std",
                  size=17,
                  color="rgb(99, 95, 93)"),
        align="center",
        bordercolor="#c7c7c7",
        borderwidth=2,
        borderpad=4,
    )

    # Statistic of land types
    card_land_txt = {
        'Agua': 'No aplica.',
        'Árboles': 'No aplica.',
        'Césped/Pasto': 'No aplica.',
        'Vegetación inundada': 'No aplica.',
        'Cultivos': 'No aplica.',
        'Arbusto y matorral': 'No aplica.',
        'Urbanización': 'No aplica.',
        'Descubierto': 'No aplica.',
        'Nieve y hielo': 'No aplica.'
    }

    card_land_num = {
        'Agua': 'No aplica.',
        'Árboles': 'No aplica.',
        'Césped/Pasto': 'No aplica.',
        'Vegetación inundada': 'No aplica.',
        'Cultivos': 'No aplica.',
        'Arbusto y matorral': 'No aplica.',
        'Urbanización': 'No aplica.',
        'Descubierto': 'No aplica.',
        'Nieve y hielo': 'No aplica.'
    }

    for (colname, colval) in land_data.iteritems():
        # Calculate change per land type
        amount = colval.values[-1]/colval.values[0]

        if amount < 1:
            word = 'disminución'
        elif amount > 1:
            word = 'incremento'

        if colname in card_land_txt.keys():
            card_land_txt[colname] = f'{word} en su cobertura del suelo.'
            card_land_num[colname] = amount

    # Preprocessing of dataframe for the present (most recent) year
    present_land = land_data.iloc[-1:]
    present_land.reset_index(drop=True)
    df_land = present_land.T
    df_land = df_land.reset_index()
    df_land = df_land.rename(columns={'index': 'suelo', 2021: 'km2'})
    df_land = df_land.sort_values(by='km2', ascending=False)

    # Horizontal bar chart for present data
    land_bar = px.bar(
        df_land, y='suelo', x='km2', title="Cobertura de suelo",
        text_auto='.2s', orientation='h', color='suelo',
        color_discrete_map=dts.sources['DynamicWorld']['colors']
    )

    land_bar.update_traces(
        textfont_size=17,
        textangle=0,
        textposition="outside",
        cliponaxis=False
    )

    land_bar.update_layout(
        yaxis_title=' ',
        xaxis_title='km2',
        font_size=17,
        font_color='rgb(142, 136, 131)',
        font_family='Franklin Gothic Book',
        title_font_size=25,
        title_font_color='rgb(99, 95, 93)',
        title_font_family='ITC Franklin Gothic Std',
        title_xanchor='left',
        title_yanchor='top',
        title_x=0.45,
        title_y=.9,
        legend_title=" ",
        paper_bgcolor='white',
        plot_bgcolor='white',
        legend=dict(
            y=1.1,
            # x=0.5
        )
    )

    land_bar.update_xaxes(
        showgrid=True, gridwidth=1, gridcolor='rgb(245, 243, 242)')
    land_bar.update_yaxes(showgrid=False)

    return area, land_bar, card_land_txt, card_land_num


# POPULATION, URBAN AND DENSITY GRAPHS AND STATS (past)

def urbanization_df(city, country):
    '''
    The function creates (if not previously created)
    two dataframes that are return from the full_run
    function from the degree_of_urbanization.py

    This works with Landscan and GISA 2 data.

    -----------------

    Parameters:
        - city (str): city to create the visuals for
        - country (str): country to create the visuals for
    -----------------

    Returns:
        - dfstats (dataframe):
        - dflargets (dataframe):
    '''
    dfstats_path = datadir / 'dou_stats.csv'
    dflargest_path = datadir / 'dou_largest.csv'

    # Check if the dataframes exists if not, creates, saves and reads them
    if not dfstats_path.exists() or not dflargest_path.exists():
        print('Downloading urbanization csv...')
        bbox, uc, fua = du.get_bbox(city, country, datadir, buff=10)
        du.full_run(bbox, datadir)
        print('Done.')

        dfstats = pd.read_csv(dfstats_path)
        dflargest = pd.read_csv(dflargest_path)

        return dfstats, dflargest

    # If already exists then only reads them
    else:
        dfstats = pd.read_csv(dfstats_path)
        dflargest = pd.read_csv(dflargest_path)

        return dfstats, dflargest


def pop_past_graphs(city, country):
    '''
    The function creates the population visuals
    for the past section. These are three graphs:
    urbanization, population and population
    density.

    This works with Landscan and GISA 2 data.

    -----------------

    Parameters:
        - city (str): city to create the visuals for
        - country (str): country to create the visuals for
    -----------------

    Returns:
        - figures (list): list of the three graphs in json
        format
    '''

    df_stats, df_largest = urbanization_df(city, country)

    spline_df = 4
    r_smooth_spline = robjects.r['smooth.spline']

    # Get the real data poitns por urb, pop and density
    x = df_largest.year.values
    y1 = df_largest.Pob.values
    y2 = df_largest.Area.values
    y3 = df_largest.Pop_density.values

    # Convert the data points in R objects
    r_x = robjects.FloatVector(x)
    r_y1 = robjects.FloatVector(y1)
    r_y2 = robjects.FloatVector(y2)
    r_y3 = robjects.FloatVector(y3)

    x_smooth = np.linspace(2000, 2019)

    # Create the smooth lines for the tendency
    rspline1 = r_smooth_spline(x=r_x, y=r_y1, df=spline_df)
    rspline2 = r_smooth_spline(x=r_x, y=r_y2, df=spline_df)
    rspline3 = r_smooth_spline(x=r_x, y=r_y3, df=spline_df)
    ySpline1 = np.array(robjects.r['predict'](
        rspline1, robjects.FloatVector(x_smooth)).rx2('y'))
    ySpline2 = np.array(robjects.r['predict'](
        rspline2, robjects.FloatVector(x_smooth)).rx2('y'))
    ySpline3 = np.array(robjects.r['predict'](
        rspline3, robjects.FloatVector(x_smooth)).rx2('y'))

    # List of lists with the three different data
    data = [
        [y1, ySpline1,
         'Población',
         'Población urbana'],
        [y2, ySpline2,
         'Área urbana',
         'Superficie construida en km'],
        [y3, ySpline3,
         'Densidad de población',
         'Densidad']
    ]

    # Iterate in the list to create the three graphs and save them in a list
    figures = []
    for d in data:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x, y=d[0],
                mode='markers',
                line_color='rgba(169,169,169, 0.7)',
                marker=dict(size=8),
                name='Población'))
        fig.add_trace(
            go.Scatter(
                x=x_smooth, y=d[1],
                mode='lines',
                line_color='#2F5C97',
                line_width=2,
                name='Tendencia'))

        fig.update_layout(
            title=d[2],
            xaxis_title='Año',
            yaxis_title=d[3],
            font_size=17,
            font_color='rgb(142, 136, 131)',
            font_family='Franklin Gothic Book',
            title_font_size=25,
            title_font_color='rgb(99, 95, 93)',
            title_font_family='ITC Franklin Gothic Std',
            title_xanchor='left',
            title_yanchor='top',
            title_x=0.22,
            title_y=.95,
            paper_bgcolor='white',
            plot_bgcolor='white',
            margin=dict(l=20, r=20),
            showlegend=False)

        fig.update_xaxes(
            showgrid=True, gridwidth=1, gridcolor='rgb(245, 243, 242)')
        fig.update_yaxes(
            showgrid=True, gridwidth=1, gridcolor='rgb(245, 243, 242)',
            range=[0,  (np.max(d[0])*1.05)])

        figures.append(fig.to_json())

    return figures


def pop_past_stats(city, country):
    '''
    The function calculates the change of
    population, urbanization and population
    density.

    This works with Landscan and GISA 2 data.

    -----------------

    Parameters:
        - city (str): city to create the visuals for
        - country (str): country to create the visuals for
    -----------------

    Returns:
        - cards_info (dict): the first three elements are
        the exact change in number and the last three the
        text metioning if they increased of decreased.
    '''

    df_stats, df_largest = urbanization_df(city, country)

    data = ['Area', 'Pob', 'Pop_density']
    info = []

    for dat in data:
        amount = df_largest[dat].iloc[-1]/df_largest[dat].iloc[0]
        info.append(amount)

        if amount < 1:
            word = 'disminuyó'
        elif amount > 1:
            word = 'incrementó'
        else:
            word = 'se mantuvo'

        info.append(word)

    cards_info = {
        'diff_urb': info[0],
        'diff_pop': info[2],
        'diff_den': info[4],
        'urb_txt': f'{info[1]} la superficie construida.',
        'pop_txt': f'{info[3]} la población.',
        'den_txt': f'{info[5]} la densidad poblacional.',
    }

    return cards_info


# DENSITY POPULATION TXT (present)

def density_landscan(country, city):
    '''
    Function to get the most receant data for
    the population density. This is meant to
    be in the present -> population.

    This works with Landscan and GISA 2 data.

    -----------------

    Parameters:
        - city (str): city to create the visuals for
        - country (str): country to create the visuals for
    -----------------

    Returns:
        - density_txt (str): a string with the most
        recent data for population density.
    '''

    df_stats, df_largest = urbanization_df(city, country)
    present_density = round(df_largest['Pop_density'].iloc[-1], 2)

    density_txt = (f'La densidad de población es de {present_density} '
                   f'(habitante por km2) en el 2019.')

    return density_txt


# URBANIZATION TXT (present)

def urbanization_txt(country, city):
    '''
    Function to get the most receant data for
    the urbanization. This is meant to be in
    the present -> urbanization.

    This works with Landscan and GISA 2 data.

    -----------------

    Parameters:
        - city (str): city to create the visuals for
        - country (str): country to create the visuals for
    -----------------

    Returns:
        - urbanization_txt (str): a string with the most
        recent data for urbanization.
    '''

    df_stats, df_largest = urbanization_df(city, country)
    present_urb = round(df_largest['Area'].iloc[-1], 2)

    urbanization_txt = (f'La superficie construida es de {present_urb} '
                        'km2 en el 2019.')

    return urbanization_txt


# PRESENT MAP: POPULATION AND LAND USE

def present_map(src, country, city):
    '''
    The goal of this function is to create a map
    based on the input 'src' with Landscan (population)
    or DynamicWorld (land use). Both through plotly geemap.

    This works with Landscan or DynamicWorld data.

    -----------------

    Parameters:
        - src (str): which data source to create the map for
        - city (str): city to create the visuals for
        - country (str): country to create the visuals for
    -----------------

    Returns:
        - Map (plotly geemap map): a map with a specific
        layer (DynamicWorld or Landscan)
    '''

    # UC and FUA
    # uc = cities_uc.loc[(cities_uc.country == country) &
    #                    (cities_uc.city == city)]
    fua = cities_fua.loc[(cities_fua.country == country) &
                         (cities_fua.city == city)]

    # Map Creation
    Map = geemap.Map()

    # Controls list (menu items to add on the map)
    controls = [
        'drawline',
        'drawopenpath',
        'drawclosedpath',
        'drawcircle',
        'drawrect',
        'eraseshape'
    ]

    # Add the controls list
    Map.add_controls(controls)

    # Center on city
    # poly_uc = uc.geometry.values[0]
    poly_fua = fua.geometry.values[0]
    point = poly_fua.centroid
    Map.set_center(point.y, point.x, zoom=10)

    # Add expanded bounding box
    bbox = dts.get_roi(fua, buff=7, buff_type='km')

    if src == 'Landscan':
        # Get Landscan tif url from the s3
        url = ('http://tec-expansion-urbana-p.s3.amazonaws.com/'
               'landscan_global/landscan-global-2020-colorized.tif')

        # Add layer with opacity
        Map.add_cog_layer(url, opacity=0.7)
        Map.set_center(point.y, point.x, zoom=10)

        return Map

    elif src == 'DynamicWorld':
        dynamic_image = dts.load_dw(
            bbox=bbox.geometry[0], year=2021, clip=False, reproj=False)

        Map.addLayer(
            dynamic_image,
            dts.sources['DynamicWorld']['vis_params'],
            'DynamicWorld' + ' ' + str(2021))

        return Map

        # A function in data_sources.py already creates a map for DynamicWorld
        # TODO: return dts.map_dw(
        #   Map, bbox_ee=bbox.geometry[0], year=2021, clip=False, reproj=False)


# GISA MAP PREDICTION (present)

def gisa_pre_map():
    '''
    It creates a static plotly map with GISA 2021 data
    from the model prediction.

    This works with GISA_2021 tiff (model prediction).

    -----------------

    Parameters:
        None
    -----------------

    Returns:
        - fig (plotly map): a map of the gisa 2021 prediction.
    '''

    # Open the GISA 2021 tif prediction
    gisa_2021 = rxr.open_rasterio(datadir / 'gisa-2021.tif').sel(band=1)

    # Get coords to create a basemap
    w = gisa_2021.coords['x'].min().item()
    s = gisa_2021.coords['y'].min().item()
    e = gisa_2021.coords['x'].max().item()
    n = gisa_2021.coords['y'].max().item()
    ctx.bounds2raster(
        w, s, e, n,
        'basemap_2021.tif',
        zoom=10, source=ctx.providers.Stamen.TonerLite, ll=True)
    basemap = rxr.open_rasterio(
        'basemap_2021.tif').sel(
            band=[1, 2, 3]).rio.reproject_match(
                gisa_2021,
                resampling=rio.enums.Resampling(2)).transpose('y', 'x', 'band')

    # Merge basemap with GISA2021 where will it be drawn in a coral color
    merge = basemap.values.copy()
    merge[np.where(gisa_2021.values > 0)] = (250, 128, 114)

    # Create the map with plotly
    fig = px.imshow(merge, labels=dict(x="", y=""))

    fig.update_layout(
        title_text='Superficie construida',
        title_x=0.5,
        font_size=17,
        font_color='rgb(142, 136, 131)',
        font_family='Franklin Gothic Book',
        title_font_size=25,
        title_font_color='rgb(99, 95, 93)',
        title_font_family='ITC Franklin Gothic Std',
        paper_bgcolor='white',
        plot_bgcolor='rgb(245, 243, 242)',
        width=800, height=800,
        margin=dict(l=20, r=20, t=50, b=20))

    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    return fig


# GISA SLIDER (past)
def gisa_slider(country, city, thresh=0.2):
    """ Plots historic built using an image overlay. """

    # Obtain bboxes and rasters
    bbox_mollweide, uc_mollweide, fua_mollweide = ru.get_bbox(
        city, country, Path('./data/output/cities/'),
        proj='ESRI:54009')
    bbox_latlon, uc_latlon, fua_latlon = ru.get_bbox(
        city, country, Path('./data/output/cities/'),
        proj='EPSG:4326')

    smod = ghsl.load_or_download(bbox_mollweide, 'SMOD',
                               data_path=datadir, resolution=1000)
    built = ghsl.load_or_download(bbox_mollweide, 'BUILT_S',
                                data_path=datadir, resolution=100)

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
    fig.update_layout(mapbox_center={'lat': (latmax + latmin)/2,
                                     'lon': (lonmax + lonmin)/2})

    fig.update_layout(
        mapbox_bounds={"west": west,
                       "east": east,
                       "south": south,
                       "north": north},
        height=1000, width=1000)

    # Create polygons of urban clusters and centers
    smod_p = ghsl.smod_polygons(smod, uc_mollweide.iloc[0].geometry.centroid)
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

    # fig.write_html('fig_image.html')

    return fig



def gisa_slider_old(country, city):
    '''
    GISA map animation with urbanization stats in the title.

    This works with GISA (2000-2019) tiffs.

    -----------------

    Parameters:
        - city (str): city to create the visuals for
        - country (str): country to create the visuals for
    -----------------

    Returns:
        - fig (plotly map animation): a map animation from the
        gisa 2000 to 2019 data.
    '''

    # Calculate total urb per year
    df_stats, df_largest = urbanization_df(city, country)
    urb_km = df_largest[['Area']]
    urb_km_list = urb_km.values.tolist()
    urb_km_list = [item for sublist in urb_km_list for item in sublist]

    # Make list of years for slider
    year_list = list(range(2000, 2020))

    # Geemap sets max width/height to 768
    max_dim = 768

    # Use a gisa (2019) raster as a reference to reduce resolution
    gisa_reference = rxr.open_rasterio(datadir / 'gisa-2019.tif').sel(band=1)
    resolution_dst = (max(gisa_reference.rio.resolution())
                      / max_dim * max(gisa_reference.shape))

    # Reproject to reduce resolution and
    # load gisa rasters into a list of xarrays
    gisa_list = [
        rxr.open_rasterio(
            datadir / f'gisa-{year}.tif').sel(band=1).rio.reproject(
                dst_crs=4326,
                resolution=resolution_dst)
        for year in year_list
    ]

    # Create xarray with year dimension
    gisa_xr = xr.concat(gisa_list, dim=pd.Index(year_list, name='year'))

    # Select limit coordinates to create the basemap
    w = gisa_xr.coords['x'].min().item()
    s = gisa_xr.coords['y'].min().item()
    e = gisa_xr.coords['x'].max().item()
    n = gisa_xr.coords['y'].max().item()
    ctx.bounds2raster(
        w, s, e, n,
        'basemap.tif',
        zoom=10, source=ctx.providers.Stamen.TonerLite, ll=True)
    basemap = rxr.open_rasterio(
        'basemap.tif').sel(
            band=[1, 2, 3]).rio.reproject_match(
                gisa_xr,
                resampling=rio.enums.Resampling(2)).transpose('y', 'x', 'band')

    # Merge the basemap with gisa
    gisa_ar = gisa_xr.values
    map_list = []
    for g in gisa_ar:
        merge = basemap.values.copy()
        merge[np.where(g > 0)] = (250, 128, 114)
        map_list.append(merge)

    # Change the list to and array and finally to an xarray
    map_ar = np.stack(map_list)
    map_xr = xr.DataArray(map_ar, dims=['year', 'y', 'x', 'band'])
    map_xr.coords['year'] = year_list

    # Create slider based on year
    fig = px.imshow(
        map_xr, aspect='equal', animation_frame='year',
        labels=dict(x="", y=""), width=map_ar.shape[2], height=map_ar.shape[1])

    fig.update_layout(
        title_x=0.5,
        font_size=17,
        font_color='rgb(142, 136, 131)',
        font_family='Franklin Gothic Book',
        title_font_size=25,
        title_font_color='rgb(99, 95, 93)',
        title_font_family='ITC Franklin Gothic Std',
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(l=20, r=20, t=50, b=20)
    )

    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 500

    for button in fig.layout.updatemenus[0].buttons:
        button['args'][1]['frame']['redraw'] = True

    for k in range(len(fig.frames)):
        fig.frames[k]['layout'].update(
            title_text=(f'Superficie construida en {year_list[k]}'
                        f' es de {round(urb_km_list[k], 2)} km2'))

    return fig


# LANDSAT GIF (past)

def landsat_animation(city, country):
    '''
    Landsat GIF created with a library from geemap.

    -----------------

    Parameters:
        - city (str): city to create the visuals for
        - country (str): country to create the visuals for
    -----------------

    Returns:
        - gif_str (str): the path for the landsat animation.
    '''

    # Create the path for the landsat animation
    gif_path = datadir / 'landsat_timelapse.gif'
    gif_str = str(gif_path)

    # Check if the path exists if not then creates it
    if not gif_path.exists():

        # Get coords for the selected city, country
        fua = cities_fua.loc[(cities_fua.country == country) &
                             (cities_fua.city == city)]
        bbox = dts.get_roi(fua, buff=0, buff_type='km').geometry[0]
        bbox_ee = ee.Geometry.Polygon(
               [t for t in zip(*bbox.exterior.coords.xy)])

        # Create the animation with geemap landsat_timelapse
        gp.landsat_timelapse(
            roi=bbox_ee, out_gif=gif_str,
            start_year=1990, end_year=2020,
            frames_per_second=2,
            title=f'{country}-{city}',
            font_size=30,
            bands=['Red', 'Green', 'Blue'],
            apply_fmask=True,
            fading=False)
        return gif_str

    else:
        return gif_str


# SLEUTH MAPS (future)

def sleuth_map(country, city, src, title):
    '''
    Creates a sleuth map with a basemap from GISA 2021
    it works for the three categories of sleuth: slow,
    usual and fast grow.

    This works with sleuth 2040 tiff (from the model
    prediction)and GISA 2021.
    -----------------

    Parameters:
        - city (str): city to create the visuals for
        - country (str): country to create the visuals for
        - src (str): which data source to create the map for
        - title (str): title for the map
    -----------------

    Returns:
        - fig (plotly map): a map of the 2040 sleuth prediction.
    '''

    # Open the 2040 sleuth prediction and the GISA 2021 prediction rasters
    sleuth = rxr.open_rasterio(datadir / src).sel(band=1)
    gisa_2021 = rxr.open_rasterio(datadir / 'gisa-2021.tif').sel(band=1)

    # Get the coords to create the basemap
    w = gisa_2021.coords['x'].min().item()
    s = gisa_2021.coords['y'].min().item()
    e = gisa_2021.coords['x'].max().item()
    n = gisa_2021.coords['y'].max().item()
    ctx.bounds2raster(
        w, s, e, n,
        'basemap_2021.tif',
        zoom=10, source=ctx.providers.Stamen.TonerLite, ll=True)
    basemap = rxr.open_rasterio(
        'basemap_2021.tif').sel(
            band=[1, 2, 3]).rio.reproject_match(
                sleuth,
                resampling=rio.enums.Resampling(2)).transpose('y', 'x', 'band')

    # Merge the basemap with the 2040 sleuth data
    merge = basemap.values.copy()
    merge[np.where(sleuth.values > 0)] = (250, 128, 114)

    # Create the plotly map
    fig = px.imshow(merge, labels=dict(x="", y=""))

    fig.update_layout(
        title_text=title,
        title_x=0.5,
        font_size=17,
        font_color='rgb(142, 136, 131)',
        font_family='Franklin Gothic Book',
        title_font_size=25,
        title_font_color='rgb(99, 95, 93)',
        title_font_family='ITC Franklin Gothic Std',
        paper_bgcolor='white',
        plot_bgcolor='white',
        width=500, height=500,
        margin=dict(l=20, r=20, t=50, b=20))

    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    return fig


# SLEUTH DATAFRAME (future)

def sleuth_df(datadir):
    '''
    Creates a dataframe of the sleuth data predictions
    from 2020 to 2040 and slow, usual and fast categories.

    This works with sleuth 2040 tiff (from the model
    prediction)and GISA 2021.
    -----------------

    Parameters:
        - datadir (str): the datadir of the city selected
        to save the dataframe to.
    -----------------

    Returns:
        - df (dataframe): a dataframe of the sleuth predictions.
    '''

    dfpath = datadir / 'sleuth_projections.csv'

    if not dfpath.exists():

        print('Downloading sleuth projections csv...')
        df = sts.get_sleuth_dataframe(datadir)
        df.to_csv(datadir / 'sleuth_projections.csv', index=False)
        print('Done.')
        return df

    else:
        print('Sleuth dataframe already downloaded.')
        return pd.read_csv(dfpath)


# SLEUTH GRAPHS (future)

def sleuth_graph():
    '''
    Creates a line plot graph to compare the slow,
    usual and fast cases of the sleuth predictions.

    This works with sleuth 2020 to 2040 predictions.
    -----------------

    Parameters:
        None
    -----------------

    Returns:
        - fig (plotly graph): a plotly line plot.
    '''

    # Preprocessing of the dataframe
    df = sleuth_df(datadir)
    df['mode'] = df['mode'].replace(
        ['slow', 'fast', 'usual'],
        ['lento', 'rápido', 'crecimiento inercial'])
    df.columns = ['escenario', 'año', 'no urbanizado', 'área urbanizada']

    # Creates the line plot with plotly
    fig = px.line(
        df, x='año', y='área urbanizada',
        color='escenario', markers=True, title=' ')

    fig.update_layout(
        xaxis_title='Año',
        yaxis_title='Área urbanizada',
        font_size=17,
        font_color='rgb(142, 136, 131)',
        font_family='Franklin Gothic Book',
        title_font_size=25,
        title_font_color='rgb(99, 95, 93)',
        title_font_family='ITC Franklin Gothic Std',
        title_xanchor='left',
        title_yanchor='top',
        title_x=0.22,
        title_y=0.95,
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(l=20, r=20),
        legend_title="Escenarios: ",
        legend=dict(orientation="h", y=1.2)
    )

    fig.update_xaxes(
        showgrid=True, gridwidth=1, gridcolor='rgb(245, 243, 242)')
    fig.update_yaxes(
        showgrid=True, gridwidth=1, gridcolor='rgb(245, 243, 242)')

    return fig


# GIF DOWNLOAD (past)

def download_landsat_thumb(country, city):
    '''
    Download the landsat data locally and does
    a preprocessing to get a better quality
    (without clouds).

    This works with landsat tiff from 1990 to 2019.
    -----------------

    Parameters:
        - city (str): city to create the visuals for
        - country (str): country to create the visuals for
    -----------------

    Returns:
        - filenames (list): a list of all the landsat paths.
    '''

    filenames = []
    year_list = list(range(1990, 2020))

    for year in year_list:

        print(f'Downloading landsat thumb year {year}...')

        # Get region of interest (bounding box)
        cities_fua['burnin'] = 1
        fua = cities_fua.loc[(cities_fua.country == country) &
                             (cities_fua.city == city)]

        bbox = dts.get_roi(fua, buff=0, buff_type='km').geometry[0]
        bbox_ee = ee.Geometry.Polygon(
            [t for t in zip(*bbox.exterior.coords.xy)])

        landsat_ee = dts.load_landsat(
            bbox, year, reproj=False, col=1, clip=True).multiply(0.0001)
        landsat_vis = landsat_ee.visualize(
            bands=['Red', 'Green', 'Blue'], min=0, max=0.4,
            gamma=[1, 1, 1]).clip(bbox_ee)
        vis_params = {
            "min": 0,
            "max": 255,
            "bands": ["vis-red", "vis-green", "vis-blue"],
            "format": 'jpg',
            'region': bbox_ee,
            'dimensions': 768
        }

        url = landsat_vis.getThumbUrl(vis_params)
        r = requests.get(url, stream=True)
        filename = datadir / f'landsat_{year}.jpg'

        with open(filename, "wb") as fd:
            for chunk in r.iter_content(chunk_size=1024):
                fd.write(chunk)

        filenames.append(str(filename))

    return filenames


def draw_year_landsatIMG(country, city):
    '''
    Draws the year in each landsat image.

    This works with landsat tiff from 1990 to 2019.
    -----------------

    Parameters:
        - city (str): city to create the visuals for
        - country (str): country to create the visuals for
    -----------------

    Returns:
        - filenames_drawn (list): a list of all the landsat drawn
        paths.
    '''
    filenames = download_landsat_thumb(country, city)

    # Get the font to draw with
    title_font = ImageFont.truetype('/Library/Fonts/Arial.ttf', 40)
    filenames_drawn = []

    # For each landsat image open it, draw it,
    # save it and append its path to the list
    for filename in filenames:
        img = Image.open(filename)
        ImageDraw.Draw(img).text(
            (40, 36), filename.split('_')[1][:-4], fill=(0, 0, 0),
            font=title_font)
        img.save(f'{filename[:-4]}_draw.jpg')
        filenames_drawn.append(f'{filename[:-4]}_draw.jpg')

    return filenames_drawn


def make_gif(country, city):
    '''
    Creates a Landsat gif.

    This works with landsat tiff from 1990 to 2019.
    -----------------

    Parameters:
        - city (str): city to create the visuals for
        - country (str): country to create the visuals for
    -----------------

    Returns:
        - filenames_drawn (list): a list of all the landsat drawn
        paths.
    '''
    # Get the file path from the landsats drawn
    filenames = draw_year_landsatIMG(country, city)

    # Create the path for the landsat GIF
    gif_name = f'{str(datadir)}/landsatGIF_{country}_{city}.gif'

    # Create the GIF by appending them whilist saving them
    frames = [Image.open(image) for image in filenames]
    frame_one = frames[0]
    frame_one.save(gif_name, format="GIF", append_images=frames,
                   save_all=True, duration=400, loop=0)

    return gif_name
