import numpy as np
import geopandas as gpd
import ee
import geemap
import geemap.colormaps as cm
from shapely.geometry import Polygon
import osmnx as ox
import xarray as xr
import rioxarray as rxr
from geocube.api.core import make_geocube
from scipy.spatial import KDTree
import requests
from pathlib import Path
import warnings

import re
import os
import glob
import pickle
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

data_dir = Path('./data/')

sources = {
    'WDPA': {
        'class': 'landcover',
        'type': 'FeatureCollection',
        'multi_temp': False,
        'bands': ['any'],
        'select_bands': ['any'],
        'source': 'WCMC/WDPA/current/polygons',
        'temp_cov': [2021],
        'crs': 'EPSG:4326',
        'scale': 1,
        'vis_params': {},
        'class_dict': {0: 'Not protected',
                       1: 'Protected'},
        'toraster_prop': 'WDPAID',
        'filterby': ('STATUS', 'Designated')
    },
    'OSM_waterLayer': {
        'class': 'landcover',
        'type': 'Collection',
        'multi_temp': False,
        'bands': ['b1'],
        'select_bands': ['b1'],
        'source': 'projects/sat-io/open-datasets/OSM_waterLayer',
        'temp_cov': [2021],
        'crs': 'EPSG:4326',
        'scale': 92.76624232772798,
        'vis_params': {"min": 1, "max": 5,
                       "palette": ["08306b", "08519c",
                                   "2171b5", "4292c6", "6baed6"]},
        'class_dict': {1: 'Ocean',
                       2: 'Large Lake/River',
                       3: 'Major River',
                       4: 'Canal',
                       5: 'Small Stream'}
    },
    'Geomorpho90-slope': {
        'class': 'continuous',
        'type': 'Collection',
        'multi_temp': False,
        'bands': ['b1'],
        'select_bands': ['b1'],
        'source': 'projects/sat-io/open-datasets/Geomorpho90m/slope',
        'temp_cov': [2020],
        'crs': 'EPSG:4326',
        'scale': 92.76624232772798,
        'vis_params': {'min': 0.2, 'max': 25}
    },
    'GISA': {
        'class': 'urban',
        'type': 'Collection',
        'multi_temp': True,
        'bands': ['b1'],
        'select_bands': ['b1'],
        'source': 'projects/sat-io/open-datasets/GISA_1972_2019',
        'temp_cov': [1972, 1978] + list(range(1985, 2020)),
        'crs': 'EPSG:4326',
        'scale': 30.000000000000004,
        'year_dict': {**{1972: 1, 1978: 2},
                      **{k: v for k, v in zip(range(1985,  2020),
                                              range(3, 38))}},
        'year_inverse': False,
        'vis_params': {
            'min': 1,
            'max': 1
        }
    },
    'WSF-Evo': {
        'class': 'urban',
        'type': 'Collection',
        'multi_temp': True,
        'bands': ['b1'],
        'select_bands': ['b1'],
        'source': 'projects/sat-io/open-datasets/WSF/WSF_EVO',
        'temp_cov': list(range(1985, 2016)),
        'crs': 'EPSG:4326',
        'scale': 30.000000000000004,
        'year_inverse': False,
        'vis_params': {
            'min': 1,
            'max': 1
        }
    },
    'WSF-2015': {
        'class': 'urban',
        'type': 'Image',
        'multi_temp': True,
        'bands': ['settlement'],
        'select_bands': ['settlement'],
        'source': 'DLR/WSF/WSF2015/v1',
        'temp_cov': [2015],
        'crs': 'EPSG:4326',
        'scale': 10,
        'year_dict': {2015: 255},
        'year_inverse': False,
        'vis_params': {
            'min': 1,
            'max': 1
        }
    },
    'WSF-2019': {
        'class': 'urban',
        'type': 'Collection',
        'multi_temp': True,
        'bands': ['b1'],
        'select_bands': ['b1'],
        'source': 'projects/sat-io/open-datasets/WSF/WSF_2019',
        'temp_cov': [2019],
        'crs': 'EPSG:4326',
        'scale': 10,
        'year_dict': {2019: 255},
        'year_inverse': False,
        'vis_params': {
            'min': 1,
            'max': 1
        }
    },
    'GHSL Built-Up': {
        'class': 'urban',
        'type': 'Image',
        'multi_temp': True,
        'bands': ['built', 'cnfd', 'dm'],
        'select_bands': ['built'],
        'source': 'JRC/GHSL/P2016/BUILT_LDSMT_GLOBE_V1',
        'temp_cov': [1975, 1990, 2000, 2015],
        'crs': 'EPSG:3857',
        'scale': 38.218514142588525,
        'year_dict': {2015: 3, 2000: 4, 1990: 5, 1975: 6},
        'year_inverse': True,
        'vis_params': {
            'min': 1,
            'max': 1
        }
    },
    'GHSL Settlements': {
        'class': 'urban',
        'type': 'Image',
        'multi_temp': False,
        'bands': ['smod_code'],
        'select_bands': ['smod_code'],
        'source': 'JRC/GHSL/P2016/SMOD_POP_GLOBE_V1/{year}',
        'temp_cov': [1975, 1990, 2000, 2015],
        'crs': 'WKT: World_Mollweide',
        'scale': 1000,
        'vis_params': {'min': 0.0, 'max': 3.0,
                       'palette': ['000000', '448564',
                                   '70daa4', 'ffffff']},
        'class_dict': {0: 'Inhabited areas',
                       1: 'RUR (rural grid cells)',
                       2: 'LDC (low density clusters)',
                       3: 'HDC (high density clusters)'}
    },
    'ESA': {
        'class': 'landcover',
        'type': 'Image',
        'multi_temp': False,
        'bands': ['Map'],
        'select_bands': ['Map'],
        'source': 'ESA/WorldCover/v100/2020',
        'temp_cov': [2020],
        'crs': 'EPSG:4326',
        'scale': 9.276624232772797,
        'vis_params': {},
        'class_dict': {10: 'Trees',
                       20: 'Shrubland',
                       30: 'Grassland',
                       40: 'Cropland',
                       50: 'Built-up',
                       60: 'Barren / sparse vegetation',
                       70: 'Snow and ice',
                       80: 'Open water',
                       90: 'Herbaceous wetland',
                       95: 'Mangroves',
                       100: 'Moss and lichen'
                       },
        'builtin_legend': 'ESA_WorldCover'
    },
    'DynamicWorld': {
        'class': 'landcover',
        'type': 'Collection',
        'multi_temp': True,
        'bands': ['water',
                  'trees',
                  'grass',
                  'flooded_vegetation',
                  'crops',
                  'shrub_and_scrub',
                  'built',
                  'bare',
                  'snow_and_ice',
                  'label'],
        'class_dict':  {'0': 'Agua',
                        '1': 'Árboles',
                        '2': 'Césped/Pasto',
                        '3': 'Vegetación inundada',
                        '4': 'Cultivos',
                        '5': 'Arbusto y matorral',
                        '6': 'Urbanización',
                        '7': 'Descubierto',
                        '8': 'Nieve y hielo'},
        'select_bands': ['label'],
        'source': 'GOOGLE/DYNAMICWORLD/V1',
        'temp_cov': list(range(2016, 2022)),
        'scale': 10,
        'vis_params': {'min': 0, 'max': 8,
                       'palette': ['#419BDF', '#397D49', '#88B053', '#7A87C6',
                                   '#E49635', '#DFC35A', '#C4281B', '#A59B8F',
                                   '#B39FE1']},
        'colors': {'Agua': '#419BDF',
                   'Árboles': '#397D49',
                   'Césped/Pasto': '#88B053',
                   'Vegetación inundada': '#7A87C6',
                   'Cultivos': '#E49635',
                   'Arbusto y matorral': '#DFC35A',
                   'Urbanización': '#C4281B',
                   'Descubierto': '#A59B8F',
                   'Nieve y hielo': '#B39FE1',
                   'Desconocido': 'gray'}
    },
    'Copernicus': {
        'class': 'landcover',
        'type': 'Image',
        'multi_temp': False,
        'bands': ['discrete_classification',
                  'discrete_classification-proba',
                  'bare-coverfraction',
                  'urban-coverfraction',
                  'crops-coverfraction',
                  'grass-coverfraction',
                  'moss-coverfraction',
                  'water-permanent-coverfraction',
                  'water-seasonal-coverfraction',
                  'shrub-coverfraction',
                  'snow-coverfraction',
                  'tree-coverfraction',
                  'forest_type',
                  'data-density-indicator',
                  'change-confidence'],
        'select_bands': ['discrete_classification'],
        'source': 'COPERNICUS/Landcover/100m/Proba-V-C3/Global/{year}',
        'temp_cov': list(range(2015, 2020)),
        'crs': 'EPSG:3857',
        'scale': 100,
        'vis_params': {},
        'class_dict':
        {0: 'Unknown',
         20: 'Shrubs',
         30: 'Herbaceous vegetation',
         40: 'Cultivated and managed vegetation / agriculture',
         50: 'Urban / built up',
         60: 'Bare / sparse vegetation',
         70: 'Snow and ice',
         80: 'Permanent water bodies',
         90: 'Herbaceous wetland',
         100: 'Moss and lichen',
         111: 'Closed forest, evergreen needle leaf',
         112: 'Closed forest, evergreen broad leaf',
         113: 'Closed forest, deciduous needle leaf',
         114: 'Closed forest, deciduous broad leaf',
         115: 'Closed forest, mixed',
         116: 'Closed forest, not matching any of the other definitions',
         121: 'Open forest, evergreen needle leaf',
         122: 'Open forest, evergreen broad leaf',
         123: 'Open forest, deciduous needle leaf',
         124: 'Open forest, deciduous broad leaf',
         125: 'Open forest, mixed',
         126: 'Open forest, not matching any of the other definitions',
         200: 'Oceans, seas'},
        'remap_dict':
        {0: 'Unknown',
         20: 'Shrubs',
         30: 'Grasslands',
         40: 'Agriculture',
         50: 'Urban',
         60: 'Bare',
         70: 'Snow and ice',
         80: 'Water',
         90: 'Other vegetation',
         100: 'Other vegetation',
         111: 'Forest',
         112: 'Forest',
         113: 'Forest',
         114: 'Forest',
         115: 'Forest',
         116: 'Forest',
         121: 'Forest',
         122: 'Forest',
         123: 'Forest',
         124: 'Forest',
         125: 'Forest',
         126: 'Forest',
         200: 'Water'},
        'builtin_legend': 'COPERNICUS/Landcover/100m/Proba-V/Global'
    },
    'GHSL Population': {
        'class': 'population',
        'type': 'Image',
        'multi_temp': False,
        'bands': ['population_count'],
        'select_bands': ['population_count'],
        'source': 'JRC/GHSL/P2016/POP_GPW_GLOBE_V1/{year}',
        'temp_cov': [1975, 1990, 2000, 2015],
        'crs': 'WKT: World_Mollweide',
        'scale': 250,
        'vis_params': {'min': 0.0, 'max': 200.0,
                       'palette': ['060606', '337663',
                                   '337663', 'ffffff']},
    },
    'WorldPop': {
        'class': 'population',
        'type': 'Image',
        'multi_temp': False,
        'bands': ['population'],
        'select_bands': ['population'],
        'source': 'WorldPop/GP/100m/pop/{country}_{year}',
        'temp_cov': list(range(2000, 2021)),
        'crs': 'EPSG:4326',
        'scale': 92.7662419566,
        'vis_params': {'min': 0.0, 'max': 50.0,
                       'palette': ['24126c', '1fff4f', 'd4ff50']},
    },
    'CGAZ_ADM0': {
        'class': 'administrative',
        'type': 'FeatureCollection',
        'source': ('projects/earthengine-legacy/assets/projects/'
                   'sat-io/open-datasets/geoboundaries/CGAZ_ADM0'),
        'crs': '',
        'scale': 0,
        'vis_params': {},
    }
}


def km_2_lat(d):
    # radius of the earth
    R = 6371
    # conversion
    lat = (180/np.pi) * d/R

    return lat


def km_2_lon(d, lat):
    # radius of the earth
    R = 6371
    # conversion
    lat_rad = np.pi/180 * lat
    lon_rad = d/(R * np.cos(lat_rad))

    return 180/np.pi * lon_rad


def get_roi(city_gdf, buff=1, buff_type='scale'):
    poly = city_gdf.geometry.values[0]
    bounds = poly.bounds
    delta_lat = bounds[3] - bounds[1]
    delta_lon = bounds[2] - bounds[0]
    mid_lat = (bounds[3] + bounds[1])/2
    mid_lon = (bounds[2] + bounds[0])/2

    if buff_type == 'scale':
        delta_lat *= buff
        delta_lon *= buff
    elif buff_type == 'km':
        buff_lat = km_2_lat(buff)
        delta_lat += 2*buff_lat
        buff_lon = km_2_lon(buff, mid_lat)
        delta_lon += 2*buff_lon

    lat_max = mid_lat + delta_lat/2
    lat_min = mid_lat - delta_lat/2
    lon_max = mid_lon + delta_lon/2
    lon_min = mid_lon - delta_lon/2

    bbox = Polygon([(lon_min, lat_min), (lon_max, lat_min),
                    (lon_max, lat_max), (lon_min, lat_max)])

    bbox_gdf = gpd.GeoDataFrame({'name': 'bbox', 'geometry': bbox},
                                index=[0], crs=city_gdf.crs)

    return bbox_gdf


def get_tlist(col):

    def get_proj(img, first):
        return ee.List(first).add(img.projection())

    tlist = [p['transform'] for p in
             col.iterate(get_proj, ee.List([])).getInfo()]

    return tlist


def get_trans(col):

    tlist = get_tlist(col)

    xScale_l       = [t[0] for t in tlist]
    xShearing_l    = [t[1] for t in tlist]
    xTranslation_l = [t[2] for t in tlist]
    yShearing_l    = [t[3] for t in tlist]
    yScale_l       = [t[4] for t in tlist]
    yTranslation_l = [t[5] for t in tlist]

    assert len(np.unique(xScale_l)) == 1
    assert len(np.unique(xShearing_l)) == 1
    assert len(np.unique(yShearing_l)) == 1
    assert len(np.unique(yScale_l)) == 1

    xScale = xScale_l[0]
    xShearing = xShearing_l[0]
    yShearing = yShearing_l[0]
    yScale = yScale_l[0]

    if xScale > 0:
        xTranslation = min(xTranslation_l)
    else:
        xTranslation = max(xTranslation_l)

    if yScale > 0:
        yTranslation = min(yTranslation_l)
    else:
        yTranslation = max(yTranslation_l)

    transform = [xScale, xShearing, xTranslation,
                 yShearing, yScale, yTranslation]

    return [float(t) for t in transform]


def get_col_crs(col):

    def get_crs_iter(img, first):
        return ee.List(first).add(img.projection().crs())

    crs_l = col.iterate(get_crs_iter, ee.List([])).getInfo()

    assert len(np.unique(crs_l)) == 1

    return crs_l[0]


def get_col_scale(col):

    def get_scale_iter(img, first):
        return ee.List(first).add(img.projection().nominalScale())

    scale_l = col.iterate(get_scale_iter, ee.List([])).getInfo()

    assert len(np.unique(scale_l)) == 1

    return scale_l[0]


def get_img_crs(img):
    proj = img.projection().getInfo()
    if 'crs' in proj.keys():
        crs = proj['crs']
        return crs
    else:
        return 'WKT: ' + proj['wkt'].split('"')[1]


def load_src(src, year, roi=None, sdict=None, clip=False, country=None):
    if roi is None:
        roi = ee.Geometry.Polygon([[-89.752197, 20.844071],
                                   [-89.752197, 21.102812],
                                   [-89.485474, 21.102812],
                                   [-89.485474, 20.844071],
                                   [-89.752197, 20.844071]])

    if sdict is None:
        sdict = sources
    pdict = sdict[src]

    # country = ''
    if 'country' in pdict['source']:
        # Use roi centroid to define 3 letter country ISO code.
        # Are there crosscountry cities?

        # Load adm0 feature collection
        if country is not None:
            data_path = './data/'
            cities_uc = gpd.read_file(
                data_path + 'output/cities/cities_uc.gpkg')
            country_iso = cities_uc[cities_uc["country"] == country]
            country_iso = country_iso["Cntry_ISO"].values[0]
            country = country_iso
        else:
            countries = ee.FeatureCollection(sdict['CGAZ_ADM0']['source'])
            countries = countries.filterBounds(roi)
            # assert countries.size().getInfo() == 1
            country = countries.first().get('shapeISO').getInfo()

    src_addrs = pdict['source'].format(year=year, country=country)
    if year not in pdict['temp_cov']:
        raise NotImplementedError

    # Load image
    if pdict['type'] == 'Collection':
        col = ee.ImageCollection(src_addrs)
        if roi is not None:
            col = col.filterBounds(roi)
        crs = get_col_crs(col)
        transform = get_trans(col)
        image = col.mosaic()
        image = image.reproject(crs=crs, crsTransform=transform)
    elif pdict['type'] == 'Image':
        image = ee.Image(src_addrs)

    if pdict['type'] == 'FeatureCollection':
        col = ee.FeatureCollection(src_addrs)
        if roi is not None:
            col = col.filterBounds(roi)
        filterby = pdict['filterby']
        if filterby is not None:
            prop = filterby[0]
            value = filterby[1]
            col = col.filter(ee.Filter.eq(prop, value))
        toraster_prop = pdict['toraster_prop']
        assert toraster_prop is not None
        image = ee.FeatureCollection.reduceToImage(
            col, [toraster_prop], ee.Reducer.anyNonZero())
        # Set up crs and scale
        utm = load_utm(data_dir)
        max_epsg = utm.CRS.loc[
            utm.intersection(
                geemap.ee_to_gdf(ee.FeatureCollection(roi)).geometry[0]
            ).area.idxmax()]
        # scale = 30  # To match landsat
        transform = [30, 0, 15, 0, -30, 15]
        image = image.reproject(crs=max_epsg, crsTransform=transform)

    # Select band
    image = image.select(pdict['select_bands'])

    # Temporal filter for multi-temporal images
    if pdict['multi_temp']:
        if 'year_dict' in pdict.keys():
            year = pdict['year_dict'][year]
        if pdict['year_inverse']:
            image = image.gte(year)
        else:
            image = image.lte(year)
        image = image.updateMask(image)

    if clip and roi is not None:
        image = image.clip(roi)

    return image


def map_src(Map, src, year, roi=None, sdict=None, clip=False, country=None):
    if roi is None:
        roi = ee.Geometry.Polygon([[-89.752197, 20.844071],
                                   [-89.752197, 21.102812],
                                   [-89.485474, 21.102812],
                                   [-89.485474, 20.844071],
                                   [-89.752197, 20.844071]])

    if sdict is None:
        sdict = sources

    image = load_src(src, year, roi, sdict, clip, country=country)
    pdict = sdict[src]
    Map.addLayer(image, pdict['vis_params'], src + ' ' + str(year))
    if pdict['type'] == 'population':
        Map.add_colorbar(pdict['vis_params'],
                         label='Population count',
                         layer_name=src + ' ' + str(year))

    if 'builtin_legend' in pdict.keys():
        Map.add_legend(builtin_legend=pdict['builtin_legend'],
                       layer_name=src + ' year',
                       position='bottomleft')
    elif 'class_dict' in pdict.keys():
        legend_dict = {f'{i} {t}': c
                       for i, t, c in
                       zip(pdict['class_dict'].keys(),
                           pdict['class_dict'].values(),
                           pdict['vis_params']['palette'])}
        Map.add_legend(legend_title=src,
                       legend_dict=legend_dict,
                       layer_name=src + ' ' + str(year),
                       position='bottomleft')


def load_hrsl(roi=None):
    source = 'projects/sat-io/open-datasets/hrsl/hrslpop'
    col = ee.ImageCollection(source)
    if roi is not None:
        col = col.filterBounds(roi)
    crs = get_col_crs(col)
    transform = get_trans(col)
    image = col.mosaic()
    image = image.reproject(crs=crs, crsTransform=transform)

    return image


def map_hrsl(Map, roi=None):
    vis_params = {'min': 0.0, 'max': 50.0,
                  'palette': ['24126c', '1fff4f', 'd4ff50']}
    image = load_hrsl(roi)
    Map.addLayer(image, vis_params, 'HRSL (Facebook)')
    Map.add_colorbar(vis_params, label='Population count',
                     layer_name='HRSL (Facebook)')


def fmask(image):
    qa = image.select('QA_PIXEL')

    dilated_cloud_bit = 1
    cloud_bit = 3
    cloud_shadow_bit = 4
    snow_bit = 5
    clear_bit = 6

    mask = qa.bitwiseAnd(1 << dilated_cloud_bit).eq(0)
    mask = mask.And(qa.bitwiseAnd(1 << cloud_bit).eq(0))
    mask = mask.And(qa.bitwiseAnd(1 << cloud_shadow_bit).eq(0))
    # mask = mask.And(qa.bitwiseAnd(1 << snow_bit).eq(0))
    # mask = qa.bitwiseAnd(1 << clear_bit)

    # keep previous mask reduced over all bands
    # TODO: verify this is correct, specially consider
    # the bands included in the reducer. Maybe we don't want
    # all bands, so keep only SR bands
    # org_mask = image.mask().reduce(ee.Reducer.min())
    return image.updateMask(mask)  # .updateMask(org_mask)


def fmask_1(image):
    qa = image.select('pixel_qa')

    cloud_bit = 5
    cloud_shadow_bit = 3

    mask = qa.bitwiseAnd(1 << cloud_bit).eq(0)
    mask = mask.And(qa.bitwiseAnd(1 << cloud_shadow_bit).eq(0))

    return image.updateMask(mask)


def renameOli(img):
    # Function to get and rename bands of interest from OLI.
    return img.select(
        ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7", "QA_PIXEL"],
        ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2", "QA_PIXEL"],
    )


def renameOli_1(img):
    # Function to get and rename bands of interest from OLI.
    return img.select(
        ["B2", "B3", "B4", "B5", "B6", "B7", "pixel_qa"],
        ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2", "pixel_qa"],
    )


def renameEtm(img):
    # Function to get and rename bands of interest from ETM+.
    return img.select(
        ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7", "QA_PIXEL"],
        ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2", "QA_PIXEL"],
    )


def renameEtm_1(img):
    # Function to get and rename bands of interest from ETM+.
    return img.select(
        ["B1", "B2", "B3", "B4", "B5", "B7", "pixel_qa"],
        ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2", "pixel_qa"],
    )


def prepOli(img):
    # Define function to prepare OLI images.
    orig = img
    img = renameOli(img)
    img = fmask(img)
    # Drop QA as to not affect geomedian
    img = img.select(["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"])
    img = ee.Image(img.copyProperties(orig, orig.propertyNames()))
    img = img.set({'epsg': orig.projection().crs()})
    return img.resample("bicubic")


def prepEtm(img):
    # Define function to prepare ETM+ images.
    orig = img
    img = renameEtm(img)
    img = fmask(img)
    # Drop QA as to not affect geomedian
    img = img.select(["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"])
    img = ee.Image(img.copyProperties(orig, orig.propertyNames()))
    img = img.set({'epsg': orig.projection().crs()})
    return img.resample("bicubic")


def prepOli_1(img):
    # Define function to prepare OLI images.
    orig = img
    img = renameOli_1(img)
    img = fmask_1(img)
    # Drop QA as to not affect geomedian
    img = img.select(["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"])
    img = ee.Image(img.copyProperties(orig, orig.propertyNames()))
    img = img.set({'epsg': orig.projection().crs()})
    return img.resample("bicubic")


def prepEtm_1(img):
    # Define function to prepare ETM+ images.
    orig = img
    img = renameEtm_1(img)
    img = fmask_1(img)
    # Drop QA as to not affect geomedian
    img = img.select(["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"])
    img = ee.Image(img.copyProperties(orig, orig.propertyNames()))
    img = img.set({'epsg': orig.projection().crs()})
    return img.resample("bicubic")


def load_utm(data_dir):
    utm_path = data_dir / 'input/UTM_Zone_Boundaries.gpkg'
    if not utm_path.exists():
        utm_path = Path('data/UTM_Zone_Boundaries.gpkg')
    utm = gpd.read_file(utm_path)
    utm['CRS'] = utm.apply(
        lambda x: ('EPSG:327' if x.HEMISPHERE == 's' else 'EPSG:326')
        + x.ZONE, axis=1)
    return utm


def load_landsat(roi=None, year=2021,
                 clip=False, col=2, reproj=False):
    assert year in list(range(1985, 2022))

    startDate = f'{year}-01-01'
    endDate = f'{year}-12-31'

    if roi is None:
        roi = Polygon([[-89.752197, 20.844071],
                       [-89.752197, 21.102812],
                       [-89.485474, 21.102812],
                       [-89.485474, 20.844071],
                       [-89.752197, 20.844071]])
    elif isinstance(roi, gpd.GeoDataFrame):
        roi = roi.iloc[0].geometry

    roi_ee = ee.Geometry.Polygon(
        [t for t in zip(*roi.exterior.coords.xy)])

    if col == 2:
        LS4 = ee.ImageCollection("LANDSAT/LT04/C02/T1_L2")
        LS5 = ee.ImageCollection("LANDSAT/LT05/C02/T1_L2")
        LS7 = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2')
        LS8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
    elif col == 1:
        LS4 = ee.ImageCollection("LANDSAT/LT04/C01/T1_SR")
        LS5 = ee.ImageCollection("LANDSAT/LT05/C01/T1_SR")
        LS7 = ee.ImageCollection("LANDSAT/LE07/C01/T1_SR")
        LS8 = ee.ImageCollection("LANDSAT/LC08/C01/T1_SR")

    # Filter for year and roi
    LS4 = LS4.filterDate(startDate, endDate).filterBounds(roi_ee)
    LS5 = LS5.filterDate(startDate, endDate).filterBounds(roi_ee)
    LS7 = LS7.filterDate(startDate, endDate).filterBounds(roi_ee)
    LS8 = LS8.filterDate(startDate, endDate).filterBounds(roi_ee)

    # Select, rename and mask
    if col == 2:
        LS4 = LS4.map(prepEtm)
        LS5 = LS5.map(prepEtm)
        LS7 = LS7.map(prepEtm)
        LS8 = LS8.map(prepOli)
    elif col == 1:
        LS4 = LS4.map(prepEtm_1)
        LS5 = LS5.map(prepEtm_1)
        LS7 = LS7.map(prepEtm_1)
        LS8 = LS8.map(prepOli_1)

    collection = LS4.merge(LS5).merge(LS7).merge(LS8)
    # return collection

    # Get a list of projections included in the collection
    # def get_utm(img, first):
    #     return ee.List(first).add(img.projection().crs())
    # utm_list = [crs for crs in
    #             collection.iterate(get_utm, ee.List([])).getInfo()]
    # utm_list = np.unique(utm_list)
    # Find overlaps on UTM zones and get zone with max overlap

    # Ignore warning raised when interection area
    # is calculated in geographic units for utm zone assesment
    warnings.simplefilter("ignore", UserWarning)

    utm = load_utm(data_dir)
    max_epsg = utm.CRS.loc[utm.intersection(roi).area.idxmax()]
    # Fix utm south problem in landsat collection
    # Landsat only uses north utm zones
    if max_epsg[7] == '7':
        filter_epsg = max_epsg[:7] + '6' + max_epsg[8:]
        adj_trans = 10000000
    else:
        adj_trans = 0
        filter_epsg = max_epsg
    # get transform for corresponding image
    proj = collection.filterMetadata('epsg', 'equals', filter_epsg)
    proj = proj.first().projection().getInfo()
    crs = max_epsg
    transform = proj['transform']
    transform[5] += adj_trans

    # geomedian composite
    bands = collection.first().bandNames()
    image = collection.reduce(ee.Reducer.geometricMedian(6))
    # image = collection.reduce(ee.Reducer.median())

    image = image.set(
        {
         "year": year,
         "system:time_start": ee.Date(startDate).millis(),
         "nBands": bands.size(),
         "system:date": ee.Date(startDate).format("YYYY"),
        }).select(image.bandNames(), bands)

    if clip:
        image = image.clip(roi_ee)

    if reproj:
        image = image.reproject(crs=crs, crsTransform=transform)

    return image


def download_image(img, roi, year, folder, name):
    task_config = {
        'crs': img.projection().crs(),
        'region': roi,
        'folder': folder,
        'crsTransform': img.projection().getInfo()['transform']
    }
    task = ee.batch.Export.image(img, name, task_config)
    task.start()
    return task


def map_landsat(Map, roi, year=2021, clip=True,
                false_color=False, col=2):
    image = load_landsat(roi, year, clip, col=col)

    # Apply scaling factor and offset
    if col == 2:
        image = image.multiply(0.0000275).add(-0.2)
    elif col == 1:
        image = image.multiply(0.0001)

    if false_color:
        bands = ['NIR', 'SWIR1', 'Red']
    else:
        bands = ['Red', 'Green', 'Blue']

    Map.addLayer(image, {
        'bands': bands,
        'min': 0.0,
        'max': 0.3,
    }, 'LS')


def local_download(img, roi, folder, basename):

    dir_path = Path(folder)
    file_path = dir_path / basename

    if not isinstance(img, ee.Image):
        print("The ee_object must be an ee.Image.")
        return

    params = {
        'name': basename,
        'filePerBand': False,
        'region': roi,
        'crs': img.projection().crs(),
        'crs_transform': img.projection().getInfo()['transform'],
        'format': 'GEO_TIFF'
    }

    try:
        # print("Generating URL ...")

        try:
            url = img.getDownloadURL(params)
        except Exception as e:
            print(f"An error occurred while downloading {basename} 1.")
            print(e)
            return
        # print(f"Downloading data from {url}\nPlease wait ...")

        r = requests.get(url, stream=True)
        if r.status_code != 200:
            print(f"An error occurred while downloading {basename} 2.")
            return

        with open(file_path, 'wb') as fd:
            fd.write(r.content)
        print(f"Data downloaded to {file_path}")

    except Exception as e:
        print(f"An error occurred while downloading {basename} 3.")
        print(r.json()["error"]["message"])
        return


def simplify_road_type(org_type, road_types_dict):
    """ Maps OSM road type to integer types in road_types_dict. """

    if not isinstance(org_type, list):
        org_type = [org_type]

    # Check if all types are link type
    # if not remove all link types
    is_link = ['link' in t for t in org_type]
    if sum(is_link) == len(org_type):
        org_type = [t.split('_link')[0] for t in org_type]
    else:
        org_type = [t for t in org_type if 'link' not in t]

    # Remove all types not in dict
    org_type = [t if t in road_types_dict.keys() else 'unknown'
                for t in org_type]
    assert len(org_type) > 0

    tp = max(org_type, key=lambda x: road_types_dict[x])
    return road_types_dict[tp]


def load_roads(bbox, data_path, raster_to_match,
               network_type='drive', force_download=False,
               d_metric=np.inf):
    """Loads road network from OSM and burns in a multiband road
       raster."""

    # https://wiki.openstreetmap.org/wiki/Key:highway
    road_types_dict = {
        'motorway': 7,
        'trunk': 6,
        'primary': 5,
        'secondary': 4,
        'tertiary': 3,
        'unclassified': 2,
        'residential': 1,
        'living_street': 1,
        'unknown': 1
    }

    crs = raster_to_match.rio.crs.to_epsg()

    # Load the road graph
    G_path = data_path / 'road_network.graphml'
    edges_path = data_path / 'roads.gpkg'
    if not edges_path.is_file() or force_download:
        if not G_path.is_file():
            # Download roads from OSM
            G = ox.graph_from_polygon(bbox, network_type="drive")
            G = ox.project_graph(G, to_crs=crs)
            ox.save_graphml(G, G_path)
        else:
            G = ox.load_graphml(G_path)

        # Create vector geodataframe to burn in
        edges = ox.graph_to_gdfs(G, nodes=False)
        # Specify weight type, larger means more accessible
        edges['weight'] = edges.apply(
            lambda x: simplify_road_type(x.highway, road_types_dict),
            axis=1)
        edges = edges[['length', 'weight', 'geometry']]
        edges.to_file(data_path / 'roads.gpkg')
    else:
        edges = gpd.read_file(data_path / 'roads.gpkg')

    # Burn in roads into raster
    roads = make_geocube(vector_data=edges, measurements=['weight'],
                         like=raster_to_match, fill=0)['weight']

    # Create bands with nearest roads indices
    roads.values = roads.values.astype(int)
    road_idx = np.column_stack(np.where(roads.values > 0))
    # KDtree for dast neighbor lookup
    tree = KDTree(road_idx)
    # Explicitly create raster grid
    I, J = roads.values.shape
    grid_i, grid_j = np.meshgrid(range(I), range(J), indexing='ij')
    # Get coordinate pairs (i,j) to loop over
    coords = np.column_stack([grid_i.ravel(), grid_j.ravel()])
    # Find nearest road for every lattice point
    # p=inf is chebyshev distance (moore neighborhood)
    dist, idxs = tree.query(coords, p=d_metric)
    # Create bands
    dist = dist.reshape(roads.shape).astype(int)
    road_i = road_idx[:, 0][idxs].reshape(roads.shape)
    road_j = road_idx[:, 1][idxs].reshape(roads.shape)
    # Merge bands into single data array
    roads_ds = roads.to_dataset()
    roads_ds['dist'] = (['y', 'x'], dist)
    roads_ds['road_i'] = (['y', 'x'], road_i)
    roads_ds['road_j'] = (['y', 'x'], road_j)
    roads_ar = roads_ds.rename({'weight': 'roads'}).to_array(dim='bands')
    # Save individual rasters
    # roads_ar.rio.to_raster(data_path / 'roads.tif', dtype=np.int32)
    roads_ar.sel(bands='roads').rio.to_raster(
        data_path / 'roads.tif', dtype=np.int32)
    roads_ar.sel(bands='road_i').rio.to_raster(
        data_path / 'road_i.tif', dtype=np.int32)
    roads_ar.sel(bands='road_j').rio.to_raster(
        data_path / 'road_j.tif', dtype=np.int32)
    roads_ar.sel(bands='dist').rio.to_raster(
        data_path / 'road_dist.tif', dtype=np.int32)

    return roads_ar


def download_rasters(country, city, data_path, s3Client, buff=10):
    """Downloads rasters to local storage.

    Copernicus, Worldpop and Gisa are requested from gee as
    thumbnails. Landsat is pulled from S3 public storage.

    Parameters
    ----------
    country : str
        Country of city to download rasters for.
    city : str
        City to download rasters for.
    data_path : str
        path to input data, city and country data sets.
        Rasters will be stored in a subdirectory called cache.
    buff: int
        buffer in km arround the functional urban area that
        the region of interest.
    """

    data_path = Path(data_path)

    # Create cache path if missing
    cache_path = data_path / f'cache/{country}-{city}'
    cache_path.mkdir(parents=True, exist_ok=True)

    # Load functional urban areas (fua)
    cities_fua = gpd.read_file(
        data_path / 'output/cities/cities_fua.gpkg')
    # Get region of interest (bounding box)
    fua = cities_fua.loc[(cities_fua.country == country) &
                         (cities_fua.city == city)]
    assert len(fua) == 1

    # We need a landsat raster as a template to reproj onto
    # TODO: avoid this requirement, send to UTM directly.
    print('Downloading landsat data sets ...')
    landsat_years = list(range(2015, 2022))
    for year in landsat_years:
        print('Downloading: Data from ' + str(year))
        file_name = 'landsat-'+str(year)+'.tif'
        ls_path = cache_path / file_name
        if ls_path.exists():
            print(f'{ls_path} already exists, skipping.')
        else:
            landsat_from_s3(ls_path, country=country, city=city, year=year)
            print(f"Data downloaded to {ls_path}")
    print('Done.')

    # Other sources bbox must be larger to aleviate utm
    # empty bands problem
    bbox_s = get_roi(fua, buff=buff+5, buff_type='km').geometry[0]
    bbox_ee_s = ee.Geometry.Polygon(
        [t for t in zip(*bbox_s.exterior.coords.xy)])

    src_list = [  # 'Copernicus', 'WorldPop',
                'GISA',
                'Geomorpho90-slope', 'OSM_waterLayer',
                'WDPA']
    for src_id in src_list:
        print(f'Downloading {src_id} data set ...')
        download_src_to_local(src_id, bbox_ee_s, cache_path, country=country)
        print('Done.')

    print('Downloading road network ...')
    landsat = rxr.open_rasterio(cache_path / 'landsat-2021.tif')
    lbands = landsat.attrs['long_name']
    landsat = landsat.assign_coords(band=list(lbands))
    landsat.attrs.update(long_name='Landsat')

    road_files = ['roads.tif', 'road_i.tif',
                  'road_j.tif', 'road_dist.tif']
    road_exists = all([(cache_path / f).exists() for f in road_files])
    if road_exists:
        print('All road files already exist, skipping.')
    else:
        roads = load_roads(bbox_s, cache_path, landsat)
    print('Done.')

    # Making Urbanization Predictions
    print('Making GISA predictions ...')
    gisa_predicted_years = range(max(sources["GISA"]["temp_cov"])+1, 2022)
    for year in gisa_predicted_years:
        make_gisa_pred(year, cache_path)
        print("Done.")

    # Get UrbanLandCover
    print('Obtaining Historic Landover ...')

    landcover_exists = cache_path / 'historic_landcover.csv'
    if landcover_exists.exists():
        print('Historic Landover already exist, skipping.')
    else:
        historic_landcover = get_historic_landcover(bbox_s)
        historic_landcover.to_csv(cache_path / 'historic_landcover.csv')
        print("Done.")
    return cache_path


def landsat_from_s3(ls_path, country="Mexico", city="Mérida", year=2021):

    bucket = 'tec-expansion-urbana-p'

    s3_path = f'GEE/{country}-{city}/landsat-{year}.tif'

    url = f'http://{bucket}.s3.amazonaws.com/{s3_path}'

    landsat_data = requests.get(url).content

    with open(ls_path, 'wb') as handler:
        handler.write(landsat_data)


def download_src_to_local(src_id, bbox_ee, cache_path, country=None):
    src_dict = sources[src_id]

    names_dict = {
        # 'Copernicus': 'copernicus',
        # 'WorldPop': 'worldpop',
        'GISA': 'gisa',
        'Geomorpho90-slope': 'slope',
        'OSM_waterLayer': 'water',
        'WDPA': 'protected'
    }

    year_list = src_dict['temp_cov']
    for year in year_list:
        fname = f'{names_dict[src_id]}-{year}.tif'
        if len(year_list) == 1:
            fname = f'{names_dict[src_id]}.tif'
        fpath = cache_path / fname
        if fpath.exists():
            print(f'{fname} already exists, skipping.')
            continue
        src_img = load_src(src_id, year, bbox_ee, country=country).unmask(0)
        local_download(src_img, bbox_ee, cache_path, fname)


def lon_2_meter(lat, delta):
    """ Length of a longitude arc length in meters. """
    # From:
    # https://en.wikipedia.org/wiki/Longitude#Length_of_a_degree_of_longitude
    lat_rad = np.pi/180*abs(lat)

    a = 6378137.0
    b = 6356752.3142
    e2 = (a**2 - b**2)/a**2

    l = a*np.cos(lat_rad) / np.sqrt(1 - e2 * np.sin(lat_rad)**2)
    l *= np.pi/180

    return l*delta


def lat_2_meter(lat, delta):
    """ Length of a degree of latitude in meters. """
    # From:
    # https://en.wikipedia.org/wiki/Latitude#Meridian_distance_on_the_ellipsoid
    lat_rad = np.pi/180*abs(lat)

    a = 6378137.0
    b = 6356752.3142
    e2 = (a**2 - b**2)/a**2

    l = a*(1 - e2) / (1 - e2 * np.sin(lat_rad)**2)**(3/2)
    l *= np.pi/180

    return l*delta


def pop_2_density(raster, units='ha', save=False):

    # Load population raster
    if isinstance(raster, Path) or isinstance(raster, str):
        raster_path = Path(raster)
        pop_rxr = rxr.open_rasterio(raster_path)
    elif isinstance(raster, xr.DataArray):
        pop_rxr = raster
        save = False
    else:
        print('input must be either path or DataArray.')
        return

    area_grid = get_area_grid(pop_rxr, units)

    density_ar = pop_rxr.values / area_grid
    density_xr = pop_rxr.copy(data=density_ar)
    if save:
        fname = f'{raster_path.stem}-density-{units}-{raster_path.suffix}'
        density_xr.rio.to_raster(raster_path.parent / fname)

    return density_xr


def get_area_grid(raster_xr, units):

    c_factor = {'m': 1, 'km': 1/1e6, 'ha': 1/1e4}

    x_ar = raster_xr.coords['x'].values
    y_ar = raster_xr.coords['y'].values
    lon_grid, lat_grid = np.meshgrid(x_ar, y_ar)
    delta_x, delta_y = [abs(x) for x in raster_xr.rio.resolution()]

    area_grid = lat_2_meter(
        lat_grid, delta_y) * lon_2_meter(lat_grid, delta_x)
    area_grid *= c_factor[units]

    return area_grid


def load_dw(bbox=None, year=2021, clip=False, reproj=False):
    assert year in list(range(2016, 2022))

    startDate = f'{year}-01-01'
    endDate = f'{year}-12-31'

    if bbox is None:
        roi = Polygon([[-89.752197, 20.844071],
                       [-89.752197, 21.102812],
                       [-89.485474, 21.102812],
                       [-89.485474, 20.844071],
                       [-89.752197, 20.844071]])

    bbox_ee = ee.Geometry.Polygon(
        [t for t in zip(*bbox.exterior.coords.xy)])

    # Load ImageCollection
    dw = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')

    # Filter for year and roi
    dw = dw.filterDate(startDate, endDate).filterBounds(bbox_ee)

    # Select 'label'
    classification = dw.select('label')

    utm = load_utm(data_dir)

    warnings.simplefilter("ignore", UserWarning)
    max_epsg = utm.CRS.loc[utm.intersection(bbox).area.idxmax()]

    # adj_trans = 0
    # filter_epsg = max_epsg

    # get transform for corresponding image
    # proj = dw.filterMetadata('epsg', 'equals', filter_epsg)
    # proj = proj.first().projection().getInfo()

    to_list = dw.toList(classification.size())

    for img_id in range(classification.size().getInfo()):
        img = ee.Image(to_list.get(img_id))
        if img.projection().crs().getInfo() == max_epsg:
            break

    proj = img.projection().getInfo()

    crs = max_epsg
    transform = proj['transform']
    # transform[5] += adj_trans

    # Composite
    dwComposite = classification.reduce(ee.Reducer.mode())

    # Image Set
    image = dwComposite.set(
        {
         "year": year,
         "system:time_start": ee.Date(startDate).millis(),
         "nBands": 1,
         "system:date": ee.Date(startDate).format("YYYY"),
        })

    # Clip
    if clip:
        image = image.clip(bbox_ee)

    if reproj:
        image = image.reproject(crs=crs, crsTransform=transform)

    return image


def freqHistogram_dw(bbox=None, year=2021, clip=False, reproj=False):
    # Load Image
    image = load_dw(bbox, year, clip, reproj)

    # Pixel Counts
    pixelCountStats = image.reduceRegion(
        reducer=ee.Reducer.frequencyHistogram().unweighted(),
        geometry=bbox_to_ee(bbox),
        scale=10,
        maxPixels=1e10
    )

    pixelCounts = ee.Dictionary(pixelCountStats.get('label_mode'))

    data = pixelCounts.getInfo()
    data["year"] = year

    return data


def get_historic_landcover(bbox=None):

    src_dict = sources["DynamicWorld"]

    if bbox is None:
        roi = Polygon([[-89.752197, 20.844071],
                       [-89.752197, 21.102812],
                       [-89.485474, 21.102812],
                       [-89.485474, 20.844071],
                       [-89.752197, 20.844071]])

    year_list = src_dict['temp_cov']

    data_all = []

    for year in year_list:
        data_all.append(
            freqHistogram_dw(bbox=bbox, year=year, clip=True, reproj=True))

    data_all_df = pd.DataFrame(data_all)

    return data_all_df


def map_dw(Map, bbox_ee=None, year=2021, clip=False, reproj=False):
    # Load Image
    image = load_dw(bbox_ee, year, clip, reproj)

    # Add Layer
    return Map.addLayer(
        image, sources["DynamicWorld"]["vis_params"], 'Land Use Land Cover')


def bbox_to_ee(bbox):

    bbox_ee = ee.Geometry.Polygon(
        [t for t in zip(*bbox.exterior.coords.xy)])

    return bbox_ee


def make_gisa_pred(year, cache_path):
    # Create cache path if missing
    model_path = cache_path / 'urb_model.pkl'

    # Load Model
    if model_path.exists():
        print("Model already exists.")
        urb_model = pickle.load(open(model_path, 'rb'))
    else:
        print("Model does not exist. Training ...")
        urb_model = train_urb_model(cache_path)

    # Make Predictions

    # Path to Images
    path_to_images = str(cache_path)

    # Load Rasters
    landsat_file = path_to_images + '/landsat-' + str(year) + '.tif'
    gisa_file = path_to_images + '/gisa-2019.tif'
    landsat = rxr.open_rasterio(landsat_file)
    gisa = rxr.open_rasterio(gisa_file)

    # Raster to DataFrame
    landsat_shape = landsat.shape
    landsat_dataframe = pd.DataFrame(
        landsat.values.reshape(
            landsat_shape[0],
            landsat_shape[1]*landsat_shape[2]).T,
        columns=landsat.long_name)

    # DropNA
    landsat_dataframe_cleaned = landsat_dataframe.dropna()

    # Urbanization Prediction
    gisa_predicted = urb_model.predict(landsat_dataframe_cleaned.values)

    # Add prediction to cleaned DataFrame
    landsat_dataframe_cleaned["Gisa"] = gisa_predicted

    # Joined DataFrames on index
    lansat_dataframe_predicted = landsat_dataframe.join(
        landsat_dataframe_cleaned["Gisa"])

    # Raster Creation
    gisa_predicted = landsat.sel(band=1).copy(
        data=lansat_dataframe_predicted["Gisa"].fillna(0).values.reshape(
            landsat_shape[1], landsat_shape[2]))
    gisa_predicted.attrs["long_name"] = "Gisa"

    # Raster Reprojection on GISA
    gisa_predicted_reprojected = gisa_predicted.rio.reproject_match(gisa)
    gisa_predicted_reprojected = gisa_predicted_reprojected.assign_coords({
        "x": gisa.x,
        "y": gisa.y,
    })

    # Raster Corrections:
    # - NA Values
    gisa_predicted_reprojected = gisa_predicted_reprojected.where(
        (gisa_predicted_reprojected != np.finfo(np.float64).max)
        & (gisa_predicted_reprojected > 0))
    gisa_predicted_reprojected = gisa_predicted_reprojected.fillna(0)
    gisa_predicted_reprojected = gisa.copy(
        data=gisa_predicted_reprojected.values.reshape(
            1,
            gisa_predicted_reprojected.shape[0],
            gisa_predicted_reprojected.shape[1]))

    # - Merge: Gisa Predicted + Gisa Last Year Available (2019)
    gisa_predicted_reprojected = gisa + gisa_predicted_reprojected
    gisa_predicted_reprojected = gisa_predicted_reprojected.where(
        gisa_predicted_reprojected < 1)
    gisa_predicted_reprojected = gisa_predicted_reprojected.fillna(1)

    raster_path = path_to_images + '/gisa-' + str(year) + '.tif'
    gisa_predicted_reprojected.rio.to_raster(raster_path)
    print(f"Data downloaded to {raster_path}")

    # gisa_predicted_reprojected.plot()
    # plt.show()


def train_urb_model(cache_path):
    # Path
    path_to_images = str(cache_path)
    # Check all gisa Files
    gisa_files = glob.glob(path_to_images + '/gisa-*.tif')
    gisa_files.sort()

    # Obtain available years
    years = re.findall(r'\d{4}', '|'.join(gisa_files))
    years.sort()

    non_reprojected_years = []
    reprojected_years = []

    # Iterate over years
    for year in years:
        # Check if gisa and landsat files exist for year
        gisa_file = path_to_images + '/gisa-' + str(year) + '.tif'
        landsat_file = path_to_images + '/landsat-' + str(year) + '.tif'

        if os.path.exists(gisa_file) and os.path.exists(landsat_file):
            reprojected_years.append(year)
        else:
            non_reprojected_years.append(year)

    merge_dataframe_full = pd.DataFrame()

    for year in reprojected_years:
        # Name of Files
        gisa_file = path_to_images + '/gisa-' + str(year) + '.tif'
        landsat_file = path_to_images + '/landsat-' + str(year) + '.tif'

        # Open Files
        gisa = rxr.open_rasterio(gisa_file)
        landsat = rxr.open_rasterio(landsat_file)

        # Reproject GISA on Landsat
        gisa_repr_match = gisa.rio.reproject_match(landsat)

        # Landsat to Dataframe
        landsat_shape = landsat.shape
        merged_dataframe = pd.DataFrame(
            landsat.values.reshape(
                landsat_shape[0],
                landsat_shape[1]*landsat_shape[2]).T,
            columns=landsat.long_name)

        # Add Landsat Layer
        merged_dataframe["Gisa"] = gisa_repr_match.values.flatten()

        # Add Year
        merged_dataframe["Year"] = year

        # Drop NA values
        merged_dataframe.dropna(inplace=True)

        # Concat DataFrames
        if merge_dataframe_full.empty:
            merge_dataframe_full = merged_dataframe.copy()
        else:
            merge_dataframe_full = pd.concat(
                [merge_dataframe_full, merged_dataframe])

    merge_dataframe_sample = merge_dataframe_full.sample(frac=0.001)

    input_columns = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"]
    output_columns = ["Gisa"]

    # Define explanatory variables
    X = merge_dataframe_sample[input_columns].values
    # Y as all the other bands
    y = merge_dataframe_sample[output_columns].values.ravel()

    X_train, X_test, y_train, y_test = train_test_split(
        X,  y, test_size=0.20, random_state=0)

    # Random Forest Training
    forest_model = RandomForestClassifier(criterion='gini', n_estimators=10)

    depth_range = np.arange(2, 11)
    leaf_range = np.arange(1, 31, 9)

    forest_param_grid = [{
        'max_depth': depth_range,
        'min_samples_leaf': leaf_range
    }]

    forest_grid = GridSearchCV(
        forest_model, forest_param_grid,
        cv=10, scoring="f1").fit(X_train, y_train)

    # Saving Model
    model_path = path_to_images + '/urb_model.pkl'
    print(f"Model saved to {model_path}")
    pickle.dump(forest_grid, open(model_path, 'wb'))

    return forest_grid
