import glob
import numpy as np
import pandas as pd
import rioxarray as rxr
import data_sources as dts
from geocube.api.core import make_geocube
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import label
from scipy.ndimage import labeled_comprehension


def load_raster_dict(prefix, raster_path,
                     start_year=1985, end_year=2020):
    rdict = {}
    for year in range(start_year, end_year+1):
        raster = raster_path / f'{prefix}-{year}.tif'
        if not raster.exists():
            continue
        da = rxr.open_rasterio(raster)
        rdict[str(year)] = da
    return rdict


def get_df_urban_growth(raster_path, start_year=1990, end_year=2021,
                        gdf=None):

    # Load urban rasters and area grid
    gisa_ds = load_raster_dict('gisa', raster_path,
                               start_year, end_year)
    area_grid = dts.get_area_grid(
        gisa_ds[str(start_year)], units='km')

    if gdf is not None:
        gdf = gdf.copy()
        # Create mask by burning polygons
        # Create column of ones to take
        gdf['burin'] = 1
        poly_mask = make_geocube(
            vector_data=gdf,
            measurements=['burnin'],
            like=gisa_ds[str(start_year)],
            fill=0
        )['burnin'].values

    # For all urban rasters find urbanized area
    results = []
    for year, gisa_da in gisa_ds.items():
        mask = gisa_da[0].values > 0
        impervious_area = area_grid[mask].sum()

        mask_filled = binary_fill_holes(mask)
        filled_area = area_grid[mask_filled].sum()

        npixels = mask.sum()
        total_pixels = np.prod(area_grid.shape)

        labeled, nlabels = label(mask_filled)
        lbls = np.arange(1, nlabels+1)
        label_largest = labeled_comprehension(
            labeled, labeled, lbls, np.sum, float, 0).argmax()+1
        mask_largest = (labeled == label_largest)
        area_largest = area_grid[mask_largest].sum()

        res_dict = {'year': year,
                    'impervious_area (km)': impervious_area,
                    'filled_area (km)': filled_area,
                    'area_largest_cluster': area_largest,
                    'urban_pixels': npixels,
                    'percent_urb': npixels/total_pixels}

        if gdf is not None:
            area_poly = area_grid[np.logical_and(poly_mask, mask)].sum()
            res_dict['area_polygons'] = area_poly

        results.append(res_dict)

    return pd.DataFrame(results).set_index('year')


def get_df_pop_growth(raster_path,
                      start_year=2000, end_year=2020,
                      gdf=None, trshld=2000):

    # Load urban rasters and area grid
    wp_ds = load_raster_dict('worldpop', raster_path,
                             start_year, end_year)

    if gdf is not None:
        gdf = gdf.copy()
        gdf['burin'] = 1

    results = []
    for year, wp_da in wp_ds.items():
        area_grid = dts.get_area_grid(wp_da, units='km')
        wp_ar = wp_da.values[0]

        density = wp_ar/area_grid
        mask = density > trshld

        total_pop = wp_ar.sum()
        urban_pop = wp_ar[mask].sum()
        urban_area = area_grid[mask].sum()
        res_dict = {'year': year, 'total_pop': total_pop,
                    'urban_pop': urban_pop,
                    'pop_area': urban_area}

        if gdf is not None:
            poly_mask = make_geocube(
                vector_data=gdf,
                measurements=['burnin'],
                like=wp_da,
                fill=0
            )['burnin'].values.astype(bool)

            poly_pop = wp_ar[poly_mask].sum()
            res_dict['poly_pop'] = poly_pop

        results.append(res_dict)

    return pd.DataFrame(results).set_index('year')


def get_sleuth_dataframe(cache_path):
    # Search for Sleuth files
    sleuth_slow = glob.glob(str(cache_path / 'sleuth_slow_*.tif'))
    # sleuth_usual = glob.glob(
    #     str(cache_path / 'sleuth_usual_*.tif'))
    # sleuth_fast = glob.glob(
    #     str(cache_path / 'sleuth_fast_*.tif'))

    # Obtain available years
    # It considers all files have same range
    years = []
    for file in sleuth_slow:
        year = file.split('_')[-1].split('.')[0]
        years.append(year)
    years = list(set(years))
    years.sort()

    # Iterate over each mode and year
    modes = ("slow", "usual", "fast")
    data = []
    for mode in modes:
        for year in years:
            # Sleuth File
            sleuth_file = str(cache_path) + f'/sleuth_{mode}_{year}.tif'
            sleuth_file

            # Open Files
            sleuth_raster = rxr.open_rasterio(sleuth_file)

            # Masking Raster
            sleuth_bin = sleuth_raster.where(sleuth_raster < 1, 1)

            # Frequency Count
            unique, counts = np.unique(
                sleuth_bin.data.flatten(), return_counts=True)

            # Data
            sleuth_data = {}

            sleuth_data['mode'] = mode
            sleuth_data['year'] = year
            for idx in range(len(unique)):
                sleuth_data[unique[idx]] = counts[idx]

            # Save Data
            data.append(sleuth_data)

    # Save DataFrame
    # data_df.to_csv(cache_path / 'sleuth_projections.csv', index=False)

    # Create DataFrame
    return pd.DataFrame(data)
