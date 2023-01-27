""" Process GHSL Urban centers data of cities and population into a
shapefile."""

import geopandas as gpd


def main():
    data_path = '../data/'

    ifile_path_1 = (data_path +
                    'input/GHS_STAT_UCDB2015MT_GLOBE_R2019A/'
                    'GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_2.gpkg')

    ifile_path_2 = (data_path +
                    'input/GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0/'
                    'GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0.gpkg')

    ofile_path_1 = (data_path + 'output/cities/cities_uc.gpkg')
    ofile_path_2 = (data_path + 'output/cities/cities_fua.gpkg')

    ghsl_uc = gpd.read_file(ifile_path_1)
    ghsl_fua = gpd.read_file(ifile_path_2)
    ghsl_fua = ghsl_fua.to_crs(ghsl_uc.crs)

    # Add UC to FUA data frame
    ghsl_fua['UC'] = ghsl_fua.apply(
        lambda x:
        ghsl_uc.iloc[int(x.UC_IDs.split(';')[0]) - 1].geometry,
        axis=1)
    ghsl_fua['GRGN_L1'] = ghsl_fua.apply(
        lambda x:
        ghsl_uc.iloc[int(x.UC_IDs.split(';')[0]) - 1].GRGN_L1,
        axis=1)
    ghsl_fua['GRGN_L2'] = ghsl_fua.apply(
        lambda x:
        ghsl_uc.iloc[int(x.UC_IDs.split(';')[0]) - 1].GRGN_L2,
        axis=1)

    # Filter huge set of columns of urban centers
    columns = ['CTR_MN_NM', 'GRGN_L1',
               'GRGN_L2', 'UC_NM_MN',
               'P15', 'geometry']
    ghsl_uc = ghsl_uc[columns]

    # Filter by city size
    ghsl_uc = ghsl_uc[ghsl_uc.P15 > 100000]
    ghsl_fua = ghsl_fua[ghsl_fua.UC_p_2015 > 100000]

    # Filter by region
    region = 'Latin America and the Caribbean'
    ghsl_uc = ghsl_uc[ghsl_uc.GRGN_L1 == region]
    ghsl_uc.drop(columns=['GRGN_L1'], axis=1, inplace=True)
    ghsl_fua = ghsl_fua[ghsl_fua.GRGN_L1 == region]
    ghsl_fua.drop(columns=['GRGN_L1'], axis=1, inplace=True)

    # Rename columns
    col_dict = {'CTR_MN_NM': 'country', 'GRGN_L2': 'region',
                'UC_NM_MN': 'city', 'P15': 'population'}
    ghsl_uc.rename(columns=col_dict, inplace=True)
    col_dict = {'Cntry_name': 'country', 'GRGN_L2': 'region',
                'eFUA_name': 'city'}
    ghsl_fua.rename(columns=col_dict, inplace=True)

    ghsl_fua.drop(columns=['UC']).to_file(ofile_path_2)
    ghsl_fua.drop(columns=['geometry']).rename(columns={'UC': 'geometry'}).set_geometry('geometry').to_file(ofile_path_1)


if __name__ == '__main__':
    main()
