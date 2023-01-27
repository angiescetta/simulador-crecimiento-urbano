FROM jupyter/scipy-notebook

RUN mamba install -y -c conda-forge boto3 geemap dash geocube geopandas numpy osmnx pandas plotly pyproj rasterio requests rioxarray scipy Shapely Unidecode xarray contextily scikit-learn pyyaml r-base

RUN pip install dash-extensions dash_extensions dash_bootstrap_components dash_gif_component rpy2==3.5.1 dash_unload_component

RUN mkdir app
WORKDIR app
COPY . .

EXPOSE 8050

CMD [ "python", "-u", "app.py" ]
