from dash_extensions.enrich import Output, Input, dcc, html, callback
import dash_bootstrap_components as dbc
import geopandas as gpd
import base64
from pathlib import Path

# Sidebar stylesheet CSS
logo = './assets/BID.png'
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "background-color": '#404756'
}


def b64_image(image_filename):
  # Funcion para leer imagenes
  with open(image_filename, 'rb') as f:
    image = f.read()
  return 'data:image/png;base64,' + base64.b64encode(image).decode('utf-8')
data_path = Path('./data')
cities_fua = gpd.read_file(data_path / 'output' / 'cities' / 'cities_fua.gpkg')
cities_uc = gpd.read_file(data_path / 'output' / 'cities' / 'cities_uc.gpkg')

@callback(
  Output('cit-dro', 'options'),
  Output('cit-dro', 'value'),
  Input('cou-dro', 'value'),
  )

def filter_city(cou):
  '''
  Callback to display only the cities that belog to the country that
  was previously selected.

  Input:
    - cou: contry value.

  Output:
    - option (list): cities list.
    - value (string): a city to display in the box.
    '''  
  df_cou = cities_fua[cities_fua.country == cou]
  df_cou = df_cou.city.unique()
  df_cou = list(df_cou)
  df_cou.sort()
  options = [{'label': city, 'value': city} for city in df_cou]
  return options, options[0]['value']

# Dropdown de pais y cuidad
country_dropdown = dcc.Dropdown(
  options=[
    {
      'label': country,
      'value': country
    }
    for country in cities_fua.country.unique()
  ],
  value='Argentina',
  id='cou-dro', className='dropdown-style'
)

city_dropdown = dcc.Dropdown(
  options=[
    {
      'label': city,
      'value': city
    }
    for city in cities_fua.city.unique()
  ],
  value='Bahía Blanca',
  id='cit-dro', 
  className='dropdown-style'
)

sidebar = html.Div(
  [
    html.Img(
      src=b64_image(logo),
      style= {
        'width': '75%',
        'margin-left': '12.5%',
        'margin-right': '12.5%',
        'margin-top': '5%',
      }
    ),
    html.Hr(),
    dbc.Col(
      [
        dbc.Label("Filtrar por país"),
        country_dropdown
      ], 
      className='pais-style'
    ),
    dbc.Row(
      [
        dbc.Label("Filtrar por cuidad"),
        city_dropdown
      ], 
      className='pais-style'
    ),
    html.Button(
      'Consultar', 
      id='submit-button',
      n_clicks=0,
      className='button-style-sub',
    ),
  ], 
  className='sidebar-style'
)

