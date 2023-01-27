from dash_extensions.enrich import Output, Input, State, dcc, html, callback
from dash.exceptions import PreventUpdate
import plots as pts
import json
import base64

def b64_image(image_filename):
  # Funcion para leer imagenes
  with open(image_filename, 'rb') as f:
    image = f.read()
  return 'data:image/png;base64,' + base64.b64encode(image).decode('utf-8')

HEADER_STYLE = {
  'text-align': 'center',
  'color': '#404756',
  'margin-bottom': '2rem',
  'font-size': '30px',
  'width': '67%',
}

HEADER_STYLE_TEXT = {
  'text-align': 'center',
  'color': '#404756',
  'margin-bottom': '2rem',
  'font-size': '30px',
  'width': '100%',
  'margin-left': '33%'
}

line = html.Div(html.Hr(), className='content-style')

header = html.Div([
    line,
    html.Div(id='header', style=HEADER_STYLE)
])

@callback(
  Output('store_data', 'data'),
  Output('header', 'children'),
  Output('content', 'children'),
  Input('submit-button', 'n_clicks'),
  State('cou-dro', 'value'),
  State('cit-dro', 'value'),
  prevent_initial_call=True
)
def load_data(n_clicks, country, city):
  '''
  Callback to load the whole content of the dashboard from the
  menu (pasado, presente and futuro).

  State:
  (A state would save the colected data but it won't trigger anything)
    - value (cou-dro): contry value.
    - value (cit-dro): city value.

  Input:
    - n_clicks: a click triggers the callback.

  Output:
    - data (store-data): storage that save all the graphs in a dictionary.
    - children (header): a list containing the city and country in html
      format.
    - children (content): this is use just to 'erase' the content when you
      want to visualize other city/country.
  '''

  if n_clicks > 0:

    dcc.Store(id='store_data', storage_type='session', clear_data=True)

    children = [html.H1('{0}, {1}'.format(country, city),
                        style=HEADER_STYLE_TEXT)]

    pts.download_data(country, city)

    figs = pts.pop_past_graphs(city, country)
    cards_pop_urb = pts.pop_past_stats(city, country)
    density_txt = pts.density_landscan(country, city)
    urbanization_txt = pts.urbanization_txt(country, city)

    (area, pre_bar_land,
      card_land_txt, 
      card_land_num) = pts.land_graph(country, city)

    pre_gisa = pts.gisa_pre_map()
    pre_map_pop = pts.present_map('Landscan', country, city)
    pre_map_land = pts.present_map('DynamicWorld', country, city)

    gisa_slider = pts.gisa_slider(country, city)
    landsat_gif = pts.landsat_animation(city, country)

    sleuth_fast_map = pts.sleuth_map(
      country, 
      city, 
      'sleuth_fast_2040.tif', 
      'Crecimiento Rápido 2040'
    )
    sleuth_usual_map = pts.sleuth_map(
      country,
      city, 
      'sleuth_usual_2040.tif',
      'Crecimiento Inercial 2040'
    )
    sleuth_slow_map = pts.sleuth_map(
      country, 
      city, 
      'sleuth_slow_2040.tif',
      'Crecimiento Lento 2040'
    )
    sleuth_graph = pts.sleuth_graph()

    json_figs = json.dumps(
      {
        'figs': figs,
        'cards_pop_urb': cards_pop_urb,
        'card_land_txt': card_land_txt,
        'card_land_num': card_land_num,
        'density_txt': density_txt,
        'urbanization_txt': urbanization_txt,
        'area': area.to_json(),
        'pre_bar_land': pre_bar_land.to_json(),
        'pre_gisa': pre_gisa.to_json(),
        'pre_map_pop': pre_map_pop.to_json(),
        'pre_map_land': pre_map_land.to_json(),
        'gisa_slider': gisa_slider.to_json(),
        'landsat_gif': landsat_gif,
        'sleuth_fast_map': sleuth_fast_map.to_json(),
        'sleuth_usual_map': sleuth_usual_map.to_json(),
        'sleuth_slow_map': sleuth_slow_map.to_json(),
        'sleuth_graph': sleuth_graph.to_json(),
      }
    )
    return json_figs, children, None
  else:
    return PreventUpdate
