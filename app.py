from dash_extensions.enrich import Output, Input, State, DashProxy, MultiplexerTransform, dcc, html
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import dash_unload_component as duc
import plotly.graph_objects as go
import dash_gif_component as gif
from pathlib import Path
import geopandas as gpd
import base64
import plotly
import json
import sys
import threading as th
import os
import signal
import subprocess
import warnings
sys.path.append('./src')
import plots as pts

from components import sidebar, header, menu

warnings.filterwarnings('ignore')


def open_browser():
    subprocess.run("python -mwebbrowser http://localhost:8050/", shell=True)


def b64_image(image_filename):
    # Funcion para leer imagenes
    with open(image_filename, 'rb') as f:
        image = f.read()
    return 'data:image/png;base64,' + base64.b64encode(image).decode('utf-8')


def indicator_generator(amount):
    # Generador del indicador de cambio en las tarjetas
    # de poblacion y uso de suelo
    if amount == 'No aplica.':
        return None
    else:
        amount = round(amount*100, 2)

        fig = go.Figure(
            go.Indicator(
                mode="delta",
                value=amount,
                delta={"reference": 100, "relative": True},
            )
        )
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0),
                          paper_bgcolor="rgba(0,0,0,0)")

        return dcc.Graph(figure=fig, style={"height": 50, "width": 100})

# Estilos basicos de CSS
# Color palette for IDB
color_palette = ['#292D73', '#2F5C97', '#326D8D', '#707788', '#C5D0DF']

# Main content stylesheet CSS
CONTENT_STYLE = {
    "margin-left": "auto",
    "margin-right": "auto",
    "top": 0,
    "left": 0,
    "bottom": '50rem',
    "height": '100%',
    "padding": "2rem 4rem",
    'background-color': '#FBFBFB'
}

SUBTITLE_YEAR = {
    'margin-top': '-2rem',
    'margin-bottom': '2rem',
    'width': '67%',
    'margin-left': '14%',
}
content = html.Div([
    header,
    html.Div(
        id='content',
        className='content-style'),
], style=CONTENT_STYLE)

# Creacion de la app y uso de Darkly theme
app = DashProxy(
    external_stylesheets=[dbc.themes.DARKLY],
    transforms=[MultiplexerTransform()]
)

server = app.server

app.layout = html.Div([
    dcc.Location(id="times"),
    dcc.Store(id='store_data', storage_type='session'),
    duc.Unload(id='page-listener'),
    html.Div(id='listener'),
    menu,
    sidebar,
    content])


@app.callback(Output('listener', 'children'), [Input('page-listener', 'close')])
def detect_close(close):
    if close is None:
        print("Current window is closed? ", close)
        return None
    else:
        print("Current window is closed? ", close)
        os.kill(os.getpid(), signal.SIGTERM)
    return None

@app.callback(Output('content', 'children'),
              State('store_data', 'data'),
              Input('times', 'pathname'),
              prevent_initial_call=True)
def render_content(json_data, pathname):
    '''
    Callback to display the whole content of the dashboard from the
    menu (pasado, presente and futuro).

    State:
    (A state would save the colected data but it won't trigger anything)
        - data (store-data): storage that saved all the graphs in a dictionary.

    Input:
        - pathname (times): in the app layout we used the
          dcc.Location(id="times"), which creates a pathname of each section
          of our menu and here we access it
          to display specific visuals to a specific path.

    Output:
        - children (store-data): a list of dash containers with the different
          visualizations depending of the pathname

    '''

    if json_data is None:
        raise PreventUpdate

    data = json.loads(json_data)

    figs_json = data['figs']
    figs = [plotly.io.from_json(i) for i in figs_json]

    cards_pop_urb = data['cards_pop_urb']
    card_land_txt = data['card_land_txt']
    card_land_num = data['card_land_num']
    density_txt = data['density_txt']
    urbanization_txt = data['urbanization_txt']

    area = plotly.io.from_json(data['area'])

    pre_gisa = plotly.io.from_json(data['pre_gisa'])
    pre_map_pop = plotly.io.from_json(data['pre_map_pop'])
    pre_bar_land = plotly.io.from_json(data['pre_bar_land'])
    pre_map_land = plotly.io.from_json(data['pre_map_land'])

    gisa_slider = plotly.io.from_json(data['gisa_slider'])
    landsat_gif = data['landsat_gif']

    sleuth_fast_map = plotly.io.from_json(data['sleuth_fast_map'])
    sleuth_usual_map = plotly.io.from_json(data['sleuth_usual_map'])
    sleuth_slow_map = plotly.io.from_json(data['sleuth_slow_map'])

    sleuth_graph = plotly.io.from_json(data['sleuth_graph'])

    if pathname == '/pas/urb':
        pas_urb = [
            dbc.Row([
                html.Div(
                    [html.H3('(2000-2019)', style={'text-align': 'center'})],
                    style=SUBTITLE_YEAR)
            ]),
            dbc.Row([
                dbc.Col(
                    dcc.Graph(figure=gisa_slider),
                    style={
                        "margin-left": "10rem",
                        "margin-bottom": "1rem",
                        "display": "flex",
                        "flex-direction": "row",
                        "justify-content": "center"
                    }
                )
            ]),
            dbc.Row([
                dbc.Col(
                    gif.GifPlayer(gif=b64_image(landsat_gif), still=None),
                    style={
                        "margin-left": "10rem",
                        "display": "flex",
                        "flex-direction": "row",
                        "justify-content": "center"
                    }
                )
            ]),
            dbc.Row([
                html.Div([
                    html.H5(
                        ['El Gif de Landsat mostrado presenta poca calidad debido a la nubes.',
                         html.Br(),
                         'Para descargar un Landsat Gif con menos nubes, click en descargar.',
                         html.Br()],
                        style={'text-align': 'center'}
                    ),
                    html.H5(
                        ['ADVERTENCIA: EL GIF PUEDE TARDAR VARIOS MINUTOS EN GENERARSE.'],
                        style={'text-align': 'center', 'color': 'red'}
                    )
                ])
            ],
                    style={'margin-bottom': '2rem',
                           'margin-top': '2rem',
                           'text-align': 'center',
                           'align-content': 'center',
                           'display': 'inline'
                           }
            ),
            dbc.Row([
                html.Button(
                    'Descargar',
                    id='gif-button',
                    n_clicks=0,
                    className='gif-button-style'),
                dcc.Download(id='download-GIF')
            ]),
            dbc.Row([
                dbc.Col(
                    html.P('Fuente:'),
                    className='src-txt-style',
                    width=10
                ),
                dbc.Col(
                    html.A(
                        'Landsat,',
                        href='https://developers.google.com/earth-engine/datasets/catalog/landsat',
                        target="_blank", className='src-link-style'
                    ),
                    width=1
                ),
                dbc.Col(
                    html.A(
                        'GISA',
                        href='https://samapriya.github.io/awesome-gee-community-datasets/projects/gisa/',
                        target="_blank", className='src-link-style'
                    ),
                    width=1
                )
            ], className='src-style-paur')
        ]
        return pas_urb
    elif pathname == '/pas/pop':
        pas_pop = [
            dbc.Row([
                html.Div(
                    [html.H3('(2000-2019)', style={'text-align': 'center'})],
                    style=SUBTITLE_YEAR)
            ]),
            dbc.Row([
                dbc.Col(
                    dbc.Card(
                        [dbc.CardBody([
                            html.H5("Población", className="card-title"),
                            indicator_generator(cards_pop_urb['diff_pop']),
                            html.P(str(cards_pop_urb['pop_txt']),
                                   className="card-text")])],
                        color="#005073", inverse=True),
                    width=4
                ),
                dbc.Col(
                    dbc.Card(
                        [dbc.CardBody(
                            [html.H5("Urbanización", className="card-title"),
                             indicator_generator(cards_pop_urb['diff_urb']),
                             html.P(str(cards_pop_urb['urb_txt']),
                                    className="card-text")])],
                        color="dark", inverse=True),
                    width=4
                ),
                dbc.Col(
                    dbc.Card(
                        [dbc.CardBody(
                            [html.H5("Densidad", className="card-title"),
                             indicator_generator(cards_pop_urb['diff_den']),
                             html.P(str(cards_pop_urb['den_txt']),
                                    className="card-text")])],
                        color="secondary", inverse=True),
                    width=4
                )
            ]),
            html.Hr(),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=figs[0]), width=4),
                dbc.Col(dcc.Graph(figure=figs[1]), width=4),
                dbc.Col(dcc.Graph(figure=figs[2]), width=4)
            ]),
            dbc.Row([
                dbc.Col(html.P('Fuente:'), className='src-txt-style'),
                dbc.Col(html.A(
                    'Landscan,',
                    href='https://landscan.ornl.gov',
                    target="_blank", className='src-link-style')),
                dbc.Col(
                    html.A('GISA',
                           href='https://samapriya.github.io/awesome-gee-community-datasets/projects/gisa/',
                           target="_blank", className='src-link-style')),
            ], className='src-style')
        ]
        return pas_pop
    elif pathname == '/pas/lan':
        pas_lan = [
            dbc.Row([
                html.Div(
                    [html.H3('(2016-2021)', style={'text-align': 'center'})],
                    style=SUBTITLE_YEAR)
            ]),
            dbc.Row([
                dbc.Col(dbc.Card(
                    [dbc.CardBody([
                        html.H5("Urbanización", className="card-title"),
                        indicator_generator(card_land_num['Urbanización']),
                        html.P(card_land_txt['Urbanización'],
                               className="card-text")
                    ])],
                    color="#801a00", inverse=True),
                        width=4
                        ),
                dbc.Col(dbc.Card(
                    [dbc.CardBody([html.H5("Cultivos",
                                           className="card-title"),
                                   indicator_generator(card_land_num['Cultivos']),
                                   html.P(card_land_txt['Cultivos'],
                                          className="card-text")
                                   ])],
                    color='#264d00', inverse=True),
                        width=4
                        ),
                dbc.Col(dbc.Card(
                    [dbc.CardBody([html.H5("Árboles",
                                           className="card-title"),
                                   indicator_generator(card_land_num['Árboles']),
                                   html.P(card_land_txt['Árboles'],
                                          className="card-text")
                                   ])],
                    color="dark", inverse=True),
                        width=4
                        ),
            ], style={'margin-bottom': '1rem'}),
            dbc.Row([dbc.Col(dbc.Card(
                [dbc.CardBody([html.H5("Vegetación inundada",
                                       className="card-title"),
                               indicator_generator(card_land_num['Vegetación inundada']),
                               html.P(card_land_txt['Vegetación inundada'],
                                      className="card-text")
                               ])],
                color="dark", inverse=True),
                             width=4
                             ),
                     dbc.Col(dbc.Card(
                         [dbc.CardBody([html.H5("Césped/Pasto",
                                                className="card-title"),
                                        indicator_generator(card_land_num['Césped/Pasto']),
                                        html.P(card_land_txt['Césped/Pasto'],
                                               className="card-text")
                                        ])],
                         color="dark", inverse=True),
                             width=4
                             ),
                     dbc.Col(dbc.Card(
                         [dbc.CardBody([html.H5("Arbusto y matorral",
                                                className="card-title"),
                                        indicator_generator(card_land_num['Arbusto y matorral']),
                                        html.P(card_land_txt['Arbusto y matorral'],
                                               className="card-text")
                                        ])],
                         color="dark", inverse=True),
                             width=4
                             ),
                     ], style={'margin-bottom': '1rem'}),
            dbc.Row([
                dbc.Col(dbc.Card(
                    [dbc.CardBody([html.H5("Agua",
                                           className="card-title"),
                                   indicator_generator(card_land_num['Agua']),
                                   html.P(card_land_txt['Agua'],
                                          className="card-text")
                                   ])],
                    color="dark", inverse=True),
                        width=4
                        ),
                dbc.Col(dbc.Card(
                    [dbc.CardBody([html.H5("Descubierto",
                                           className="card-title"),
                                   indicator_generator(card_land_num['Descubierto']),
                                   html.P(card_land_txt['Descubierto'],
                                          className="card-text")
                                   ])],
                    color="dark", inverse=True),
                        width=4
                        ),
                dbc.Col(dbc.Card(
                    [dbc.CardBody([html.H5("Nieve y hielo",
                                           className="card-title"),
                                   indicator_generator(card_land_num['Nieve y hielo']),
                                   html.P(card_land_txt['Nieve y hielo'],
                                          className="card-text")
                                   ])],
                    color="dark", inverse=True),
                        width=4
                        ),
            ], style={'margin-bottom': '2rem'}),
            dbc.Row([dbc.Col(dcc.Graph(figure=area),
                             )]),
            dbc.Row([dbc.Col(html.P('Fuente:'), className='src-txt-style'),
                     dbc.Col(html.A('Dynamic World V1',
                                    href='https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1#description',
                                    target="_blank", className='src-link-style'))
                     ],
                    className='src-style')
        ]
        return pas_lan
    elif pathname == '/pre/urb':
        pre_urb = [
            dbc.Row([html.Div(
                [html.H3('(2021)', style={'text-align': 'center'})],
                style=SUBTITLE_YEAR)
                     ]),
            dbc.Row(
                [html.Div(
                    html.H5(urbanization_txt, style={'text-align': 'center'})
                )], style={
                    'margin-bottom': '2rem',
                    'text-align': 'center',
                    'align-content': 'center'
                }),
            dbc.Row([dbc.Col(dcc.Graph(figure=pre_gisa),
                             style={
                                 "margin-left": "10rem",
                                 "display": "flex",
                                 "justify-content": "center"
                             })
                     ],
                    ),
            dbc.Row([dbc.Col(html.P('Fuente:'), className='src-txt-style'),
                     dbc.Col(html.A(
                         'GISA',
                         href='https://samapriya.github.io/awesome-gee-community-datasets/projects/gisa/',
                         target="_blank", className='src-link-style'
                     ))], className='src-style-urb')
        ]
        return pre_urb
    elif pathname == '/pre/pop':
        pre_pop = [
            dbc.Row([
                html.Div(
                    [html.H3('(2020)', style={'text-align': 'center'})],
                    style=SUBTITLE_YEAR
                )]),
            dbc.Row([
                html.Div(html.H5(density_txt, style={'text-align': 'center'}))
            ], style={
                'margin-bottom': '2rem',
                'text-align': 'center',
                'align-content': 'center'
            }),
            dbc.Row([dbc.Col(dcc.Graph(figure=pre_map_pop))]),
            dbc.Row([dbc.Col(html.P('Fuente:'), className='src-txt-style'),
                     dbc.Col(
                         html.A(
                             'Landscan',
                             href='https://landscan.ornl.gov',
                             target="_blank", className='src-link-style')
                     )], className='src-style-pp')
        ]
        return pre_pop
    elif pathname == '/pre/lan':
        pas_lan = [
            dbc.Row([html.Div([
                html.H3('(2021)', style={'text-align': 'center'}
                        )], style=SUBTITLE_YEAR)
                     ]),
            dbc.Row([dbc.Col(dcc.Graph(figure=pre_bar_land))],
                    style={'margin-bottom': '1rem'}),
            dbc.Row([dbc.Col(dcc.Graph(figure=pre_map_land))]),
            dbc.Row([dbc.Col(html.P('Fuente:'), className='src-txt-style'),
                     dbc.Col(html.A(
                         'Dynamic World V1',
                         href='https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1#description',
                         target="_blank", className='src-link-style'
                     ))],
                    className='src-style-pland')
        ]
        return pas_lan
    elif pathname == '/fut':
        fut = [
            dbc.Row([
                html.Div([
                    html.H3('(2020-2040)', style={'text-align': 'center'})
                ], style=SUBTITLE_YEAR)
            ]),
            dbc.Row([
                dbc.Col(
                    dcc.Graph(figure=sleuth_fast_map),
                    style={
                        "width": "26rem",
                        'margin-bottom': '1rem',
                        "display": "flex",
                        "justify-content": "center",
                    }
                ),
                dbc.Col(
                    dcc.Graph(figure=sleuth_usual_map),
                    style={
                        "width": "26rem",
                        'margin-bottom': '1rem',
                        "display": "flex",
                        "justify-content": "center"
                    }
                )
            ]),
            dbc.Row([
                dbc.Col(
                    dcc.Graph(figure=sleuth_slow_map),
                    style={
                        "width": "26rem",
                        "display": "flex",
                        "justify-content": "center"
                    }
                ),
                dbc.Col(
                    dcc.Graph(figure=sleuth_graph),
                    style={"width": "26rem",
                           "display": "flex",
                           "justify-content": "center"
                           }
                )
            ]),
            dbc.Row([
                dbc.Col(
                    html.P('Estos resultados son demostrativos, la funcionalidad de esta pestaña continua en desarrollo.'),
                    className='src-txt-style')
            ], className='src-style-preliminar'),
            dbc.Row([
                dbc.Col(html.P('Fuente:'), className='src-txt-style'),
                dbc.Col(html.A(
                    'Sleuth Python',
                    href='https://github.com/gperaza/sleuth_python',
                    target="_blank", className='src-link-style'))
            ], className='src-style-pp'),
        ]
        return fut


@app.callback(Output('download-GIF', 'data'),
              Input('gif-button', 'n_clicks'),
              State('cou-dro', 'value'),
              State('cit-dro', 'value'),
              prevent_initial_call=True)
def download_gif(n_clicks, country, city):
    '''
    Callback to create and download a better quality landsat gif
    in pasado-urbanización.

    State:
        (A state would save the colected data but it won't trigger anything)
        - value (cou-dro): contry value.
        - value (cit-dro): city value.

    Input:
        - n_clicks (gif-button): a click triggers the callback.

    Output:
        - data (download-GIF): download automatically the gif
          from gif_dir.

    '''

    if n_clicks > 0:
        gif_dir = pts.make_gif(country, city)
        return dcc.send_file(gif_dir)
    else:
        return PreventUpdate


if __name__ == "__main__":
    th.Timer(0.000000001, open_browser).start()
    app.run_server(host='0.0.0.0')
