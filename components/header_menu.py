from dash_extensions.enrich import  html
import dash_bootstrap_components as dbc

# Label style in the dropdowns
TOGGLE_STYLE = {
    'background-color': '#FBFBFB',
    'border-color': '#FBFBFB',
    'color': 'gray',
    'font-size': '1.125rem'
}

# Dropdown menu stylesheet CSS
DROPDOWN_MENU = {
    "margin-left": "15%",
    "margin-bottom": "-2rem",
    "background-color": "#FBFBFB",
    "display": "flex",
    "padding-top": "35px",
    "padding-left": "60px"
}

menu = html.Div([dbc.Row([
    dbc.Col(
        dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem("Urbanización", href="/pas/urb"),
                dbc.DropdownMenuItem("Población", href="/pas/pop"),
                dbc.DropdownMenuItem("Cobertura de suelo", href="/pas/lan"),
            ],
            label="Pasado",
            align_end=False,
            toggle_style=TOGGLE_STYLE
        )),

    dbc.Col(
        dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem("Urbanización", href="/pre/urb"),
                dbc.DropdownMenuItem("Población", href="/pre/pop"),
                dbc.DropdownMenuItem("Cobertura de suelo", href="/pre/lan"),
            ],
            label="Presente",
            align_end=False,
            toggle_style=TOGGLE_STYLE,
            className='menu-items'
        )),
    dbc.Col(
        dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem("Crecimiento futuro", href="/fut")
            ],
            label="Futuro",
            align_end=False,
            toggle_style=TOGGLE_STYLE
        )),
])], style=DROPDOWN_MENU)