from dash import Dash, html, dcc, Input, Output, dash_table
import plotly.express as px
import pandas as pd

app = Dash(__name__)

year_kpis = pd.read_csv("./data/dataset_preprocessed_2.csv")

app.layout = html.Div([
    html.Div([
        html.H1(children='Just a Little Data Technical Test Dashboard'),
    ]),
    html.Br(),
    html.Div(children=[
        html.Div(children=[
            html.H3('KPIs per Week'),
            html.Br(),
            html.Label("Choose the KPI"),
            dcc.Dropdown(['gross_revenue', 'net_revenue', 'boxes', 'unique_clients_week'], 'gross_revenue', id="KPI"),
            html.Br(),
            dcc.Graph(
                style={'height': 300},
                id="Graph 1"
            ),
            html.Br(),
        ], style={'padding': 10, 'flex': 1}),
    ], style={'display': 'flex', 'flex-direction': 'row'}),
    html.Br(),
    html.Div([
        html.H3("Net Revenue 6 Months Projection"),
        html.Br(),
        html.Label("Choose the ML Model"),
        dcc.Dropdown(['linear_regression', 'mlp', 'random_forest', 'arima', 'my_model'],
                     'linear_regression', id="ML Model"),
        html.Br(),
        dcc.Graph(
                  style={'height': 300},
                  id="Revenue Projection"
                  ),
    ]),
    html.Br(),
    html.Div([
        html.H3("KPIs per Year"),
        dash_table.DataTable(year_kpis.to_dict('records'), [{"name": i, "id": i} for i in year_kpis.columns])
    ])
])


current_kpi = "gross_revenue"


@app.callback(
    Output("Graph 1", "figure"),
    Input("KPI", "value"))
def update_kpi_line_chart(value):
    global current_kpi
    week_kpis = pd.read_csv("./data/dataset_preprocessed_1.csv")
    fig = px.line(week_kpis, x="week", y=value, color='year')
    current_kpi = value
    return fig


@app.callback(
    Output("Revenue Projection", "figure"),
    Input("ML Model", "value"))
def update_projection_line_chart(value):
    projection = pd.read_csv(f"./data/projections/{value}.csv")
    fig = px.line(projection, x="week", y="projected_net_revenue", color='year')
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
