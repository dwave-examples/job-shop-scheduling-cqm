'''
This file is copied from apps/dash-clinical-analytics/app.py under the following license

MIT License

Copyright (c) 2019 Plotly

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Modifications are licensed under

Apache License, Version 2.0
(see ./LICENSE for details)

'''

'''
TODO:
1) connect the run button to the optimization function (x)
2) connect the reset button to the reset function (z)
3) split the output onto a separate tab (x)
4) add option to switch solver / run multiple solvers against eachother (x)
5) add benchmarking results from (Ku & Beck 2016) (likely won't add this)
6) connect output table to live results (x)
7) add loading bar to indicate optimization is running (x)
8) sort y-axis alphabetically
9) run multiple models in parallel
10) add timer while loading
11) same x-axis for us and COIN-OR
12) finish get_empty_figure() func (x)
'''
import json


import dash
from dash import dcc, html, ctx
from dash.dependencies import Input, Output, ClientsideFunction
import plotly.graph_objs as go
from plotly.colors import n_colors
import plotly.express as px
import pandas as pd
from datetime import datetime as dt
import pathlib

from src.model_data import JobShopData
from src.job_shop_scheduler import run_shop_scheduler

#built-in color scales at https://plotly.com/python/builtin-colorscales/y

SCENARIOS = {
    '3x3': "instance3_3.txt",
    '5x5': "instance5_5.txt",
    '10x10': "instance10_10.txt",
    '15x15': "instance15_15.txt",
    '20x15': "instance20_15.txt",
    '20x25': "instance20_25.txt",
    '30x30': "instance30_30.txt"
}

MODEL_OPTIONS = {
    "Quadtratic Model": "QM",
    "Mixed Integer Model": "MIP"
}

SOLVER_OPTIONS = {
    "D-Wave Hybrid Solver": "Hybrid",
    "COIN-OR Branch-and-Cut Solver (CBC)": "MIP"
}

RESOURCE_NAMES = json.load(open('./src/data/resource_names.json', 'r'))['industrial']

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    prevent_initial_callbacks="initial_duplicate"
)
app.title = "Job Shop Scheduling Demo"

server = app.server
app.config.suppress_callback_exceptions = True

# Path
BASE_PATH = pathlib.Path(__file__).parent.resolve()
DATA_PATH = BASE_PATH.joinpath("input").resolve()

model_data = JobShopData()


def get_minimum_task_times(job_shop_data: JobShopData) -> pd.DataFrame:
    """This function takes a JobShopData object and gets the minimum time each
    task can be completed by, considering the precedence tasks. This function
    is used to generate the Jobs to Be Scheduled Gantt chart.

    Args:
        job_shop_data (JobShopData): The data for the job shop scheduling problem.

    Returns:
        pd.DataFrame: A DataFrame that can be used to generate a Gantt chart in the
        dashboard. The DataFrame has the following columns: Task, Start, Finish, 
        Resource, and delta.
    """
    task_data = []
    for job, tasks in job_shop_data.job_tasks.items():
        start_time = 0
        for task in tasks:
            end_time = start_time + task.duration
            task_data.append({'Job': task.job, 'Start': start_time, 'Finish': end_time, 'Resource': task.resource})
            start_time = end_time
    df = pd.DataFrame(task_data)
    df['delta'] = df.Finish - df.Start
    df['Job'] = df['Job'].astype(str)
    df['Resource'] = df['Resource'].astype(str)
    return df


def description_card():
    """
    :return: A Div containing dashboard title & descriptions.
    """
    return html.Div(
        id="description-card",
        children=[
            html.H5("Job Shop Scheduling"),
            html.H3("Welcome to the Job Shop Scheduling Dashboard"),
            html.Div(
                id="intro",
                children="Run the job shop scheduling problem for several different scenarios. Explore the Gantt Chart for solution details.",
            ),
        ],
    )


def generate_duration_slider(resource, min_value=1, max_value=5):
    """

    :param resource: The resource to generate a slider for.
    :param min: The minimum duration.
    :param max: The maximum duration.
    :return: A Div containing a slider for the given resource.
    """
    return html.Div(
        id="duration-slider-div{}".format(resource),
        children=[
            html.P("Duration for {}".format(resource)),
            dcc.Slider(
                id="duration-slider-{}".format(resource),
                min=min_value,
                max=max_value,
                step=1,
                value=1,
                marks={i: "{}".format(i) for i in range(min_value, max_value + 1)},
            ),
        ],
    )


def get_empty_figure(message):
    fig = go.Figure()
    fig.update_layout(
        xaxis =  { "visible": False },
        yaxis = { "visible": False },
        annotations = [
            {   
                "text": message,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {
                    "size": 28
                }
            }
        ]
    )
    return fig


def generate_control_card():
    """
    :return: A Div containing controls for graphs.
    """
    scenario_options = []
    for scenario in SCENARIOS:
        scenario_options.append({"label": scenario, "value": scenario})
    model_options = []
    for model_option, model_value in MODEL_OPTIONS.items():
        model_options.append({"label": model_option, "value": model_value})
    solver_options = []
    for solver_option, solver_value in SOLVER_OPTIONS.items():
        solver_options.append({"label": solver_option, "value": solver_value})
    return html.Div(
        id="control-card",
        children=[
            html.P("Select Scenario"),
            dcc.Dropdown(
                id="scenario-select",
                options=scenario_options,
                value=scenario_options[0]["value"],
            ),
            html.P("Select Model"),
            dcc.Dropdown(
                id="model-select",
                options=model_options,
                value=model_options[0]["value"],
            ),
            html.P("Select Solver"),
            dcc.Checklist(
                id="solver-select",
                options=solver_options,
                value=[solver_options[0]["value"]],
            ),
            html.Br(),
            html.Div(
                id="button-group",
                children=[
                    html.Button(id="run-button", children="Run Optimization", className='dwave-button', n_clicks=0)
                ]
            ),
        ],
    )


@app.callback(
    Output("optimized_gantt_chart", 'figure'),
    Output("optimized_gantt_chart", 'style'),
    Output("optimized_gantt_chart_alt", 'figure'),
    Output("optimized_gantt_chart_alt", 'style'),
    [
        Input('run-button', 'n_clicks'),
        Input("model-select", "value"),
        Input("solver-select", "value"),
    ]
)
def run_optimization_cqm(run_click, model, solver):
    if ctx.triggered_id == "run-button":
        if 'Hybrid' in solver:
            use_mip_solver = False
            allow_quadratic_constraints = model == 'QM'
            results = run_shop_scheduler(model_data, use_mip_solver=use_mip_solver, allow_quadratic_constraints=allow_quadratic_constraints)
            fig = generate_gantt_chart(df=results, y_axis='Resource', color='Job')
            fig2 = generate_gantt_chart(df=results, y_axis='Job', color='Resource')
            return fig, {'visibility': 'visible'}, fig2, {'visibility': 'visible'}
        else:
            empty_figure = get_empty_figure('Choose D-Wave Hybrid Solver to run this solver')
            return empty_figure, {'visibility': 'visible'}, empty_figure, {'visibility': 'visible'}
    elif run_click == 0:
        empty_figure = get_empty_figure('Run optimization to see results')
        return empty_figure, {'visibility': 'visible'}, empty_figure, {'visibility': 'visible'}


@app.callback(
    Output("mip_gantt_chart", 'figure'),
    Output("mip_gantt_chart", 'style'),
    Output("mip_gantt_chart_alt", 'figure'),
    Output("mip_gantt_chart_alt", 'style'),
    [
        Input('run-button', 'n_clicks'),
        Input("model-select", "value"),
        Input("solver-select", "value"),
    ]
)
def run_optimization_mip(run_click, model, solver):
    if ctx.triggered_id == "run-button":
        if 'MIP' in solver:
            use_mip_solver = True
            allow_quadratic_constraints = model == 'QM'
            if allow_quadratic_constraints:
                fig = get_empty_figure('Unable to run MIP solver with quadratic constraints')
                fig2 = get_empty_figure('Unable to run MIP solver with quadratic constraints')
            else:
                results = run_shop_scheduler(model_data, use_mip_solver=use_mip_solver, allow_quadratic_constraints=allow_quadratic_constraints)
                fig = generate_gantt_chart(df=results, y_axis='Resource', color='Job')
                fig2 = generate_gantt_chart(df=results, y_axis='Job', color='Resource')
            return fig, {'visibility': 'visible'}, fig2, {'visibility': 'visible'}
        else:
            message = 'Select COIN-OR Branch and Cut Solver to run this solver'
            return get_empty_figure(message), {'visibility': 'visible'}, get_empty_figure(message), {'visibility': 'visible'}
    elif run_click == 0:
        empty_figure = get_empty_figure('Run optimization to see results')
        return empty_figure, {'visibility': 'visible'}, empty_figure, {'visibility': 'visible'}



@app.callback(
    Output('unscheduld_gantt_chart', 'figure'),
    [
        Input("scenario-select", "value"),
    ]
)
def generate_unscheduled_gantt_chart(scenario):
    """Generates a Gantt chart of the unscheduled tasks for the given scenario.

    Args:
        scenario (str): The name of the scenario; must be a key in SCENARIOS.

    Returns:
        : A Plotly figure object.

    """
    fig = generate_gantt_chart(scenario=scenario, y_axis='Job', color='Resource')
    return fig


def generate_gantt_chart(scenario=None, df=None, y_axis: str='Job', color: str='Resource'):
    """Generates a Gantt chart of the unscheduled tasks for the given scenario.

    Args:
        scenario (str): The name of the scenario; must be a key in SCENARIOS.

    Returns:
        : A Plotly figure object.

    """
    if df is None:
        filename = SCENARIOS[scenario]
        if 'json' in filename:
            model_data.load_from_json(DATA_PATH.joinpath(filename), resource_names=RESOURCE_NAMES)
        else:
            model_data.load_from_file(DATA_PATH.joinpath(filename), resource_names=RESOURCE_NAMES)
        df = get_minimum_task_times(model_data)
    if y_axis == 'Job':
        #convert to int if it is a string
        if df['Job'].dtype == 'object':
            df['JobInt'] = df['Job'].str.replace('Job', '').astype(int)
        else:
            df['JobInt'] = df['Job']
        df = df.sort_values(by=['JobInt', color, 'Start'])
        df = df.drop(columns=['JobInt'])
    else:
        df = df.sort_values(by=[y_axis, color, 'Start'])
    df['delta'] = df.Finish - df.Start
    df[color] = df[color].astype(str)
    df[y_axis] = df[y_axis].astype(str)
    num_items = len(df[color].unique())
    #d-wave primary
    colorscale = px.colors.make_colorscale(['#ffa143', '#17bebb', '#2a7de1'])
    #d-wave bright
    colorscale = px.colors.make_colorscale(['#FFA143', '#06ECDC', '#03b8ff'])
    colorscale = 'Agsunset'
    fig = px.timeline(df, x_start="Start", x_end="Finish", y=y_axis, color=color,
    color_discrete_sequence = px.colors.sample_colorscale(colorscale, [n/(num_items -1) for n in range(num_items)]))

    # fig = px.timeline(df, x_start="Start", x_end="Finish", y=y_axis, color=color,
    # color_discrete_sequence = px.colors.sample_colorscale("Plotly3", [n/(num_items -1) for n in range(num_items)]))

    for idx, fig_data in enumerate(fig.data):
        resource = fig.data[idx].name
        data_list = []
        for job in fig.data[idx].y:
            try:
                data_list.append(df[(df[y_axis] == job) & (df[color] == resource)].delta.tolist()[0])
            except:
                continue
        fig.data[idx].x = data_list
        
    fig.layout.xaxis.type = 'linear'
    return fig


#set the application HTML
app.layout = html.Div(
    id="app-container",
    children=[
        # Banner
        html.Div(
            id="banner",
            className="banner",
            children=[html.Img(src="assets/dwave_logo.svg")],
        ),
        # Left column
        html.Div(
            id="left-column",
            className="four columns",
            children=[description_card(), generate_control_card()]
            + [
                html.Div(
                    ["initial child"], id="output-clientside", style={"display": "none"}
                )
            ],
        ),
        # Right column
        html.Div(
            id="right-column",
            className="gantt-container",
            children=[
                dcc.Tabs(id="tabs", value='input_tab', children=[
                    dcc.Tab(label='Input',
                            value='input_tab',
                            className='tab',
                            children=[html.Div(
                                dcc.Loading(id = "loading-icon-input", children=[ html.Div(
                                id="unscheduled_gantt_chart_card",
                                className="gantt-div",
                                children=[
                                    html.B("Jobs to be Scheduled", className="gantt-title"),
                                    html.Hr(className="gantt-hr"),
                                    dcc.Graph(id="unscheduld_gantt_chart"),
                                ],
                                )])
                            )])
                        ,
                        dcc.Tab(label='D-Wave',
                                value='dwave_tab', 
                                children=[html.Div(
                                    dcc.Loading(id = "loading-icon-dwave", 
                                        children=[ 
                                            html.Div(
                                                id="optimized_gantt_chart_card",
                                                className="gantt-div",
                                                children=[
                                                    html.B("D-Wave Hybrid Solver", className="gantt-title"),
                                                    html.Hr(className="gantt-hr"),
                                                    dcc.Graph(id="optimized_gantt_chart", style={'visibility': 'visible'}),
                                                ]
                                                ),
                                            html.Div(
                                                id="optimized_gantt_chart_card_alt",
                                                className="gantt-div",
                                                children=[
                                                    html.B("D-Wave Hybrid Solver Alt", className="gantt-title"),
                                                    html.Hr(className="gantt-hr"),
                                                    dcc.Graph(id="optimized_gantt_chart_alt", style={'visibility': 'visible'}),
                                                ]
                                                )
                                            ], 
                                        type="default"))
                                 ])
                    ,
                    dcc.Tab(label='COIN-OR', value='coin_or_tab', children=[html.Div(
                    dcc.Loading(id = "loading-icon-coinor", 
                        children=[ 
                            html.Div(
                                id="mip_gantt_chart_card",
                                className="gantt-div",
                                children=[
                                    html.B("COIN-OR", className="gantt-title"),
                                    html.Hr(className="gantt-hr"),
                                    dcc.Graph(id="mip_gantt_chart", style={'visibility': 'hidden'}),
                                ]
                            ),
                            html.Div(
                                id="mip_gantt_chart_card_alt",
                                className="gantt-div",
                                children=[
                                    html.B("COIN-OR", className="gantt-title"),
                                    html.Hr(className="gantt-hr"),
                                    dcc.Graph(id="mip_gantt_chart_alt", style={'visibility': 'hidden'}),
                                ]
                            )
                        ], 
                        type="default"))])
             ])
             ])
    ])


app.clientside_callback(
    ClientsideFunction(namespace="clientside", function_name="resize"),
    Output("output-clientside", "children"),
    [Input("wait_time_table", "children")],
)


# Run the server
if __name__ == "__main__":
    app.run_server(debug=True)