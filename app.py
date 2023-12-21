'''
This file is forked from apps/dash-clinical-analytics/app.py under the following license

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

import json
import time

import dash
from dash import dcc, html, ctx, DiskcacheManager
from dash.dependencies import Input, Output, ClientsideFunction, State
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import pathlib
import diskcache
cache = diskcache.Cache("./cache")
background_callback_manager = DiskcacheManager(cache)

from src.model_data import JobShopData
from src.job_shop_scheduler import run_shop_scheduler

#built-in color scales at https://plotly.com/python/builtin-colorscales/y


SCENARIOS = {
    '3x3': "instance3_3.txt",
    '5x5': "instance5_5.txt",
    '10x10': "instance10_10.txt",
    '15x15': "taillard15_15.txt",
    '20x15': "instance20_15.txt",
    '20x25': "instance20_25.txt",
    '30x30': "instance30_30.txt"
}

MODEL_OPTIONS = {
    "Mixed Integer Quadratic Model": "QM",
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
    prevent_initial_callbacks="initial_duplicate",
    background_callback_manager=background_callback_manager
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
    fig.update_layout(
    margin=dict(l=20, r=20, t=10, b=10),
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
            html.P("Select Scenario", className='control-p'),
            dcc.Dropdown(
                id="scenario-select",
                options=scenario_options,
                value=scenario_options[0]["value"],
                clearable=False
            ),
            html.P("Select Model", className='control-p'),
            dcc.Dropdown(
                id="model-select",
                options=model_options,
                value=model_options[0]["value"],
                clearable=False
            ),
            html.P("Select Solver", className='control-p'),
            dcc.Checklist(
                id="solver-select",
                options=solver_options,
                value=[solver_options[0]["value"]],
            ),
            html.P("Solver Time Limit", className='control-p'),
            dcc.Input(
            id="solver_time_limit",
            type='number',
            value=20,
            min=5,
            max=300,
            step=5,
            ),
            html.Div(
                id="button-group",
                children=[
                    html.Button(id="run-button", children="Run Optimization", className='run-button', n_clicks=0),
                    html.Button(
                        id="cancel-button",
                        children="Cancel Optimization",
                        className='cancel-button',
                        n_clicks=0,
                        style={"visibility": "hidden"}
                        )
                ]
            ),
        ],
    )


@app.callback(
    Output('optimized_gantt_chart', 'figure', allow_duplicate=True),
    Output('mip_gantt_chart', 'figure', allow_duplicate=True),
    Output('dwave_summary_table', 'figure', allow_duplicate=True),
    Output('mip_summary_table', 'figure', allow_duplicate=True),
    Output('dwave_summary_table', 'style', allow_duplicate=True),
    Output('mip_summary_table', 'style', allow_duplicate=True),  
    [Input('run-button', 'n_clicks')],
    prevent_initial_call=False
)
def load_initial_figures(n_clicks: int) -> \
    tuple[go.Figure, go.Figure, str, str]:
    """This function loads the initial figures for the Gantt charts.
    It will only be used on the initial load; after that this will
    pass PreventUpdate

    Args:
        n_clicks (int): The number of times the run button has been
        clicked.

    Raises:
        PreventUpdate: If the run button has already been clicked,
        this will raise PreventUpdate to prevent the initial figures
        from being loaded again.

    Returns:
        tuple: A tuple of two Plotly figures and two strings. The first
        figure is the Gantt chart for the D-Wave hybrid solver, the second
        figure is the Gantt chart for the COIN-OR Branch-and-Cut solver,
        the first string is the style for the D-Wave summary table, and
        the second string is the style for the COIN-OR summary table.
    """    
    if n_clicks == 0:
        empty_figure = get_empty_figure('Run optimization to see results')
        empty_table = generate_output_table(0,0,0)
        return empty_figure, empty_figure, empty_table, empty_table,\
            {'visibility': 'hidden'}, {'visibility': 'hidden'}
    else:
        raise PreventUpdate

@app.callback(
    Output("dwave_tab", 'className', allow_duplicate=True),
    Output("mip_tab", 'className', allow_duplicate=True),
    Output('optimized_gantt_chart', 'figure', allow_duplicate=True),
    Output('mip_gantt_chart', 'figure', allow_duplicate=True),
    Output('dwave_summary_table', 'style', allow_duplicate=True),
    Output('mip_summary_table', 'style', allow_duplicate=True),
    [
        Input('run-button', 'n_clicks'),
        Input("cancel-button", "n_clicks")
    ]
)
def update_tab_loading_state(run_click: int, cancel_click: int) -> \
    tuple[str, str, go.Figure, go.Figure, dict, dict]:
    """This function updates the tab loading state after the run button
    or cancel button has been clicked. 

    Args:
        run_click (int): The number of times the run button has been
            clicked.
        cancel_click (int): The number of times the cancel button has
            been clicked.

    Raises:
        PreventUpdate: If the event is trigged by anything other than 
            the run button or cancel button, this will raise PreventUpdate
            to prevent the tab loading state from being updated.

    Returns:
        tuple: A tuple of four objects: the class name for the D-Wave tab,
        the class name for the Classical tab, the figure for the D-Wave tab,
        and the figure for the Classical  tab.
    """    
    if ctx.triggered_id == "run-button":
        if run_click == 0:
            empty_figure = get_empty_figure('Run optimization to see results')
            return 'tab', 'tab', empty_figure, empty_figure, {'visibility': 'hidden'}, {'visibility': 'hidden'}
        else:
            empty_figure = get_empty_figure('Running...')
            return 'tab-loading', 'tab-loading', empty_figure, empty_figure, {'visibility': 'hidden'}, {'visibility': 'hidden'}
    elif ctx.triggered_id == "cancel-button":
        if cancel_click > 0:
            empty_figure = get_empty_figure('Last run cancelled prior to completion. Re-run to see results')
            return 'tab', 'tab', empty_figure, empty_figure, {'visibility': 'hidden'}, {'visibility': 'hidden'}
    raise PreventUpdate


@app.callback(
    Output("optimized_gantt_chart", 'figure'),
    Output('dwave_summary_table', 'figure'),
    Output('dwave_tab', 'className'),
    Output('dwave_summary_table', 'style'),
    background=True,
    inputs=[
        Input('run-button', 'n_clicks'),
        State("model-select", "value"),
        State("solver-select", "value"),
        State("scenario-select", "value"),
        State("solver_time_limit", "value"),
    ],
    running=[(Output("cancel-button", "style"), {"visibility": "visible"}, {'visibility': 'hidden'}),
             (Output("run-button", "style"), {"visibility": "hidden"}, {'visibility': 'visible'})],
    cancel=[Input("cancel-button", "n_clicks")],
    prevent_initial_call=True
)
def run_optimization_cqm(run_click: int, model: str, solver: str, scenario: str, time_limit: int) \
    -> tuple[go.Figure, go.Figure, str, str]:
    """This function runs the optimization using the D-Wave hybrid solver.

    Args:
        run_click (int): The number of times the run button has been
            clicked.
        model (str): The model to use for the optimization.
        solver (str): The solver to use for the optimization.
        scenario (str): The scenario to use for the optimization.
        time_limit (int): The time limit for the optimization.

    Raises:
        PreventUpdate: If this was not trigged by a run-button click,
            this will raise PreventUpdate to prevent the optimization
            from running.

    Returns:
        tuple: A tuple of four objects: the Plotly figure for the Gantt
            chart, the Plotly figure for the output table, the class name
            for the tab, and the style for the output table.
    """    
    if run_click == 0 or ctx.triggered_id != "run-button":
        empty_figure = get_empty_figure('Run optimization to see results')
        return empty_figure, 'tab'
    if ctx.triggered_id == "run-button":
        if 'Hybrid' in solver:
            model_data = JobShopData()
            filename = SCENARIOS[scenario]
            model_data.load_from_file(DATA_PATH.joinpath(filename), resource_names=RESOURCE_NAMES)
            allow_quadratic_constraints = model == 'QM'
            start = time.time()
            results = run_shop_scheduler(model_data, 
                                         use_mip_solver=False,
                                         allow_quadratic_constraints=allow_quadratic_constraints,
                                         solver_time_limit=time_limit)   
            end = time.time()      
            fig = generate_gantt_chart(df=results, y_axis='Job', color='Resource')
            table = generate_output_table(results['Finish'].max(), time_limit, int(end - start))
            return fig,table,'tab-success', {'visibility': 'visible'}
        else:
            time.sleep(0.1)
            empty_figure = get_empty_figure('Choose D-Wave Hybrid Solver to run this solver')
            table = generate_output_table()
            return empty_figure,table,'tab-warning', {'visibility': 'hidden'}
    else:
        raise PreventUpdate


@app.callback(
    Output("mip_gantt_chart", 'figure'),
    Output('mip_summary_table', 'figure'),
    Output('mip_tab', 'className'),
    Output('mip_summary_table', 'style'),
    [
        Input('run-button', 'n_clicks'),
        State("model-select", "value"),
        State("solver-select", "value"),
        State("scenario-select", "value"),
        State("solver_time_limit", "value")
    ],
    running=[(Output("cancel-button", "style"), {"visibility": "visible"}, {'visibility': 'hidden'}),
             (Output("run-button", "style"), {"visibility": "hidden"}, {'visibility': 'visible'})],
    cancel=[Input("cancel-button", "n_clicks")],
    prevent_initial_call=True
)
def run_optimization_mip(run_click: int,
                         model: str,
                         solver: str,
                         scenario: str,
                         time_limit: int) \
                         -> tuple[go.Figure, go.Figure, str, str]:
    """This function runs the optimization using the COIN-OR Branch-and-Cut solver.

    Args:
        run_click (int): The number of times the run button has been
            clicked.
        model (str): The model to use for the optimization.
        solver (str): The solver to use for the optimization.
        scenario (str): The scenario to use for the optimization.
        time_limit (int): The time limit for the optimization.

    Raises:
        PreventUpdate: If this was not trigged by a run-button click,
            this will raise PreventUpdate to prevent the optimization
            from running.

    Returns:
        tuple: A tuple of four objects: the Plotly figure for the Gantt
            chart, the Plotly figure for the output table, the class name
            for the tab, and the style for the output table.
    """    
    if run_click == 0:
        empty_figure = get_empty_figure('Run optimization to see results')
        return empty_figure, 'tab'
    if ctx.triggered_id == "run-button":
        if 'MIP' in solver:
            model_data = JobShopData()
            filename = SCENARIOS[scenario]
            model_data.load_from_file(DATA_PATH.joinpath(filename), resource_names=RESOURCE_NAMES)
            use_mip_solver = True
            allow_quadratic_constraints = model == 'QM'
            if allow_quadratic_constraints:
                time.sleep(0.1) #sleep to allow the loading icon to appear first
                fig = get_empty_figure('Unable to run MIP solver with quadratic constraints')
                class_name = 'tab-fail'
                mip_table = generate_output_table(0, 0, 0)
                mip_table_style = {'visibility': 'hidden'}
            else:
                start = time.time()
                results = run_shop_scheduler(model_data, 
                                             use_mip_solver=use_mip_solver,
                                             allow_quadratic_constraints=allow_quadratic_constraints,
                                             solver_time_limit=time_limit)
                end = time.time()
                if len(results) == 0:
                    fig = get_empty_figure('MIP solver failed to find a solution.')
                    mip_table = generate_output_table(0, 0, 0)
                    class_name = 'tab-fail'
                    mip_table_style = {'visibility': 'hidden'}
                else:
                    fig = generate_gantt_chart(df=results, y_axis='Job', color='Resource')
                    class_name = 'tab-success'
                    mip_table = generate_output_table(results['Finish'].max(), time_limit, int(end-start))
                    mip_table_style = {'visibility': 'visible'}
            return fig, mip_table, class_name, mip_table_style  
        else:
            time.sleep(0.1) #sleep to allow the loading icon to appear first
            message = 'Select COIN-OR Branch and Cut Solver to run this solver'
            empty_figure = get_empty_figure(message)
            mip_table = generate_output_table(0, 0, 0)
            mip_table_style = {'visibility': 'hidden'}
            return empty_figure, mip_table, 'tab-warning', mip_table_style
    else:
        raise PreventUpdate


@app.callback(
    Output('unscheduld_gantt_chart', 'figure'),
    [
        Input("scenario-select", "value"),
    ]
)
def generate_unscheduled_gantt_chart(scenario: str) -> go.Figure:
    """Generates a Gantt chart of the unscheduled tasks for the given scenario.

    Args:
        scenario (str): The name of the scenario; must be a key in SCENARIOS.

    Returns:
        go.Figure: A Plotly figure object with the input data
    """
    fig = generate_gantt_chart(scenario=scenario, y_axis='Job', color='Resource')
    return fig


def generate_gantt_chart(
        scenario=None,
        df=None,
        y_axis: str='Job',
        color: str='Resource') -> go.Figure:
    """Generates a Gantt chart of the unscheduled tasks for the given scenario.

    Args:
        scenario (str): The name of the scenario; must be a key in SCENARIOS.

    Returns:
        go.Figure: A Plotly figure object.
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

    for idx, _ in enumerate(fig.data):
        resource = fig.data[idx].name
        data_list = []
        for job in fig.data[idx].y:
            try:
                data_list.append(df[(df[y_axis] == job) & (df[color] == resource)].delta.tolist()[0])
            except:
                continue
        fig.data[idx].x = data_list
        
    fig.layout.xaxis.type = 'linear'
    fig.update_layout(
    margin=dict(l=20, r=20, t=10, b=10),
    xaxis_title="Time Period",
    )
    return fig


def generate_output_table(make_span: int, 
                          solver_time_limit: int, 
                          total_time: int) -> go.Figure:
    """This function generates an output table for the optimization results.
    The table will contain the make-span, solver time limit, and total time
    for the optimization.

    Args:
        make_span (int): The make-span for the optimization.
        solver_time_limit (int): The solver time limit for the optimization.
        total_time (int): The total time for the optimization.

    Returns:
        go.Figure: A Plotly figure object containing the output table.
    """    
    fig = go.Figure(data=[
                        go.Table(header=dict(values=['Make-span', 'Solver Time Limit', 'Total Time']),
                        cells=dict(values=[[make_span], [solver_time_limit], [total_time]]))
                        ]
                    )
    fig.update_layout(
    margin=dict(l=20, r=20, t=10, b=10),
    height=100,
    autosize=False
    )
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
                                html.Div(
                                id="unscheduled_gantt_chart_card",
                                className="gantt-div",
                                children=[
                                    html.B("Jobs to be Scheduled", className="gantt-title"),
                                    html.Hr(className="gantt-hr"),
                                    dcc.Loading(id = "loading-icon-input", children=[ 
                                        dcc.Graph(id="unscheduld_gantt_chart"),
                                    ],
                                )])
                            )])
                        ,
                        dcc.Tab(label='D-Wave',
                                value='dwave_tab',
                                id='dwave_tab',
                                className='tab',
                                children=[html.Div(
                                    html.Div(
                                        id="optimized_gantt_chart_card",
                                        className="gantt-div",
                                        children=[
                                            html.B("D-Wave Hybrid Solver", className="gantt-title"),
                                            html.Hr(className="gantt-hr"),
                                            dcc.Loading(id = "loading-icon-dwave", 
                                            children=[ 
                                                    dcc.Graph(id="optimized_gantt_chart"),
                                                ]
                                            ),
                                            dcc.Graph(id="dwave_summary_table")
                                        ]))
                                    ])
                                    ,
                        dcc.Tab(label='Classical',
                                id='mip_tab',
                                className='tab',
                                value='mip_tab', 
                                children=[html.Div(
                                    html.Div(
                                        id="mip_gantt_chart_card",
                                        className="gantt-div",
                                        children=[
                                            html.B("COIN-OR Branch-and-Cut Solver", className="gantt-title"),
                                            html.Hr(className="gantt-hr"),
                                            dcc.Loading(id = "loading-icon-coinor", 
                                                children=[ 
                                                    dcc.Graph(id="mip_gantt_chart")
                                                    ]
                                                ),
                                            dcc.Graph(id="mip_summary_table")
                                            ]))
                                        ])
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