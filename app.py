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

import time

import dash
from dash import ctx, DiskcacheManager
from dash.dependencies import Input, Output, ClientsideFunction, State
from dash.exceptions import PreventUpdate
from dash_html import set_html
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import pathlib
import diskcache
cache = diskcache.Cache("./cache")
background_callback_manager = DiskcacheManager(cache)

from src.model_data import JobShopData
from src.job_shop_scheduler import run_shop_scheduler
from app_configs import SCENARIOS, RESOURCE_NAMES, HTML_CONFIGS

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    prevent_initial_callbacks="initial_duplicate",
    background_callback_manager=background_callback_manager
)
app.title = HTML_CONFIGS['title']

server = app.server
app.config.suppress_callback_exceptions = True

BASE_PATH = pathlib.Path(__file__).parent.resolve()
DATA_PATH = BASE_PATH.joinpath("input").resolve()

model_data = JobShopData()


@app.callback(
    Output("left-column", "className"),
    inputs=[
        Input("left-column-collapse", "n_clicks"),
        State("left-column", "className"),
    ],
    prevent_initial_call=True,
)
def toggle_left_column(left_column_collapse: int, class_name: str) -> str:
    """Toggles left column 'collapsed' class that hides and shows the left column.

    Args:
        left_column_collapse (int): The (total) number of times the collapse button has been clicked.
        class_name (str): Current class name of the left column, 'collapsed' if not visible, empty string if visible

    Returns:
        str: The new class name of the left column.
    """
    if class_name:
        return ""
    return "collapsed"


@app.callback(
    Output('solver-select', 'className'),
    Output('solver-select', 'value'),
    inputs=[
        Input('model-select', 'value'),
    ],
    prevent_initial_call=True
)
def update_solver_options(model_value: str) -> list[str]:
    """Hides and shows classical solver option using 'hide-classic' class

    Args:
        model_value (str): Currently selected model from model-select dropdown.

    Returns:
        str: The new class name of the solver-select checklist.
        list: Unselects MIP and selects Hybrid or no update.
    """

    if model_value == 'QM':
        return 'hide-classic', ['Hybrid']
    return '', dash.no_update



def get_minimum_task_times(job_shop_data: JobShopData) -> pd.DataFrame:
    """Takes a JobShopData object, gets the minimum time each
    task can be completed by, and generates the Jobs to Be Scheduled
    Gantt chart.

    Args:
        job_shop_data (JobShopData): The data for the job shop scheduling problem.

    Returns:
        pd.DataFrame: A DataFrame of the jobs to be scheduled including Task, Start, Finish, 
        Resource, and delta.
    """
    task_data = []
    for tasks in job_shop_data.job_tasks.values():
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

def get_empty_figure(message: str) -> go.Figure:
    """Generates an empty chart figure message.
    This is used to replace the chart object
    when no chart is available.

    Args:
        message (str): The message to display in the center of the chart.

    Returns:
        go.Figure: A Plotly figure object containing the message.
    """    
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
    tuple[go.Figure, go.Figure, go.Figure, go.Figure, dict, dict]:
    """Loads the initial figures for the Gantt charts.

    Args:
        n_clicks (int): The number of times the run button has been
        clicked.

    Returns:
        go.Figure: Gantt chart for the D-Wave hybrid solver
        go.Figure: Gantt chart for the Classical solver
        go.Figure: Results table for the D-Wave hybrid solver
        go.Figure: Results table for the Classical solver
        dict: Style for the D-Wave summary table
        dict: Style for the Classical summary table
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
    """Updates the tab loading state after the run button
    or cancel button has been clicked. 

    Args:
        run_click (int): The number of times the run button has been
            clicked.
        cancel_click (int): The number of times the cancel button has
            been clicked.

    Returns:
        str: Class name for the D-Wave tab
        str: Class name for the Classical tab
        go.Figure: Figure for the D-Wave tab
        go.Figure: Figure for the Classical tab
        dict: Style for the D-Wave summary table
        dict: Style for the Classical summary table
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
    running=[
        (Output("cancel-button", "style"), {"display": "inline-block"}, {"display": "none"}),
        (Output("run-button", "style"), {"display": "none"}, {"display": "inline-block"}),
    ],
    cancel=[Input("cancel-button", "n_clicks")],
    prevent_initial_call=True
)
def run_optimization_cqm(run_click: int, model: str, solver: str, scenario: str, time_limit: int) \
    -> tuple[go.Figure, go.Figure, str, dict]:
    """Runs optimization using the D-Wave hybrid solver.

    Args:
        run_click (int): The number of times the run button has been
            clicked.
        model (str): The model to use for the optimization.
        solver (str): The solver to use for the optimization.
        scenario (str): The scenario to use for the optimization.
        time_limit (int): The time limit for the optimization.

    Returns:
        go.Figure: Gantt chart for the D-Wave hybrid solver
        go.Figure: Results table for the D-Wave hybrid solver
        str: Class name for the D-Wave tab
        dict: Style for the D-Wave summary table
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
                                         solver_time_limit=time_limit,
                                         )
            end = time.time()      
            fig = generate_gantt_chart(df=results, y_axis='Job', color='Resource')
            table = generate_output_table(results['Finish'].max(), time_limit, int(end - start))
            return fig,table,'tab-success', {'visibility': 'visible'}
        else:
            time.sleep(0.1)
            empty_figure = get_empty_figure(HTML_CONFIGS['solver_messages']['dwave']['solver_not_chosen'])
            table = generate_output_table(0, 0, 0)
            return empty_figure,table,'tab-warning', {'visibility': 'hidden'}
    else:
        raise PreventUpdate


@app.callback(
    Output("mip_gantt_chart", 'figure'),
    Output('mip_summary_table', 'figure'),
    Output('mip_tab', 'className'),
    Output('mip_summary_table', 'style'),
    background=True,
    inputs=[
        Input('run-button', 'n_clicks'),
        State("model-select", "value"),
        State("solver-select", "value"),
        State("scenario-select", "value"),
        State("solver_time_limit", "value")
    ],
    running=[
        (Output("cancel-button", "style"), {"display": "inline-block"}, {"display": "none"}),
        (Output("run-button", "style"), {"display": "none"}, {"display": "inline-block"}),
    ],
    cancel=[Input("cancel-button", "n_clicks")],
    prevent_initial_call=True
)
def run_optimization_mip(run_click: int,
                         model: str,
                         solver: str,
                         scenario: str,
                         time_limit: int) \
                         -> tuple[go.Figure, go.Figure, str, dict]:
    """Runs optimization using the COIN-OR Branch-and-Cut solver.

    Args:
        run_click (int): The number of times the run button has been
            clicked.
        model (str): The model to use for the optimization.
        solver (str): The solver to use for the optimization.
        scenario (str): The scenario to use for the optimization.
        time_limit (int): The time limit for the optimization.

    Returns:
        go.Figure: Gantt chart for the Classical solver
        go.Figure: Results table for the Classical solver
        str: Class name for the Classical tab
        dict: Style for the Classical summary table
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
                fig = get_empty_figure(HTML_CONFIGS['solver_messages']['mip']['quadratic_error'])
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
                    fig = get_empty_figure(HTML_CONFIGS['solver_messages']['mip']['no_solution'])
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
            message = HTML_CONFIGS['solver_messages']['mip']['solver_not_chosen']
            empty_figure = get_empty_figure(message)
            mip_table = generate_output_table(0, 0, 0)
            mip_table_style = {'visibility': 'hidden'}
            return empty_figure, mip_table, 'tab-warning', mip_table_style
    else:
        raise PreventUpdate


@app.callback(
    Output('unscheduled_gantt_chart', 'figure'),
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
        scenario: str=None,
        df: pd.DataFrame=None,
        y_axis: str='Job',
        color: str='Resource') -> go.Figure:
    """Generates a Gantt chart of the unscheduled tasks for the given scenario.

    Args:
        scenario (str): The name of the scenario; must be a key in SCENARIOS.
        df (pd.DataFrame): A DataFrame containing the data to plot. If this is
            not None, then the scenario argument will be ignored.
        y_axis (str): The column to use for the y-axis of the Gantt chart.
        color (str): The column to use for the color of the Gantt chart.

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
    """Generates an output table for the optimization results.
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


# import the html code and sets it in the app
# creates the visual layout and app (see `dash_html.py`)
set_html(app)


app.clientside_callback(
    ClientsideFunction(namespace="clientside", function_name="resize"),
    Output("output-clientside", "children"),
    [Input("wait_time_table", "children")],
)


# Run the server
if __name__ == "__main__":
    app.run_server(debug=True)