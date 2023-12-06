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


import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, ClientsideFunction

from plotly.colors import n_colors
import plotly.express as px
import pandas as pd
from datetime import datetime as dt
import pathlib

from src.model_data import JobShopData
from demo import run_job_shop_scheduler


#built-in color scales at https://plotly.com/python/builtin-colorscales/y

SCENARIOS = {
    '3x3': "instance3_3.txt",
    '5x5': "instance5_5.txt",
    '10x10': "instance10_10.txt"
}

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
            task_data.append({'Task': task.job, 'Start': start_time, 'Finish': end_time, 'Resource': task.resource})
            start_time = end_time
    df = pd.DataFrame(task_data)
    df['delta'] = df.Finish - df.Start
    df['Task'] = df['Task'].astype(str)
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


def generate_control_card():
    """

    :return: A Div containing controls for graphs.
    """
    scenario_options = []
    for scenario in SCENARIOS:
        scenario_options.append({"label": scenario, "value": scenario})
    return html.Div(
        id="control-card",
        children=[
            html.P("Select Scenario"),
            dcc.Dropdown(
                id="scenario-select",
                options=scenario_options,
                value=scenario_options[0]["value"],
            ),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Div(
                id="button-group",
                children=[
                    html.Button(id="reset-button", children="Reset", n_clicks=0),
                    html.Button(id="run-button", children="Run Optimization", n_clicks=0)
                ]
            ),
        ],
    )


@app.callback(
    Output("optimization-output", "value"),
    [Input('run-button', 'n_clicks')],
)
def run_optimization(reset_click):
    if reset_click > 0:
        print ('starting optimization')
        
        results = run_job_shop_scheduler(model_data, max_time=5)
        print ('finished optimization')
        # update_gantt_chart_from_results(results)


@app.callback(
    Output('unscheduld_gantt_chart', 'figure'),
    [
        Input("scenario-select", "value"),
        Input("reset-button", "n_clicks"),
    ],
)
def generate_unscheduled_gantt_chart(scenario, reset):
    """Generates a Gantt chart of the unscheduled tasks for the given scenario.

    Args:
        scenario (str): The name of the scenario; must be a key in SCENARIOS.

    Returns:
        : A Plotly figure object.

    """
    filename = SCENARIOS[scenario]
    if 'json' in filename:
        model_data.load_from_json(DATA_PATH.joinpath(filename))
    else:
        model_data.load_from_file(DATA_PATH.joinpath(filename))
    df = get_minimum_task_times(model_data)
    df = df.sort_values(by=['Resource', 'Start'])
    df['delta'] = df.Finish - df.Start

    num_resources = len(df.Resource.unique())
    fig = px.timeline(df, x_start="Start", x_end="Finish", y="Task", color="Resource",
    color_discrete_sequence = px.colors.sample_colorscale("Plotly3", [n/(num_resources -1) for n in range(num_resources)]))

    for idx, fig_data in enumerate(fig.data):
        resource = fig.data[idx].name
        data_list = []

        for job in fig.data[idx].y:
            try:
                data_list.append(df[(df.Task == job) & (df.Resource == resource)].delta.tolist()[0])
            except:
                continue
        fig.data[idx].x = data_list
    fig.layout.xaxis.type = 'linear'
    return fig



@app.callback(
    Output("gantt_chart", "figure", allow_duplicate=True),
    [
        Input("scenario-select", "value"),
        Input("reset-button", "n_clicks"),

    ],
)
def update_gantt_chart(scenario, reset_click):
    # Return to original hm(no colored annotation) by resetting
    return generate_gantt_chart(scenario)


def generate_gantt_chart(scenario_name: str='Bakery', results=None):
    """Generates a Gantt chart of the schedule for the given scenario.

    Args:
        scenario_name (str, optional): The name of the scenario; must
        be a key in SCENARIOS. Defaults to 'Bakery'.

    Returns:
        : A Plotly figure object.

    """
    if results is None:
        df = pd.read_json('input/sample_output.json')
    else:
        df = results
    df['delta'] = df.Finish - df.Start
    print ('output df', df)
    num_tasks = len(df.Task.unique())
    fig = px.timeline(df, x_start="Start", x_end="Finish", y="Resource", color="Task",
    color_discrete_sequence = px.colors.sample_colorscale("matter", [n/(num_tasks -1) for n in range(num_tasks)]))

    for idx, fig_data in enumerate(fig.data):
        task = fig.data[idx].name
        data_list = []

        for resource in fig.data[idx].y:
            try:
                data_list.append(df[(df.Resource == resource) & (df.Task == task)].delta.tolist()[0])
            except:
                continue
        fig.data[idx].x = data_list
    # fig.update_yaxes(autorange="reversed") 
    fig.layout.xaxis.type = 'linear'
    return fig


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
            className="eight columns",
            children=[
                # Gantt chart of scheduled tasks
                    html.Div(
                    id="gantt_chart_card",
                    children=[
                        html.B("Jobs to be Scheduled"),
                        html.Hr(),
                        dcc.Graph(id="unscheduld_gantt_chart"),
                    ],
                ),
                html.Div(
                    id="gantt_chart_card",
                    children=[
                        html.B("Optimized Schedule"),
                        html.Hr(),
                        dcc.Graph(id="gantt_chart"),
                        dcc.Loading(
                            parent_className='gantt_chart',
                            id="loading-1",
                            children=[html.Div(id="loading-output-1")],
                            type="default",
                        ),
                    ],
                ),
            ],
        ),
    ],
)


app.clientside_callback(
    ClientsideFunction(namespace="clientside", function_name="resize"),
    Output("output-clientside", "children"),
    [Input("wait_time_table", "children")],
)


# Run the server
if __name__ == "__main__":
    app.run_server(debug=True)