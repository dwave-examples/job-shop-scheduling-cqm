"""
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

"""

import pathlib

import diskcache
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from dash import DiskcacheManager

cache = diskcache.Cache("./cache")
background_callback_manager = DiskcacheManager(cache)

from app_configs import RESOURCE_NAMES, SCENARIOS
from src.model_data import JobShopData

BASE_PATH = pathlib.Path(__file__).parent.resolve()
DATA_PATH = BASE_PATH.joinpath("input").resolve()

model_data = JobShopData()


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
            task_data.append(
                {
                    "Job": task.job,
                    "Start": start_time,
                    "Finish": end_time,
                    "Resource": task.resource,
                }
            )
            start_time = end_time
    df = pd.DataFrame(task_data)
    df["delta"] = df.Finish - df.Start
    df["Job"] = df["Job"].astype(str)
    df["Resource"] = df["Resource"].astype(str)
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
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[
            {
                "text": message,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 28},
            }
        ],
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=10, b=10),
    )
    return fig


def generate_gantt_chart(
    scenario: str = None, df: pd.DataFrame = None, y_axis: str = "Job", color: str = "Resource"
) -> go.Figure:
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
        if "json" in filename:
            model_data.load_from_json(DATA_PATH.joinpath(filename), resource_names=RESOURCE_NAMES)
        else:
            model_data.load_from_file(DATA_PATH.joinpath(filename), resource_names=RESOURCE_NAMES)
        df = get_minimum_task_times(model_data)
    if y_axis == "Job":
        if df["Job"].dtype == "object":
            df["JobInt"] = df["Job"].str.replace("Job", "").astype(int)
        else:
            df["JobInt"] = df["Job"]
        df = df.sort_values(by=["JobInt", color, "Start"])
        df = df.drop(columns=["JobInt"])
    else:
        df = df.sort_values(by=[y_axis, color, "Start"])
    df["delta"] = df.Finish - df.Start
    df[color] = df[color].astype(str)
    df[y_axis] = df[y_axis].astype(str)
    num_items = len(df[color].unique())
    colorscale = "Agsunset"
    fig = px.timeline(
        df,
        x_start="Start",
        x_end="Finish",
        y=y_axis,
        color=color,
        color_discrete_sequence=px.colors.sample_colorscale(
            colorscale, [n / (num_items - 1) for n in range(num_items)]
        ),
    )

    for idx, _ in enumerate(fig.data):
        resource = fig.data[idx].name
        data_list = []
        for job in fig.data[idx].y:
            try:
                data_list.append(
                    df[(df[y_axis] == job) & (df[color] == resource)].delta.tolist()[0]
                )
            except:
                continue
        fig.data[idx].x = data_list

    fig.layout.xaxis.type = "linear"
    fig.update_layout(
        margin=dict(l=20, r=20, t=10, b=10),
        xaxis_title="Time Period",
    )
    return fig


def generate_output_table(make_span: int, solver_time_limit: int, total_time: int) -> go.Figure:
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
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(values=["Make-span", "Solver Time Limit", "Total Time"]),
                cells=dict(values=[[make_span], [solver_time_limit], [total_time]]),
            )
        ]
    )
    fig.update_layout(margin=dict(l=20, r=20, t=10, b=10), height=100, autosize=False)
    return fig
