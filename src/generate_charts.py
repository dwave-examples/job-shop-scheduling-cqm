#    Copyright 2024 D-Wave Systems Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import pathlib

import diskcache
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from dash import DiskcacheManager

cache = diskcache.Cache("./cache")
background_callback_manager = DiskcacheManager(cache)

from src.model_data import JobShopData

Y_AXIS_LABEL = "Job"
COLOR_LABEL = "Resource"


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
                    Y_AXIS_LABEL: task.job,
                    "Start": start_time,
                    "Finish": end_time,
                    COLOR_LABEL: task.resource,
                }
            )
            start_time = end_time
    df = pd.DataFrame(task_data)
    df["delta"] = df.Finish - df.Start
    df[Y_AXIS_LABEL] = df[Y_AXIS_LABEL].astype(str)
    df[COLOR_LABEL] = df[COLOR_LABEL].astype(str)
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
    df: pd.DataFrame = None
) -> go.Figure:
    """Generates a Gantt chart of the unscheduled tasks for the given scenario.

    Args:
        df (pd.DataFrame): A DataFrame containing the data to plot. If this is
            not None, then the scenario argument will be ignored.

    Returns:
        go.Figure: A Plotly figure object.
    """
    if df[Y_AXIS_LABEL].dtype == "object":
        df["JobInt"] = df[Y_AXIS_LABEL].str.replace(Y_AXIS_LABEL, "").astype(int)
    else:
        df["JobInt"] = df[Y_AXIS_LABEL]
    df = df.sort_values(by=["JobInt", COLOR_LABEL, "Start"])
    df = df.drop(columns=["JobInt"])

    df["delta"] = df.Finish - df.Start
    df[COLOR_LABEL] = df[COLOR_LABEL].astype(str)
    df[Y_AXIS_LABEL] = df[Y_AXIS_LABEL].astype(str)
    num_items = len(df[COLOR_LABEL].unique())
    colorscale = "Agsunset"

    fig = px.timeline(
        df,
        x_start="Start",
        x_end="Finish",
        y=Y_AXIS_LABEL,
        color=COLOR_LABEL,
        color_discrete_sequence=px.colors.sample_colorscale(
            colorscale, [n / (num_items - 1) for n in range(num_items)]
        ),
    )

    for index, data in enumerate(fig.data):
        resource = data.name
        fig.data[index].x = [
            df[(df[Y_AXIS_LABEL] == job) & (df[COLOR_LABEL] == resource)].delta.tolist()[0] for job in data.y
        ]

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
                cells=dict(values=[[make_span], [f"{solver_time_limit}s"], [f"{total_time:.2f}s"]]),
            )
        ]
    )
    fig.update_layout(margin=dict(l=20, r=20, t=10, b=10), height=100, autosize=False)
    return fig
