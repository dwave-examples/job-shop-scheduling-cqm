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

from __future__ import annotations

import dash
from dash import dcc, html

from app_configs import HTML_CONFIGS, SCENARIOS

MODEL_OPTIONS = ["Mixed Integer Model", "Mixed Integer Quadratic Model"]
SOLVER_OPTIONS = ["D-Wave Hybrid Solver", "Classical Solver (COIN-OR Branch-and-Cut)"]


def description_card():
    """A Div containing dashboard title & descriptions."""
    return html.Div(
        id="description-card",
        children=[
            html.H1(HTML_CONFIGS["main_header"]),
            html.P(children=HTML_CONFIGS["welcome_instructions"]),
        ],
    )


def generate_control_card() -> html.Div:
    """Generates the control card for the dashboard.

    Contains the dropdowns for selecting the scenario, model, and solver.

    Returns:
        html.Div: A Div containing the dropdowns for selecting the scenario,
        model, and solver.
    """

    scenario_options = [{"label": scenario, "value": scenario} for scenario in SCENARIOS]
    model_options = [{"label": model_option, "value": i} for i, model_option in enumerate(MODEL_OPTIONS)]
    solver_options = [{"label": solver_value, "value": i} for i, solver_value in enumerate(SOLVER_OPTIONS)]

    return html.Div(
        id="control-card",
        children=[
            html.Label("Scenario (jobs x resources)"),
            dcc.Dropdown(
                id="scenario-select",
                options=scenario_options,
                value=scenario_options[0]["value"],
                clearable=False,
                searchable=False,
            ),
            html.Label("Model"),
            dcc.Dropdown(
                id="model-select",
                options=model_options,
                value=model_options[0]["value"],
                clearable=False,
                searchable=False,
            ),
            html.Label("Solver"),
            dcc.Checklist(
                id="solver-select",
                options=solver_options,
                value=[solver_options[0]["value"]],
            ),
            html.Label("Solver Time Limit"),
            dcc.Input(
                id="solver-time-limit",
                type="number",
                value=HTML_CONFIGS["solver_options"]["default_time_seconds"],
                min=HTML_CONFIGS["solver_options"]["min_time_seconds"],
                max=HTML_CONFIGS["solver_options"]["max_time_seconds"],
                step=HTML_CONFIGS["solver_options"]["time_step_seconds"],
            ),
            html.Div(
                id="button-group",
                children=[
                    html.Button(id="run-button", children="Run Optimization", n_clicks=0),
                    html.Button(
                        id="cancel-button",
                        children="Cancel Optimization",
                        n_clicks=0,
                        className="display-none",
                    ),
                ],
            ),
        ],
    )


# set the application HTML
def set_html(app):
    app.layout = html.Div(
        id="app-container",
        children=[
            dcc.Store("last-selected-solvers"),
            dcc.Store("running-dwave"),
            dcc.Store("running-classical"),
            # Banner
            html.Div(
                id="banner",
                className="banner",
                children=[html.Img(src="assets/dwave_logo.svg")],
            ),
            html.Div(
                id="columns",
                children=[
                    # Left column
                    html.Div(
                        id="left-column",
                        children=[
                            html.Div(
                                [  # Fixed width Div to collapse
                                    html.Div(
                                        [  # Padding and content wrapper
                                            description_card(),
                                            generate_control_card(),
                                            html.Div(
                                                ["initial child"],
                                                id="output-clientside",
                                                style={"display": "none"},
                                            ),
                                        ]
                                    )
                                ]
                            ),
                            html.Div(
                                html.Button(id="left-column-collapse", children=[html.Div()]),
                            ),
                        ],
                    ),
                    # Right column
                    html.Div(
                        id="right-column",
                        children=[
                            dcc.Tabs(
                                id="tabs",
                                value="input-tab",
                                children=[
                                    dcc.Tab(
                                        label=HTML_CONFIGS["tabs"]["input"]["name"],
                                        value="input-tab",
                                        className="tab",
                                        children=[
                                            html.Div(
                                                html.Div(
                                                    id="unscheduled-gantt-chart-card",
                                                    className="gantt-div",
                                                    children=[
                                                        html.H3(
                                                            HTML_CONFIGS["tabs"]["input"]["header"],
                                                            className="gantt-title",
                                                        ),
                                                        dcc.Loading(
                                                            id="loading-icon-input",
                                                            children=[
                                                                dcc.Graph(
                                                                    id="unscheduled-gantt-chart",
                                                                    responsive=True,
                                                                ),
                                                            ],
                                                        ),
                                                    ],
                                                )
                                            )
                                        ],
                                    ),
                                    dcc.Tab(
                                        label=HTML_CONFIGS["tabs"]["dwave"]["name"],
                                        value="dwave-tab",
                                        id="dwave-tab",
                                        className="tab",
                                        disabled=True,
                                        children=[
                                            html.Div(
                                                html.Div(
                                                    id="optimized-gantt-chart-card",
                                                    className="gantt-div",
                                                    children=[
                                                        html.H3(
                                                            HTML_CONFIGS["tabs"]["dwave"]["header"],
                                                            className="gantt-title",
                                                        ),
                                                        dcc.Graph(
                                                            id="optimized-gantt-chart",
                                                            responsive=True,
                                                        ),
                                                        dcc.Graph(id="dwave-summary-table"),
                                                    ],
                                                )
                                            )
                                        ],
                                    ),
                                    dcc.Tab(
                                        label=HTML_CONFIGS["tabs"]["classical"]["name"],
                                        id="mip-tab",
                                        className="tab",
                                        value="mip-tab",
                                        disabled=True,
                                        children=[
                                            html.Div(
                                                html.Div(
                                                    id="mip-gantt-chart-card",
                                                    className="gantt-div",
                                                    children=[
                                                        html.H3(
                                                            HTML_CONFIGS["tabs"]["classical"][
                                                                "header"
                                                            ],
                                                            className="gantt-title",
                                                        ),
                                                        dcc.Graph(
                                                            id="mip-gantt-chart", responsive=True
                                                        ),
                                                        dcc.Graph(id="mip-summary-table"),
                                                    ],
                                                )
                                            )
                                        ],
                                    ),
                                ],
                            )
                        ],
                    ),
                ],
            ),
        ],
    )