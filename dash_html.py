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

from dash import dcc, html

from app_configs import (
    CLASSICAL_TAB_LABEL,
    DESCRIPTION,
    DWAVE_TAB_LABEL,
    MAIN_HEADER,
    SCENARIOS,
    SOLVER_TIME,
    THUMBNAIL
)

MODEL_OPTIONS = ["Mixed Integer Model", "Mixed Integer Quadratic Model"]
SOLVER_OPTIONS = ["D-Wave Hybrid Solver", "Classical Solver (COIN-OR Branch-and-Cut)"]


def description_card():
    """A Div containing dashboard title & descriptions."""
    return html.Div(
        id="description-card",
        children=[html.H1(MAIN_HEADER), html.P(DESCRIPTION)],
    )


def dropdown(label: str, id: str, options: list) -> html.Div:
    """Slider element for value selection."""
    return html.Div(
        children=[
            html.Label(label),
            dcc.Dropdown(
                id=id,
                options=options,
                value=options[0]["value"],
                clearable=False,
                searchable=False,
            ),
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
    model_options = [
        {"label": model_option, "value": i} for i, model_option in enumerate(MODEL_OPTIONS)
    ]
    solver_options = [
        {"label": solver_value, "value": i} for i, solver_value in enumerate(SOLVER_OPTIONS)
    ]

    return html.Div(
        id="control-card",
        children=[
            dropdown(
                "Scenario (jobs x resources)",
                "scenario-select",
                scenario_options,
            ),
            dropdown(
                "Model",
                "model-select",
                model_options,
            ),
            html.Label("Solver"),
            dcc.Checklist(
                id="solver-select",
                options=solver_options,
                value=[solver_options[0]["value"]],
            ),
            html.Label("Solver Time Limit (seconds)"),
            dcc.Input(
                id="solver-time-limit",
                type="number",
                **SOLVER_TIME,
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
            html.Div(id="banner", children=[html.Img(src=THUMBNAIL)]),
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
                                        label="Input",
                                        value="input-tab",
                                        className="tab",
                                        children=[
                                            html.Div(
                                                html.Div(
                                                    id="unscheduled-gantt-chart-card",
                                                    className="gantt-div",
                                                    children=[
                                                        html.H3(
                                                            "Unscheduled Jobs and Resources",
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
                                        label=DWAVE_TAB_LABEL,
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
                                                            "D-Wave Hybrid Solver",
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
                                        label=CLASSICAL_TAB_LABEL,
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
                                                            "Classical Solver (COIN-OR Branch-and-Cut)",
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
