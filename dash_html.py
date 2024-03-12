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

import dash
from dash import dcc, html
from dash.dependencies import Input

from app_configs import SCENARIOS, MODEL_OPTIONS, SOLVER_OPTIONS, HTML_CONFIGS



def description_card():
    """
    :return: A Div containing dashboard title & descriptions.
    """
    return html.Div(
        id="description-card",
        children=[
            html.H5(HTML_CONFIGS['main_header']),
            html.H3(HTML_CONFIGS['welcome_message']),
            html.Div(
                id="intro",
                children=HTML_CONFIGS['welcome_instructions'],
            ),
        ],
    )

def generate_control_card() -> html.Div:
    """
    This function generates the control card for the dashboard, which
    contains the dropdowns for selecting the scenario, model, and solver.

    Returns:
        html.Div: A Div containing the dropdowns for selecting the scenario,
        model, and solver.
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
            value=HTML_CONFIGS['solver_options']['default_time_seconds'],
            min=HTML_CONFIGS['solver_options']['min_time_seconds'],
            max=HTML_CONFIGS['solver_options']['max_time_seconds'],
            step=HTML_CONFIGS['solver_options']['time_step_seconds'],
            ),
            html.Div(
                id="button-group",
                children=[
                    html.Button(id="run-button", children="Run Optimization", className='run-button', n_clicks=0, disabled=False),
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

#set the application HTML
def set_html(app):
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
                        dcc.Tab(label=HTML_CONFIGS['tabs']['input']['name'],
                                value='input_tab',
                                className='tab',
                                children=[html.Div(
                                    html.Div(
                                    id="unscheduled_gantt_chart_card",
                                    className="gantt-div",
                                    children=[
                                        html.B(HTML_CONFIGS['tabs']['input']['header'], className="gantt-title"),
                                        html.Hr(className="gantt-hr"),
                                        dcc.Loading(id = "loading-icon-input", children=[ 
                                            dcc.Graph(id="unscheduld_gantt_chart"),
                                        ],
                                    )])
                                )])
                            ,
                            dcc.Tab(label=HTML_CONFIGS['tabs']['dwave']['name'],
                                    value='dwave_tab',
                                    id='dwave_tab',
                                    className='tab',
                                    children=[html.Div(
                                        html.Div(
                                            id="optimized_gantt_chart_card",
                                            className="gantt-div",
                                            children=[
                                                html.B(HTML_CONFIGS['tabs']['dwave']['header'], className="gantt-title"),
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
                            dcc.Tab(label=HTML_CONFIGS['tabs']['classical']['name'],
                                    id='mip_tab',
                                    className='tab',
                                    value='mip_tab', 
                                    children=[html.Div(
                                        html.Div(
                                            id="mip_gantt_chart_card",
                                            className="gantt-div",
                                            children=[
                                                html.B(HTML_CONFIGS['tabs']['classical']['header'], className="gantt-title"),
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
