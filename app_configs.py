# Copyright 2024 D-Wave Systems Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file stores input parameters for the app."""

import json

# Sets Dash debug which hides and shows Dash debug menu.
# Set to True if developing and False if demoing.
# App should be restarted to see change.
DEBUG = False

# THEME_COLOR is used for the button, text, and banner and should be dark
# and pass accessibility checks with white: https://webaim.org/resources/contrastchecker/
# THEME_COLOR_SECONDARY can be light or dark and is used for sliders, loading icon, and tabs
THEME_COLOR = "#074C91"  # D-Wave dark blue default #074C91
THEME_COLOR_SECONDARY = "#2A7DE1"  # D-Wave blue default #2A7DE1

THUMBNAIL = "assets/dwave_logo.svg"

APP_TITLE = "JSS Demo"
MAIN_HEADER = "Job Shop Scheduling"
DESCRIPTION = """\
Run the job shop scheduling problem for several different scenarios.
Explore the Gantt Chart for solution details.
"""

CLASSICAL_TAB_LABEL = "Classical Results"
DWAVE_TAB_LABEL = "D-Wave Results"

# The list of scenarios that the user can choose from in the app.
# These can be found in the 'input' directory.
SCENARIOS = {
    "3x3": "instance3_3.txt",
    "5x5": "instance5_5.txt",
    "10x10": "instance10_10.txt",
    "15x15": "taillard15_15.txt",
    "20x15": "instance20_15.txt",
    "20x25": "instance20_25.txt",
    "30x30": "instance30_30.txt",
}

# solver time limits in seconds (value means default)
SOLVER_TIME = {
    "min": 5,
    "max": 300,
    "step": 5,
    "value": 5,
}

# The list of resources that the user can choose from in the app
RESOURCE_NAMES = json.load(open("./src/data/resource_names.json", "r"))["industrial"]
