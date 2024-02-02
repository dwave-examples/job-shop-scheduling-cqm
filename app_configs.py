'''
This file stores input parameters for the app.
'''

import json

HTML_CONFIGS = {
    'title': 'Job Shop Scheduling Demo',
    'main_header': "Job Shop Scheduling",
    'welcome_message': "Welcome to the Job Shop Scheduling Dashboard",
    'welcome_instructions': "Run the job shop scheduling problem for several different scenarios. Explore the Gantt Chart for solution details",
    "solver_options": {
        "min_time_seconds": 5,
        "max_time_seconds": 300,
        "time_step_seconds": 5,
        "default_time_seconds": 5
    },
    "solver_messages": {
        "mip": {
            "quadratic_error": "Unable to run MIP solver with quadratic constraints",
            "no_solution": "No solution found for MIP solver",
            "solver_not_chosen": "Select COIN-OR Branch and Cut Solver to run this solver"
            },
        'dwave': {
            "no_solution": "No solution found for D-Wave solver",
            "solver_not_chosen": "Select D-Wave Hybrid Solver to run this solver"
            }
    },
    "tabs": {
        "input": {
            "name": "Input",
            "header": "Jobs to be Scheduled",
        },
        "classical": {
            "name": "Classical",
            "header": "COIN-OR Branch-and-Cut Solver",
        },
        "dwave": {
            "name": "D-Wave",
            "header": "D-Wave Hybrid Solver",
        }, 
    }
}


# The list of scenarios that the user can choose from in the app
SCENARIOS = {
    '3x3': "instance3_3.txt",
    '5x5': "instance5_5.txt",
    '10x10': "instance10_10.txt",
    '15x15': "taillard15_15.txt",
    '20x15': "instance20_15.txt",
    '20x25': "instance20_25.txt",
    '30x30': "instance30_30.txt"
}

# The list of models that the user can choose from in the app
MODEL_OPTIONS = {
    "Mixed Integer Model": "MIP",
    "Mixed Integer Quadratic Model": "QM",
}

# The list of solvers that the user can choose from in the app
SOLVER_OPTIONS = {
    "D-Wave Hybrid Solver": "Hybrid",
    "COIN-OR Branch-and-Cut Solver (CBC)": "MIP"
}

# The list of resources that the user can choose from in the app
RESOURCE_NAMES = json.load(open('./src/data/resource_names.json', 'r'))['industrial']