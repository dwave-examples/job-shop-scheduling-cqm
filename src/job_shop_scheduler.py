"""
This module contains the JobShopSchedulingCQM class, which is used to build and 
solve a Job Shop Scheduling problem using CQM.

"""

import argparse
import sys
from time import time

import pandas as pd
from tabulate import tabulate
from job_shop_formulation_cqm import JobShopSchedulingCQM
from job_shop_formulation_nl import JobShopSchedulingNL

sys.path.append("./src")
import utils.plot_schedule as job_plotter
from model_data import JobShopData
from utils.greedy import GreedyJobShop
from utils.utils import write_solution_to_file, is_valid_schedule


def generate_greedy_makespan(job_data: JobShopData, num_samples: int = 100) -> int:
    """This function generates random samples using the greedy algorithm; it will keep the
    top keep_pct percent of samples.

    Args:
        job_data (JobShopData): An instance of the JobShopData class
        num_samples (int, optional): The number of samples to take (number of times
            the GreedyJobShop algorithm is run). Defaults to 100.

    Returns:
        int: The best makespan found by the greedy algorithm.
    """
    solutions = []
    for _ in range(num_samples):
        greedy = GreedyJobShop(job_data)
        task_assignments = greedy.solve()
        solutions.append(max([v[1] for v in task_assignments.values()]))
    best_greedy = min(solutions)

    return best_greedy


def solution_as_dataframe(solution) -> pd.DataFrame:
        """This function returns the solution as a pandas DataFrame
            Args:
            solution: The solution to the problem
        Returns:
            pd.DataFrame: A pandas DataFrame containing the solution
        """

        df_rows = []
        for (j, i), (start, dur) in solution.items():
            df_rows.append([j, start, start + dur, i])
        df = pd.DataFrame(df_rows, columns=["Job", "Start", "Finish", "Resource"])
        return df


def run_shop_scheduler(
    job_data: JobShopData,
    solver_time_limit: int = 60,
    verbose: bool = False,
    out_sol_file: str = None,
    out_plot_file: str = None,
    profile: str = None,
    max_makespan: int = None,
    greedy_multiplier: float = 1.4,
    solver_name: str = "NL") -> pd.DataFrame:
    """This function runs the job shop scheduler on the given data.

    Args:
        job_data (JobShopData): A JobShopData object that holds the data for this job shop
            scheduling problem.
        solver_time_limit (int, optional): Upperbound on how long the schedule can be; leave empty to
            auto-calculate an appropriate value. Defaults to None.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
        out_sol_file (str, optional): Path to the output solution file. Defaults to None.
        out_plot_file (str, optional): Path to the output plot file. Defaults to None.
        profile (str, optional): The profile variable to pass to the Sampler. Defaults to None.
        max_makespan (int, optional): Upperbound on how long the schedule can be; leave empty to
            auto-calculate an appropriate value. Defaults to None.
            If None, the makespan will be set to a value that is greedy_mulitiplier
            times the makespan found by the greedy algorithm.
        greedy_multiplier (float, optional): The multiplier to apply to the greedy makespan,
            to get the upperbound on the makespan. Defaults to 1.4.
        solver_name: name of the solver either CQM, NL or MIP

    Returns:
        pd.DataFrame: A DataFrame that has the following columns: Task, Start, Finish, and
        Resource.

    """
    ordered_tasks = job_data.get_ordered_tasks()

    if max_makespan is None:
        best_greedy_span = generate_greedy_makespan(job_data)
        max_makespan = int(best_greedy_span * greedy_multiplier)

    if solver_name == "NL":
        model = JobShopSchedulingNL(data=job_data, max_makespan=max_makespan, verbose=verbose)
    elif solver_name in ["CQM", "MIP"]:
        model = JobShopSchedulingCQM(data=job_data, max_makespan=max_makespan, solver_name=solver_name, verbose=verbose)
    else:
        raise ValueError(f'solver_name not defined')

    model_building_start = time()
    model.make()
    model_building_time = time() - model_building_start

    solver_start_time = time()
    model.solve(time_limit=solver_time_limit, profile=profile)
    solver_time = time() - solver_start_time
    model.compute_results() 

    if not is_valid_schedule(model.solution, ordered_tasks):
        print("Solution is not valid")
    
    if verbose:
        print(" \n" + "=" * 55 + "SOLUTION RESULTS" + "=" * 55)
        print(
            tabulate(
                [
                    [
                        "Completion Time",
                        "Max Make-Span",
                        "best_greedy_span",
                        "Model Building Time (s)",
                        "Solver Call Time (s)",
                        "Total Runtime (s)",
                    ],
                    [
                        model.completion_time,
                        max_makespan,
                        best_greedy_span,
                        int(model_building_time),
                        int(solver_time),
                        int(solver_time + model_building_time),
                    ],
                ],
                headers="firstrow",
            )
        )

    # Write solution to a file.
    if out_sol_file is not None:
        write_solution_to_file(job_data, model.solution, model.completion_time, out_sol_file)

    # Plot solution
    if out_plot_file is not None:
        job_plotter.plot_solution(job_data, model.solution, out_plot_file)

    df = solution_as_dataframe(model.solution)
    return df


if __name__ == "__main__":
    """Modeling and solving Job Shop Scheduling using CQM solver."""

    # Instantiate the parser
    parser = argparse.ArgumentParser(
        description="Job Shop Scheduling Using LeapHybridCQMSampler",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-i",
        "--instance",
        type=str,
        help="path to the input instance file; ",
        default="input/instance3_3.txt",
    )

    parser.add_argument("-tl", "--time_limit", type=int, help="time limit in seconds", default=10)

    parser.add_argument(
        "-os",
        "--output_solution",
        type=str,
        help="path to the output solution file",
        default="output/solution.txt",
    )

    parser.add_argument(
        "-op",
        "--output_plot",
        type=str,
        help="path to the output plot file",
        default="output/schedule.png",
    )

    parser.add_argument(
        "-s",
        "--solver_name",
        type=str,
        help="Define name of the solver, NL, CQM or MIP",

        default="NL"
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", default=True, help="Whether to print verbose output"
    )

    parser.add_argument(
        "-p",
        "--profile",
        type=str,
        help="The profile variable to pass to the Sampler. Defaults to None.",
        default=None,
    )

    parser.add_argument(
        "-mm",
        "--max_makespan",
        type=int,
        help="Upperbound on how long the schedule can be; leave empty to auto-calculate an appropriate value.",
        default=None,
    )

    # Parse input arguments.
    args = parser.parse_args()
    input_file = args.instance
    time_limit = args.time_limit
    out_plot_file = args.output_plot
    out_sol_file = args.output_solution
    max_makespan = args.max_makespan
    profile = args.profile
    solver_name = args.solver_name
    verbose = args.verbose

    job_data = JobShopData()
    job_data.load_from_file(input_file)

    results = run_shop_scheduler(
        job_data,
        time_limit,
        verbose=verbose,
        profile=profile,
        max_makespan=max_makespan,
        out_sol_file=out_sol_file,
        out_plot_file=out_plot_file,
        solver_name=solver_name
    )
