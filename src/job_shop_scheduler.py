from time import time
import warnings

from tabulate import tabulate
import argparse
from dimod import ConstrainedQuadraticModel, Binary, Integer, SampleSet
from dwave.system import LeapHybridCQMSampler
import pandas as pd

import sys
sys.path.append('./src')
from utils.utils import print_cqm_stats, write_solution_to_file
import utils.plot_schedule as job_plotter
import utils.mip_solver as mip_solver
from model_data import JobShopData



class JobShopSchedulingCQM():
    """Builds and solves a Job Shop Scheduling problem using CQM."""
    def __init__(self, model_data: JobShopData, max_makespan: int = None):
        self.model_data = model_data
        self.cqm = None
        self.x = {}
        self.y = {}
        self.makespan = {}
        self.best_sample = {}
        self.solution = {}
        self.completion_time = 0
        self.max_makespan = max_makespan
        if self.max_makespan is None:
            self.max_makespan = model_data.get_max_makespan()


    def define_cqm_model(self):
        """Define CQM model."""
        self.cqm = ConstrainedQuadraticModel()


    def define_variables(self, model_data: JobShopData):
        """Define CQM variables.

        Args:
            model_data: a JobShopData data class
        """
        # Define make span as an integer variable
        self.makespan = Integer("makespan", lower_bound=0,
                                upper_bound=self.max_makespan)

        # Define integer variable for start time of using machine i for job j
        self.x = {
            (j, i): Integer('x{}_{}'.format(j, i), lower_bound=0,
                            upper_bound=self.max_makespan)
            for j in model_data.jobs for i in model_data.resources}

        # Add binary variable which equals to 1 if job j precedes job k on
        # machine i
        self.y = {(j, k, i): Binary('y{}_{}_{}'.format(j, k, i))
                  for j in model_data.jobs
                  for k in model_data.jobs 
                  for i in model_data.resources}


    def define_objective_function(self) -> None:
        """Define objective function, which is to minimize
        the makespan of the schedule."""
        self.cqm.set_objective(self.makespan)


    def add_precedence_constraints(self, model_data: JobShopData) -> None:
        """Precedence constraints ensures that all operations of a job are
        executed in the given order.

        Args:
            model_data: a JobShopData data class
        """

        for job in model_data.jobs:  # job
            for prev_task, curr_task in zip(model_data.job_tasks[job][:-1], model_data.job_tasks[job][1:]):
                machine_curr = curr_task.resource
                machine_prev = prev_task.resource
                self.cqm.add_constraint(self.x[(job, machine_curr)] -
                                        self.x[(job, machine_prev)]
                                        >= prev_task.duration,
                                        label='pj{}_m{}'.format(job, machine_curr))


    def add_quadratic_overlap_constraint(self, model_data: JobShopData) -> None:
        """Add quadratic constraints to ensure that no two jobs can be scheduled
         on the same machine at the same time.

         Args:
             model_data: a JobShopData data class
        """
        for j in model_data.jobs:
            for k in model_data.jobs:
                if j < k:
                    for i in model_data.resources:
                        task_k = model_data.get_resource_job_tasks(job=k, resource=i)
                        task_j = model_data.get_resource_job_tasks(job=j, resource=i)

                        if task_k.duration > 0 and task_j.duration > 0:
                            self.cqm.add_constraint(
                                self.x[(j, i)] - self.x[(k, i)] + (
                                        task_k.duration - task_j.duration) \
                                         * self.y[(j, k, i)] + 2 * self.y[(j, k, i)] * (
                                        self.x[(k, i)] - self.x[(j, i)]) >=
                                        task_k.duration,
                                label='OneJobj{}_j{}_m{}'.format(j, k, i))
                            

    def add_disjunctive_constraints(self, model_data: JobShopData) -> None:
        """This function adds the disjunctive constraints the prevent two jobs
        from being scheduled on the same machine at the same time. This is a
        non-quadratic alternative to the quadratic overlap constraint.

        Args:
            model_data (JobShopData): The data for the job shop scheduling
        """        
        V = self.max_makespan
        for j in model_data.jobs:
            for k in model_data.jobs:
                if j < k:
                    for i in model_data.resources:
                        task_k = model_data.get_resource_job_tasks(job=k, resource=i)
                        self.cqm.add_constraint(
                            self.x[(j, i)] - self.x[(k, i)] - task_k.duration + self.y[(j, k, i)] * V >= 0,
                            label='disjunction1{}_j{}_m{}'.format(j, k, i))
                        
                        task_j = model_data.get_resource_job_tasks(job=j, resource=i)
                        self.cqm.add_constraint(
                            self.x[(k, i)] - self.x[(j, i)] - task_j.duration + (1-self.y[(j, k, i)]) * V >= 0,
                            label='disjunction2{}_j{}_m{}'.format(j, k, i))


    def add_makespan_constraint(self, model_data: JobShopData) -> None:
        """Ensures that the make span is at least the largest completion time of
        the last operation of all jobs.

        Args:
            model_data: a JobShopData data class
        """
        for job in model_data.jobs:
            last_job_task = model_data.job_tasks[job][-1]
            last_machine = last_job_task.resource
            self.cqm.add_constraint(
                self.makespan - self.x[(job, last_machine)] >= last_job_task.duration,
                label='makespan_ctr{}'.format(job))


    def call_cqm_solver(self, time_limit: int, model_data: JobShopData, profile: str) -> None:
        """Calls CQM solver.

        Args:
            time_limit (int): time limit in second
            model_data (JobShopData): a JobShopData data class
            profile (str): The profile variable to pass to the Sampler. Defaults to None.
            See documentation at 
            https://docs.ocean.dwavesys.com/en/stable/docs_cloud/reference/generated/dwave.cloud.config.load_config.html#dwave.cloud.config.load_config
        """
        sampler = LeapHybridCQMSampler(profile=profile)
        raw_sampleset = sampler.sample_cqm(self.cqm, time_limit=time_limit, label='Job Shop Demo')
        self.feasible_sampleset = raw_sampleset.filter(lambda d: d.is_feasible)
        num_feasible = len(self.feasible_sampleset)
        if num_feasible > 0:
            best_samples = \
                self.feasible_sampleset.truncate(min(10, num_feasible))
        else:
            warnings.warn("Warning: Did not find feasible solution")
            best_samples = raw_sampleset.truncate(10)

        print(" \n" + "=" * 30 + "BEST SAMPLE SET" + "=" * 30)
        print(best_samples)

        self.best_sample = best_samples.first.sample

        self.solution = {
            (j, i): (model_data.get_resource_job_tasks(job=j, resource=i),
                     self.best_sample[self.x[(j, i)].variables[0]],
                     model_data.get_resource_job_tasks(job=j, resource=i).duration)
            for i in model_data.resources for j in model_data.jobs}

        self.completion_time = self.best_sample['makespan']


    def call_mip_solver(self, time_limit: int=100) -> SampleSet:
        """This function calls the MIP solver and returns the solution

        Args:
            time_limit (int, optional): The maximum amount of time to
            allow the MIP solver to before returning. Defaults to 100.

        Returns:
            SampleSet: The solution to the problem from the MIP solver,
                or an empty SampleSet if no solution was found.
        """        
        solver = mip_solver.MIPCQMSolver()
        sol = solver.sample_cqm(cqm=self.cqm, time_limit=time_limit)
        self.solution = {}
        if len(sol) == 0:
            warnings.warn("Warning: Did not find feasible solution")
            return self.solution
        best_sol = sol.first.sample
        
        for (var, val) in best_sol.items():

            if var.startswith('x'):
                job, machine = var[1:].split('_')
                task = self.model_data.get_resource_job_tasks(job=job, resource=machine)
                self.solution[(job, machine)] = task, val, task.duration

    
    def solution_as_dataframe(self) -> pd.DataFrame:
        """This function returns the solution as a pandas DataFrame

        Returns:
            pd.DataFrame: A pandas DataFrame containing the solution
        """        
        df_rows = []
        for (j, i), (task, start, dur) in self.solution.items():
            df_rows.append([j, task, start, start + dur, i])
        df = pd.DataFrame(df_rows, columns=['Job', 'Task', 'Start', 'Finish', 'Resource'])
        return df
    

def run_shop_scheduler(
    job_data: JobShopData,
    solver_time_limit: int = 60,
    use_mip_solver: bool = False,
    verbose: bool = False,
    allow_quadratic_constraints: bool = True,
    out_sol_file: str = None,
    out_plot_file: str = None,
    profile: str = None,
    max_makespan: int = None
    ) -> pd.DataFrame:
    """This function runs the job shop scheduler on the given data.

    Args:
        job_data (JobShopData): A JobShopData object that holds the data for this job shop 
            scheduling problem.
        solver_time_limit (int, optional): Upperbound on how long the schedule can be; leave empty to 
            auto-calculate an appropriate value. Defaults to None.
        use_mip_solver (bool, optional): Whether to use the MIP solver instead of the CQM solver.
            Defaults to False.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
        allow_quadratic_constraints (bool, optional): Whether to allow quadratic constraints. 
            Defaults to True.
        out_sol_file (str, optional): Path to the output solution file. Defaults to None.
        out_plot_file (str, optional): Path to the output plot file. Defaults to None.
        profile (str, optional): The profile variable to pass to the Sampler. Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame that has the following columns: Task, Start, Finish, and
        Resource.

    """    
    if allow_quadratic_constraints and use_mip_solver:
        raise ValueError("Cannot use quadratic constraints with MIP solver")
    model_building_start = time()
    model = JobShopSchedulingCQM(model_data=job_data, max_makespan=max_makespan)
    model.define_cqm_model()
    model.define_variables(job_data)
    model.add_precedence_constraints(job_data)
    if allow_quadratic_constraints:
        model.add_quadratic_overlap_constraint(job_data)
    else:
        model.add_disjunctive_constraints(job_data)
    model.add_makespan_constraint(job_data)
    model.define_objective_function()

    if verbose:
        print_cqm_stats(model.cqm)
    model_building_time = time() - model_building_start
    solver_start_time = time()
    if use_mip_solver:
        sol = model.call_mip_solver(time_limit=solver_time_limit)
    else:
        model.call_cqm_solver(time_limit=solver_time_limit, model_data=job_data, profile=profile)
        sol = model.best_sample
    solver_time = time() - solver_start_time

    if verbose:
        print(" \n" + "=" * 55 + "SOLUTION RESULTS" + "=" * 55)
        print(tabulate([["Completion Time", "Max Possible Make-Span",
                        "Model Building Time (s)", "Solver Call Time (s)",
                        "Total Runtime (s)"],
                        [model.completion_time, 
                         model.max_makespan,
                         int(model_building_time), 
                         int(solver_time),
                         int(solver_time +  model_building_time)
                         ]],
                    headers="firstrow"))
    
    # Write solution to a file.
    if out_sol_file is not None:
        write_solution_to_file(
            job_data, model.solution, model.completion_time, out_sol_file)

    # Plot solution
    if out_plot_file is not None:
        job_plotter.plot_solution(job_data, model.solution, out_plot_file)

    df = model.solution_as_dataframe()
    return df



if __name__ == "__main__":
    """Modeling and solving Job Shop Scheduling using CQM solver."""

    # Start the timer
    start_time = time()

    # Instantiate the parser
    parser = argparse.ArgumentParser(
        description='Job Shop Scheduling Using LeapHybridCQMSampler',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-instance', type=str,
                        help='path to the input instance file; ',
                        default='input/instance5_5.txt')

    parser.add_argument('-tl', type=int,
                        help='time limit in seconds')

    parser.add_argument('-os', type=str,
                        help='path to the output solution file',
                        default='output/solution.txt')

    parser.add_argument('-op', type=str,
                        help='path to the output plot file',
                        default='output/schedule.png')
    
    parser.add_argument('-use_mip_solver', action='store_true',
                        help='Whether to use the MIP solver instead of the CQM solver')
    
    parser.add_argument('-verbose', action='store_true',
                        help='Whether to print verbose output')
    
    parser.add_argument('-allow_quad', action='store_true',
                        help='Whether to allow quadratic constraints')
    
    parser.add_argument('-profile', type=str,
                        help='The profile variable to pass to the Sampler. Defaults to None.',
                        default=None)
    
    parser.add_argument('-max_makespan', type=int,
                        help='Upperbound on how long the schedule can be; leave empty to auto-calculate an appropriate value.',
                        default=None)
    
    
    # Parse input arguments.
    args = parser.parse_args()
    input_file = args.instance
    time_limit = args.tl
    out_plot_file = args.op
    out_sol_file = args.os
    allow_quadratic_constraints = args.allow_quad

    job_data = JobShopData()
    job_data.load_from_file(input_file)

    results = run_shop_scheduler(job_data, time_limit, verbose=True, use_mip_solver=args.use_mip_solver,
                          allow_quadratic_constraints=allow_quadratic_constraints, profile=args.profile,
                          max_makespan=None)
