from dimod import ConstrainedQuadraticModel, Binary, Integer, SampleSet
from utils.utils import print_cqm_stats, write_solution_to_file
import utils.plotjobs as job_plotter
from data import Data
import matplotlib.pyplot as plt
from dwave.system import LeapHybridCQMSampler
from time import time
from tabulate import tabulate
from cqmsolver.hss_sampler import HSSCQMSampler

try:
    from cqmsolver.hss_sampler import HSSCQMSampler
except:
    pass
import argparse


class JSSCQM():
    """ Builds and Solve a Job Shop Scheduling problem using CQM.
    """

    def __init__(self):
        self.cqm = None
        self.x = {}
        self.y = {}
        self.make_span = {}
        self.best_feasible_sample = {}
        self.solution = {}
        self.completion_time = 0

    def define_cqm_model(self):
        """Define CQM model"""
        self.cqm = ConstrainedQuadraticModel()

    def define_variables(self, data):
        """Define CQM variables
        Args:
            data: a JSS data class
        """

        # Define make span as an integer variable
        self.make_span = Integer("make_span", lower_bound=0,
                                 upper_bound=data.MaxMakeSpan)

        # Define integer variable for start time of using machine i for job j
        self.x = {
            (j, i): Integer('x{}_{}'.format(j, i), lower_bound=0,
                            upper_bound=data.MaxMakeSpan)
            for j in range(data.num_jobs) for i in range(data.num_machines)}

        # Add binary variable which equals to 1 if job j precedes job k on
        # machine i
        self.y = {(j, k, i): Binary('y{}_{}_{}'.format(j, k, i))
                  for j in range(data.num_jobs)
                  for k in range(data.num_jobs) for i in
                  range(data.num_machines)}

    def define_objective_function(self):
        """Define objective function"""
        self.cqm.set_objective(self.make_span)

    def add_precedence_constraints(self, data):
        """Precedence constraints: It ensures that all operations of a job are
        executed in the given order.
        Args:
            data: a JSS data class
        """

        for j in range(data.num_jobs):  # job
            for t in range(1, data.num_machines):  # tasks
                machine_curr = data.task_machine[(j, t)]
                machine_prev = data.task_machine[(j, t - 1)]
                self.cqm.add_constraint(self.x[(j, machine_curr)] -
                                        self.x[(j, machine_prev)]
                                        >= data.task_duration[(j, t - 1)],
                                        label='pj{}_m{}'.format(j, t))

    def add_quadratic_one_job_only_cstr(self, data):
        """Add quadratic constraints to ensure that no two jobs can be scheduled
         on the same machine at the same time.
         Args:
             data: JSS data class
        """

        for j in range(data.num_jobs):
            for k in range(data.num_jobs):
                if j < k:
                    for i in range(data.num_machines):
                        task_k = data.machine_task[(k, i)]
                        task_j = data.machine_task[(j, i)]
                        self.cqm.add_constraint(
                            self.x[(j, i)] - self.x[(k, i)] +
                            (data.task_duration[k, task_k] - data.task_duration[
                                j, task_j]) * self.y[
                                (j, k, i)]
                            + 2 * self.y[(j, k, i)] * (
                                    self.x[(k, i)] - self.x[(j, i)]) >=
                            data.task_duration[(k, task_k)],
                            label='OneJobj{}_j{}_m{}'.format(j, k, i))

    def add_make_span_cstr(self, data):
        """ Ensures that the make span is at least the largest completion time of
        # the last operation of all jobs.
        Args:
            data: JSS data class
        """
        for j in range(data.num_jobs):
            last_machine = data.task_machine[(j, data.num_machines - 1)]
            self.cqm.add_constraint(
                self.make_span - self.x[(j, last_machine)] >=
                data.task_duration[(j, data.num_machines - 1)],
                label='make_span_ctr{}'.format(j))

    def call_cqm_solver(self, time_limit, data, solve_local=False):
        """Calls CQM solver
        Args:
            time_limit: time limit in second
            data: a JSS data class
            solve_local: if solve the Jss Model locally

        """
        if solve_local:
            sampler = HSSCQMSampler()
            raw_sampleset = sampler.sample(self.cqm, time_limit=time_limit)
        else:
            sampler = LeapHybridCQMSampler(profile='alpha')
            # solver='hybrid_constrained_quadratic_model_version1p_bulk_alpha')
            raw_sampleset = sampler.sample_cqm(self.cqm,
                                               time_limit=time_limit)

        feasible_sols = raw_sampleset.record[raw_sampleset.record.is_feasible]
        if len(feasible_sols):
            feasible_samples = \
                SampleSet(feasible_sols,
                          raw_sampleset.variables,
                          raw_sampleset.info,
                          raw_sampleset.vartype)

        self.best_feasible_sample = feasible_samples.first.sample
        print(" \n" + "=" * 30 + "FEASIBLE SAMPLE SET" + "=" * 30)
        print(feasible_samples)
        self.solution = {
            (j, i): self.best_feasible_sample[self.x[(j, i)].variables[0]]
            for i in range(data.num_machines) for j in
            range(data.num_jobs)}
        self.completion_time = self.best_feasible_sample['make_span']


if __name__ == "__main__":
    """ Modeling and solving Job Shop Scheduling using CQM solver 
    This problem is modeled in two ways: Linear and Quadratic
    For the linear model to ensure that no two jobs can be scheduled on the same
    machine at the same time we use two linear disjunctive equations, however,
    for quadratic model we used only one quadratic equations per machine.
    The input data files are provided under data folder. 
    
    """
    # Start the timer
    start_time = time()

    # Instantiate the parser
    parser = argparse.ArgumentParser(
        description='Job Shop Scheduling Using'
                    ' D-Wave ConstraintQuadraticModel solver')

    parser.add_argument('-instance', type=str,
                        help='path to to the input instance file',
                        default='input/instance5_5.txt')

    parser.add_argument('-tl', type=int,
                        help='Time limit in seconds')

    parser.add_argument('-os', type=str,
                        help='path to to the output solution file')

    parser.add_argument('-op', type=str,
                        help='path to to the output plot file')

    args = parser.parse_args()
    input_file = args.instance
    time_limit = args.tl

    # Define JSS data class
    jss_data = Data(input_file)

    # Set output plot and solution path if the are not provided by input args
    if args.op is None:
        out_plot_file = jss_data.instance_name + '.png'
    else:
        out_sol_out_plot_file = args.op

    if args.os is None:
        out_sol_file = jss_data.instance_name + '.sol'
    else:
        out_sol_file = args.os

    print(" \n" + "=" * 25 + "INPUT SETTINGS" + "=" * 25)
    print(tabulate([["Input-instance", "Time-limit"],
                    [jss_data.instance_name, time_limit]],
                   headers="firstrow"))

    # Read input data
    jss_data.read_input_data(filename=input_file)

    # Create an empty JSS CQM model.
    model = JSSCQM()

    # Define CQM model.
    model.define_cqm_model()

    # Define CQM variables.
    model.define_variables(jss_data)

    # Add precedence constraints.
    model.add_precedence_constraints(jss_data)

    # Add constraint to enforce one job only on a machine.
    model.add_quadratic_one_job_only_cstr(jss_data)

    # Add make span constraints.
    model.add_make_span_cstr(jss_data)

    # Define objective function.
    model.define_objective_function()

    # Print Model statistics
    print_cqm_stats(model.cqm)

    # Finished building the model now time it.
    model_building_time = time() - start_time
    current_time = time()
    # Call cqm solver.
    model.call_cqm_solver(time_limit, jss_data)

    # Finished solving the model now time it.
    solver_time = time() - current_time

    # Print results.
    print(" \n" + "=" * 52 + "SOLUTION RESULTS" + "=" * 52)
    print(tabulate([["Completion-Time(interval)", "Max-possible-make-span",
                     "model_building_time(s)", "solver-call-time(s)",
                     "Total-runtime(s)"],
                    [model.completion_time, jss_data.MaxMakeSpan,
                     model_building_time, solver_time, time() - start_time]],
                   headers="firstrow"))

    # Write solution to a file.
    write_solution_to_file(
        jss_data, model.solution, model.completion_time, out_sol_file)

    # Plot solution
    job_start_time, processing_time = \
        job_plotter.prep_solution_for_plotting(jss_data, model.solution)

    job_plotter.plotjssp(job_start_time, processing_time)
    plt.savefig(out_plot_file)
