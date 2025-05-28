import sys
import time

import warnings

from dimod import Binary, ConstrainedQuadraticModel, Integer
from dwave.system import LeapHybridCQMSampler
import utils.mip_solver as mip_solver
from model_data import JobShopData
from utils.utils import print_cqm_stats

sys.path.append("./src")


class JobShopSchedulingCQM:
    """Builds and solves a Job Shop Scheduling problem using CQM.

    Args:
        data (JobShopData): The data for the job shop scheduling
        max_makespan (int, optional): The maximum makespan allowed for the schedule.

    Attributes:
        data (JobShopData): The data for the job shop scheduling
        cqm (ConstrainedQuadraticModel): The CQM model
        x (dict): A dictionary of the integer variables for the start time of using machine i for job j
        y (dict): A dictionary of the binary variables which equals to 1 if job j precedes job k on machine i
        makespan (Integer): The makespan variable
        best_sample (dict): The best sample found by the CQM solver
        solution (dict): The solution to the problem
        completion_time (int): The completion time of the schedule
        max_makespan (int): The maximum makespan allowed for the schedule
        solver_name: name of the solver either CQM or MIP
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    """

    def __init__(
        self, data: JobShopData, max_makespan: int, solver_name: str, verbose: bool = False):
        self.data = data
        self.cqm = None
        self.x = {}
        self.y = {}
        self.makespan = {}
        self.best_sample = {}
        self.solution = {}
        self.completion_time = 0
        self.max_makespan = max_makespan
        self.solver_name = solver_name
        self.verbose = verbose   
        self.ordered_tasks = data.get_ordered_tasks()
        self.job_resources = {j:[] for j in data.jobs}
        for _, machine,  job in self.ordered_tasks:
             self.job_resources[job].append(machine)

    def make(self):
        """
        This function initializes the problem variables, objectives, and constraints.

        Modifies:
            self.model: prepares CQM or MIP model for solving
        """
                
        if self.solver_name == "MIP":
            # Cannot use quadratic constraints with MIP solver
            allow_quadratic_constraints = False
        elif self.solver_name == "CQM":
            allow_quadratic_constraints = True
        else:
            raise ValueError("solver_name not compatible with model")
    
        self.define_cqm_model()
        self.define_variables(self.data)
        self.add_precedence_constraints(self.data)
        if allow_quadratic_constraints:
            self.add_quadratic_overlap_constraint(self.data)
        else:
            self.add_disjunctive_constraints(self.data)
        self.add_makespan_constraint(self.data)
        self.define_objective_function()

        if self.verbose:
            print_cqm_stats(self.cqm)

    def solve(self, time_limit: int, profile: str=None):
        """Solve the model using either the CQM solver or MIP solver.
        Args:
            time_limit (int): time limit in second
            profile (str): The profile variable to pass to the Sampler. Defaults to None.
        
        Modifies:
            self.model: The sampler will solve the model, which modifies 
                the model in place.
        """

        if self.solver_name == "MIP":
            self.call_mip_solver(time_limit=time_limit)
        else:
            self.call_cqm_solver(time_limit=time_limit, data=self.data, profile=profile)
        

    def define_cqm_model(self) -> None:
        """Define CQM model."""
        self.cqm = ConstrainedQuadraticModel()

    def define_variables(self, data: JobShopData) -> None:
        """Define CQM variables.

        Args:
            data: a JobShopData data class

        Modifies:
            self.x: a dictionary of integer variables for the start time of using machine i for job j
            self.y: a dictionary of binary variables which equals to 1 if job j precedes job k on machine i
            self.makespan: an integer variable for the makespan of the schedule
        """
        # Define make span as an integer variable
        self.makespan = Integer("makespan", lower_bound=0, upper_bound=self.max_makespan)

        # Define integer variable for start time of using machine i for job j
        self.x = {}
        for job in data.jobs:
            for resource in self.job_resources[job]:
                task = data.get_resource_job_tasks(job=job, resource=resource)
                lb, ub = data.get_task_time_bounds(task, self.max_makespan)
                self.x[(job, resource)] = Integer(
                    "x{}_{}".format(job, resource), lower_bound=lb, upper_bound=ub
                )

        # Add binary variable which equals to 1 if job j precedes job k on
        # machine i
        self.y = {
            (j, k, i): Binary("y{}_{}_{}".format(j, k, i))
            for j in data.jobs
            for k in data.jobs
            for i in data.resources
        }

    def define_objective_function(self) -> None:
        """Define objective function, which is to minimize
        the makespan of the schedule.

        Modifies:
            self.cqm: adds the objective function to the CQM model
        """
        self.cqm.set_objective(self.makespan)

    def add_precedence_constraints(self, data: JobShopData) -> None:
        """Precedence constraints ensures that all operations of a job are
        executed in the given order.

        Args:
            data: a JobShopData data class

        Modifies:
            self.cqm: adds precedence constraints to the CQM model
        """
        for job in data.jobs:  # job
            for prev_task, curr_task in zip(
                data.job_tasks[job][:-1], data.job_tasks[job][1:]
            ):
                machine_curr = curr_task.resource
                machine_prev = prev_task.resource
                self.cqm.add_constraint(
                    self.x[(job, machine_curr)] - self.x[(job, machine_prev)] >= prev_task.duration,
                    label="pj{}_m{}".format(job, machine_curr),
                )

    def add_quadratic_overlap_constraint(self, data: JobShopData) -> None:
        """Add quadratic constraints to ensure that no two jobs can be scheduled
         on the same machine at the same time.

         Args:
             data: a JobShopData data class

        Modifies:
            self.cqm: adds quadratic constraints to the CQM model
        """
        for j in data.jobs:
            for k in data.jobs:
                if j < k:
                    for i in data.resources:
                        if not (i in self.job_resources[k] and i in self.job_resources[j]):
                            continue
                        task_k = data.get_resource_job_tasks(job=k, resource=i)
                        task_j = data.get_resource_job_tasks(job=j, resource=i)

                        if task_k.duration > 0 and task_j.duration > 0:
                            self.cqm.add_constraint(
                                self.x[(j, i)]
                                - self.x[(k, i)]
                                + (task_k.duration - task_j.duration) * self.y[(j, k, i)]
                                + 2 * self.y[(j, k, i)] * (self.x[(k, i)] - self.x[(j, i)])
                                >= task_k.duration,
                                label="OneJobj{}_j{}_m{}".format(j, k, i),
                            )

    def add_disjunctive_constraints(self, data: JobShopData) -> None:
        """This function adds the disjunctive constraints the prevent two jobs
        from being scheduled on the same machine at the same time. This is a
        non-quadratic alternative to the quadratic overlap constraint.

        Args:
            data (JobShopData): The data for the job shop scheduling

        Modifies:
            self.cqm: adds disjunctive constraints to the CQM model
        """
        V = self.max_makespan
        for j in data.jobs:
            for k in data.jobs:
                if j < k:
                    for i in data.resources:
                        task_k = data.get_resource_job_tasks(job=k, resource=i)
                        self.cqm.add_constraint(
                            self.x[(j, i)]
                            - self.x[(k, i)]
                            - task_k.duration
                            + self.y[(j, k, i)] * V
                            >= 0,
                            label="disjunction1{}_j{}_m{}".format(j, k, i),
                        )

                        task_j = data.get_resource_job_tasks(job=j, resource=i)
                        self.cqm.add_constraint(
                            self.x[(k, i)]
                            - self.x[(j, i)]
                            - task_j.duration
                            + (1 - self.y[(j, k, i)]) * V
                            >= 0,
                            label="disjunction2{}_j{}_m{}".format(j, k, i),
                        )

    def add_makespan_constraint(self, data: JobShopData) -> None:
        """Ensures that the make span is at least the largest completion time of
        the last operation of all jobs.

        Args:
            data: a JobShopData data class

        Modifies:
            self.cqm: adds the makespan constraint to the CQM model
        """
        for job in data.jobs:
            last_job_task = data.job_tasks[job][-1]
            last_machine = last_job_task.resource
            self.cqm.add_constraint(
                self.makespan - self.x[(job, last_machine)] >= last_job_task.duration,
                label="makespan_ctr{}".format(job),
            )

    def call_cqm_solver(self, time_limit: int, data: JobShopData, profile: str) -> None:
        """Calls CQM solver.

        Args:
            time_limit (int): time limit in second
            data (JobShopData): a JobShopData data class
            profile (str): The profile variable to pass to the Sampler. Defaults to None.
            See documentation at
            https://docs.dwavequantum.com/en/latest/ocean/api_ref_cloud/generated/dwave.cloud.config.load_config.html

        Modifies:
            self.feasible_sampleset: a SampleSet object containing the feasible solutions
            self.best_sample: the best sample found by the CQM solver
            self.solution: the solution to the problem
            self.completion_time: the completion time of the schedule
        """
        sampler = LeapHybridCQMSampler(profile=profile)
        min_time_limit = sampler.min_time_limit(self.cqm)
        if time_limit is not None:
            time_limit = max(min_time_limit, time_limit)
        raw_sampleset = sampler.sample_cqm(self.cqm, time_limit=time_limit, label="Job Shop Demo")
        self.feasible_sampleset = raw_sampleset.filter(lambda d: d.is_feasible)
        num_feasible = len(self.feasible_sampleset)
        if num_feasible > 0:
            best_samples = self.feasible_sampleset.truncate(min(10, num_feasible))
        else:
            warnings.warn("Warning: CQM did not find feasible solution")
            best_samples = raw_sampleset.truncate(10)

        print(" \n" + "=" * 30 + "BEST SAMPLE SET" + "=" * 30)
        print(best_samples)
        self.best_sample = best_samples.first.sample

    def call_mip_solver(self, time_limit: int = 100):
        """This function calls the MIP solver and returns the solution

        Args:
            time_limit (int, optional): The maximum amount of time to
            allow the MIP solver to before returning. Defaults to 100.

        Modifies:
            self.best_sample: the best feasoble sampel obtained form solver
        """
        solver = mip_solver.MIPCQMSolver()
        raw_sampleset = solver.sample_cqm(cqm=self.cqm, time_limit=time_limit)
        if len(raw_sampleset) == 0:
            warnings.warn("Warning: MIP did not find feasible solution")
            return
        
        best_samples = raw_sampleset.truncate(10)
        print(" \n" + "=" * 30 + "BEST SAMPLE SET" + "=" * 30)
        print(best_samples)
        self.best_sample = best_samples.first.sample
    
    def compute_results(self) -> None:
        """Extracts the results from a solved model and prints to console.
        """
        if self.solver_name == "CQM":
            self.solution = {
                (j, i): (
                    # self.data.get_resource_job_tasks(job=j, resource=i),
                    self.best_sample[self.x[(j, i)].variables[0]],
                    self.data.get_resource_job_tasks(job=j, resource=i).duration,
                )
                for j in self.data.jobs for i in self.job_resources[j]
            }
            
        elif self.solver_name == "MIP":
            for var, val in self.best_sample.items():
                if var.startswith("x"):
                    job, machine = var[1:].split("_")
                    job = int(job)
                    machine = int(machine)
                    task = self.data.get_resource_job_tasks(job=job, resource=machine)
                    self.solution[(job, machine)] = val, task.duration
        else:
            raise ValueError("Solver")
        
        self.completion_time = self.best_sample["makespan"]
