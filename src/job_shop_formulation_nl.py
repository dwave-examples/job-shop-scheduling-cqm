from dwave.system import LeapHybridNLSampler
from dwave.optimization import Model
import numpy as np
from dwave.optimization import Model
from dwave.optimization.mathematical import maximum, put
from model_data import JobShopData
import warnings


class JobShopSchedulingNL:

    """Builds and solves a Job Shop Scheduling problem using LeapHybridNLSampler.

    Args:
        data (JobShopData): The data for the job shop scheduling
        max_makespan (int, optional): The maximum makespan allowed for the schedule.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
    """
    def __init__(self, data: JobShopData,  max_makespan: int, verbose: bool=False):

        self.machine_dict = {m: i for i, m in enumerate(data.resources)}
        self.job_dicts = {j: i for i, j in enumerate(data.jobs)}

        self.n_machines = len(self.machine_dict)
        self.n_jobs = len(self.job_dicts)

        self.machines = list(self.machine_dict.values())
        self.jobs = list(self.job_dicts.values())

        self.max_makespan = max_makespan
        ordered_tasks, zero_duration_task = data.get_ordered_tasks()
        self.ordered_tasks = [(d, self.machine_dict[m], self.job_dicts[j]) for (d, m , j) in ordered_tasks]
        self.zero_duration_task = [(d, self.machine_dict[m], self.job_dicts[j]) for (d, m , j) in zero_duration_task]
        
        self.n_tasks =  len(self.ordered_tasks)
        self.task_durations = list([v[0] for v in self.ordered_tasks])
        self.task_machines = list([v[1] for v in self.ordered_tasks])
        self.task_jobs = list([v[2] for v in self.ordered_tasks])
        
        # Mapping between jobs and tasks. 
        self.job_tasks = {j:[] for j in self.jobs}
        for i, j in enumerate(self.task_jobs):
            self.job_tasks[j].append(i)
        self.job_process_order = {j:i for i, j in enumerate(self.jobs)}
        self.verbose = verbose
        self.solution = {}
        self.completion_time = 0
    
    def define_variables(self) -> None:
        """
        This function initializes the problem variables
        
        Initializes:
            self.nl_model: The NL model that will be solved is initialised here
            self.task_order: decision variable that determines the order in
                which tasks are performed
        """
        self.nl_model = Model()
        self.task_order = self.nl_model.list(self.n_tasks)
        if self.verbose:
            print(f"Model variables defined, total number of nodes {self.nl_model.num_nodes()}")


    def add_precedence_constraints(self) -> None:
        """defines a constraint which ensures that each task within the same job
        are completed in the correct order

        Modifies:
            self.nl_model: adds constraints to ensure precedence constraints are
                respected in task ordering
        """

        """Precedence constraints ensures that all operations of a job are
        executed in the given order. 

        Modifies:
            self.nl_model: adds precedence constraints to the nl model
        """


        constant_numbers = self.nl_model.constant(np.arange(self.n_tasks + 1))
        one = constant_numbers[1]
        positions = constant_numbers[1: self.n_tasks + 1]

        for job_j in self.jobs:
            job_tasks = self.job_tasks[job_j]
            for task_t in range(1, len(job_tasks)):
                succ = job_tasks[task_t]
                pred = job_tasks[task_t - 1]
                successor_task_id = constant_numbers[succ]
                predecessor_task_id = constant_numbers[pred]
                successors_index =  ((self.task_order == successor_task_id) * positions).sum()
                predecessors_index =  ((self.task_order == predecessor_task_id) * positions).sum()
                self.nl_model.add_constraint(successors_index >= predecessors_index + one)

        if self.verbose:
            print(f"Added precedence constraint, total number of nodes {self.nl_model.num_nodes()}")

    def build_schedule(self) -> None:
        """
        builds a schedule from the the model decision variables  the start time of each task
        given in self.task_order is used to build the scehdule considering machine availability 
        and job sequencing constraints.
        A task is assigned when 
        - Its assigned machine is free.
        - The previous task in the same job (if any) has completed.
        
        Modifies:
            self.start_times: the start time of each task, ordered by self.task_order 
            self.finish_times: the finish time of each task, ordered according to 
            self.task_order and calculated as the sum of start time, duration and cleaning time
            
        """
        
        machine_available_time = self.nl_model.constant([0] * self.n_machines)
        job_last_task_end_time = self.nl_model.constant([0] * self.n_jobs)
        task_end_times = self.nl_model.constant([0] * self.n_tasks)
        task_start_times = self.nl_model.constant([0] * self.n_tasks)
        job_ids = self.nl_model.constant(self.task_jobs)
        machine_ids = self.nl_model.constant(self.task_machines)    
        durations = self.nl_model.constant(self.task_durations)

        for i in range(self.n_tasks):
            task_idx = self.task_order[i]
            machine_id = machine_ids[task_idx]
            job_id = job_ids[task_idx]
            duration = durations[task_idx]
            
            # Calculate the time that assgined machine for this taks is avilable
            machine_time = machine_available_time[machine_id]

            # Calculate the time that the current job for the the current taks is avilable 
            job_time = job_last_task_end_time[job_id]
            
            # Start time is calculated when bothe machine and job is ready. 
            start_time = maximum(machine_time, job_time)
            end_time = start_time + duration
            
            task_start_times = put(task_start_times, task_idx.reshape((1,)), start_time.reshape((1,)))
            task_end_times = put(task_end_times, task_idx.reshape((1,)), end_time.reshape((1,)))
            
            # Update machine and job availability
            machine_available_time = put(machine_available_time, machine_id.reshape((1,)), end_time.reshape((1,)))
            job_last_task_end_time = put(job_last_task_end_time, job_id.reshape((1,)), end_time.reshape((1,)))

        self.start_times = task_start_times
        self.finish_times = task_end_times
        if self.verbose:
            print(f"Tasks start and finish times computed, total number of nodes {self.nl_model.num_nodes()}")   
            
    def make(self) -> None:
        """
        This function initializes the problem variables, objectives, and constraints.

        Modifies:
            self.nl_model: prepares NL model for solving
        """
        #define constraints
        self.define_variables()
        self.add_precedence_constraints()
        #build start and finish times
        self.build_schedule()
        #set the objectives
        self.define_objectives()
        self.nl_model.lock()
        
        if self.verbose:
            print(f"Finished building NL model, total number of nodes {self.nl_model.num_nodes()}")
    

    def define_objectives(self) -> None:
            """this function sets the objective of the NL model
            """
            self.obj_makespan = self.finish_times.max()
            self.nl_model.minimize(self.obj_makespan)
    
    def solve(self, time_limit: int, profile: str = None) -> None:
        """Solve the model using the LeapHybridNLSampler.
        Args:
            time_limit (int): time limit in second
            profile (str): The profile variable to pass to the Sampler. Defaults to None.
            See documentation at
            https://docs.dwavequantum.com/en/latest/ocean/api_ref_cloud/generated/dwave.cloud.config.load_config.html

        Modifies:
            self.nl_model: The sampler will solve the model, which modifies 
                the model in place.
        """
        sampler = LeapHybridNLSampler(profile=profile)
        sampler.sample(self.nl_model, time_limit=time_limit, label="jobshop NL")

        if self.verbose:
            print(f"NL model upload to Leap completed.")
        # Ensure that sampling is done
        self.nl_model.states.resolve()

    def compute_results(self) -> None:
        """Extracts the results from a solved model and prints to console.
        """

        solution_index = 0
        order = [int(task) for task in self.task_order.state(solution_index)]
        
        # Check if the solution is feasible
        solution_feasibible = all(c.state(solution_index) for c in self.nl_model.iter_constraints())
        job_dicts_rev = {j: i for i, j in self.job_dicts.items()}
        machine_dict_rev = {m: i for i, m in self.machine_dict.items()}

        if solution_feasibible:
            makespan = self.obj_makespan.state(solution_index)        
            start_times_schedule = self.start_times.state(solution_index)
            
            self.solution = {}
            for task_t in order:
                job = self.task_jobs[task_t]
                machine = self.task_machines[task_t]
                start = start_times_schedule[task_t]
                duration = self.task_durations[task_t]
                self.solution[(job_dicts_rev[job], machine_dict_rev[machine])] = (start, duration)

            # Now add the task with zero duratins to the end of schedule. 
            for (duration, machine, job) in self.zero_duration_task:
                self.solution[(job_dicts_rev[job], machine_dict_rev[machine])] = (makespan, duration)

            self.completion_time = makespan

        else:            
            warnings.warn("Warning: NL did not find feasible solution")
