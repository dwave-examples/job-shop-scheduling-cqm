# Copyright 2023 D-Wave Systems Inc.
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

from __future__ import print_function

import argparse
import time

import dimod
import pandas as pd

from dwave.system import LeapHybridCQMSampler

import sys
sys.path.append('./src')
from model_data import JobShopData
from utils.utils import print_cqm_stats


def sum_to_one(*args):
    return sum(args) == 1


def get_label(task, time):
    """Creates a standardized name for variables in the constraint satisfaction problem,
    JobShopScheduler.csp.
    """
    return "{task.job}_{task.resource},{time}".format(**locals())


class JobShopScheduler:
    def __init__(self, model_data: JobShopData, max_makespan: int=None):
        """
        Args:
            model_data: A JobShopData object that holds the data for this job shop 
            scheduling problem.
            max_makespan: An integer. The maximum makespan for the schedule. If None, 
            then this is calculated automatically.

        """
        self.model_data = model_data
        if max_makespan is None:
            self.max_makespan = model_data.get_max_makespan()
        else:
            self.max_makespan = max_makespan
        self._initialize_variables()
        self.cqm = dimod.ConstrainedQuadraticModel()
        self._add_task_start_constraint()
        self._add_precedence_constraint()
        self._add_share_machine_constraint()
        self._remove_absurd_times()
        # self._model_end()
        self._add_objective()


    def solve(self, profile: str=None, time_limit:int=10, verbose: bool=True) -> dict:
        """Solves the job shop scheduling problem.
        
        Args:
            profile: A string. The name of the profile to use for the hybrid solver.

        Returns:
            dict: A dictionary of the form {task: (start_time, end_time)} for each task in the
            problem.
        """
        if verbose:
            print_cqm_stats(self.cqm)
        sampler = LeapHybridCQMSampler(profile=profile)
        res = sampler.sample_cqm(self.cqm, time_limit=time_limit, label='Job Shop Demo BQM')

        res.resolve()
        feasible_sampleset = res.filter(lambda d: d.is_feasible)
        try:
            best_feasible = feasible_sampleset.first.sample

        except ValueError:
            raise RuntimeError(
                "Sampleset is empty, try increasing time limit or " +
                "adjusting problem config."
            )

        task_assignments = {}
        for task in self.model_data.get_tasks():
            for t in range(self.max_makespan):
                if best_feasible[get_label(task, t)] == 1:
                    start_time = t
                    end_time = t + task.duration
                    task_assignments[task] = (start_time, end_time)
                    break

        self.solution = task_assignments
        df = self.solution_as_dataframe()
        return df

    
    def _initialize_variables(self) -> None:
        """Initialize the variables for the constraint satisfaction problem.
        """
        task_start_vars = {get_label(task, t): dimod.Binary(get_label(task, t)) 
                        for t in range(self.max_makespan)
                        for task in self.model_data.get_tasks()}
        self.task_start_vars = task_start_vars




    def _add_task_start_constraint(self) -> None:
        '''
        This adds this constraint to the model requiring that each task be started exactly once.
        '''
        for task in self.model_data.get_tasks():
            self.cqm.add_constraint(sum(self.task_start_vars[get_label(task, t)] for t in range(self.max_makespan)) == 1)
        
        
    def _add_precedence_constraint(self) -> None:
        """Within the same job, tasks must be scheduled in the order they are given.
        Additionally, the start time of a task must be at least as large as the start
        time of the previous task plus the duration of the prior task.
        """
        for job_tasks in self.model_data.job_tasks.values():
            if len(job_tasks) < 2:
                continue

            for precendent_idx, precedent_task in enumerate(job_tasks[:-1]):
                next_task = job_tasks[precendent_idx + 1]
                
                # Forming constraints with the relevant times of the next task
                for t in range(self.max_makespan):
                    for tt in range(0, min(t + precedent_task.duration, self.max_makespan)):
                        self.cqm.add_constraint(
                            self.task_start_vars[get_label(precedent_task, t)] + \
                            self.task_start_vars[get_label(next_task, tt)] + \
                            self.task_start_vars[get_label(precedent_task, t)] * self.task_start_vars[get_label(next_task, tt)] <= 1)



    def _add_share_machine_constraint(self) -> None:
        """self.csp gets the constraint: At most one task per machine per time
        """
        for resource in self.model_data.resources:
            resource_tasks = self.model_data.get_resource_tasks(resource)

            # No need to build coupling for a single task
            if len(resource_tasks) < 2:
                continue

            # Apply constraint between all tasks for each unit of time
            for task1 in resource_tasks:
                for task2 in resource_tasks:
                    if task1 == task2:
                        continue
                    for t in range(self.max_makespan):
                        for tt in range(t, min(t + task1.duration, self.max_makespan)):
                            self.cqm.add_constraint(self.task_start_vars[get_label(task1, t)] + \
                                                    self.task_start_vars[get_label(task2, tt)] + \
                                                    self.task_start_vars[get_label(task1, t)] * self.task_start_vars[get_label(task2, tt)] <= 1)
                        

    def _remove_absurd_times(self) -> None:
        """This function fixes to 0 the variables for times that cannot be part
        of an optimal schedule. Variables that cannot be part of an optimal schedule
        include:
        - Times that are too early for a task to start (e.g. if the first task for a
            job takes 2 time periods, then the second task for the same job cannot
            start until time period 3 at the earliest)
        - Times that are too late for a task to start (e.g. if the last task for a
            job takes 3 time periods, then the second-to-last task for the same 
            job cannot start later than time period (total time - 2))
        """
        # Times that are too early for a task
        for job_tasks in self.model_data.job_tasks.values():
            predecessor_time = 0
            for task in job_tasks:
                for t in range(predecessor_time):
                    self.cqm.add_constraint_from_model(
                        self.task_start_vars[get_label(task, t)],
                        "==",
                        0,
                        label='predecessor_fixing_' + str(task) + '_' + str(t))


                predecessor_time += task.duration
        
        # Times that are too late for a task
        for job_tasks in self.model_data.job_tasks.values():
            successor_time = 0    # start with -1 so that we get (total task time - 1)
            for task in reversed(job_tasks):
                successor_time += task.duration
                for t in range(successor_time):
                    self.cqm.add_constraint_from_model(
                        self.task_start_vars[get_label(task, self.max_makespan - t - 1)],
                        "==",
                        0,
                        label='succession_fixing_' + str(task) + '_' + str(self.max_makespan - t - 1))


    def _add_objective(self):
        """This function adds the objective function to the constrained quadratic model.
        """
        self.cqm.set_objective(
            sum(
                sum(t * self.task_start_vars[get_label(task, t)] \
                    for task in self.model_data.get_last_tasks()
                    ) for t in range(self.max_makespan)
                )
            )
        

    def solution_as_dataframe(self) -> pd.DataFrame:
        """This function returns the solution as a pandas DataFrame

        Returns:
            pd.DataFrame: A pandas DataFrame containing the solution
        """        
        df_rows = []
        for (task), (start, end) in self.solution.items():
            df_rows.append([task.job, task, start, end, task.resource])
        df = pd.DataFrame(df_rows, columns=['Job', 'Task', 'Start', 'Finish', 'Resource'])
        return df
    
    # def _model_end(self):
    #     """This function adds a variable to the constrained quadratic model that
    #     represents the time at which the last task is completed.
    #     """
    #     self.model_end_vars = {t: dimod.Binary('model_end_var_{}'.format(t)) for t in range(self.max_time)}
    #     for task in self.model_data.get_last_tasks():
    #         for t in range(self.max_time):
    #             for tt in range(t):
    #                 self.cqm.add_constraint(
    #                     self.task_start_vars[get_label(task, t)] + \
    #                     self.model_end_vars[tt] +\
    #                     self.task_start_vars[get_label(task, t)] * self.model_end_vars[tt] <= 1,
    #                     label='model_end_constraint_{}_{}_{}'.format(task, t, tt)
    #                 )
    #     self.cqm.add_constraint(sum(self.model_end_vars[t] for t in range(self.max_time)) == 1)
    
    # def _add_objective(self):
    #     self.cqm.set_objective(
    #         sum(t * self.model_end_vars[t] for t in range(self.max_time))
    #     )
        


if __name__ == "__main__":
    """Modeling and solving Job Shop Scheduling using CQM solver."""


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
    
    parser.add_argument('-verbose', action='store_true', default=True,
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

    jss = JobShopScheduler(job_data, max_makespan=args.max_makespan)
    res = jss.solve(time_limit=time_limit, profile=args.profile)

    import pdb
    pdb.set_trace()