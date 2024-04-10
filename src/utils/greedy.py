'''
This file will greedily generate a solution for the job shop problem
'''
import sys

import numpy as np

sys.path.append('./src')
from model_data import JobShopData


class GreedyJobShop:

    def __init__(self, model_data: JobShopData):
        """Initializes the GreedyJobShop class.

        Args:
            model_data (JobShopData): A JobShopData object that holds 
                the data for this job shop
        """        
        self.model_data = model_data
    

    def solve(self, skip_probability: float = 0.1, seed : int = 42) -> dict:
        '''
        This solves the job shop scheduling problem using the 
        following strategy:
        1. Randomly select a job with open tasks
        2. Select the first open task for that job
        3. Assing the task to its required resource at the earliest time
            a. First check to see if there is a gap in the resource schedule; 
                if so, assign the task to the start of the gap
            b. If there is no gap, assign the task to the end of the resource
                schedule
        4. Repeat until all tasks are assigned

        Args:
            skip_probability (float): The probability of skipping a job each iteration.
                This is used to add randomness to the solution. Defaults to 0.1.
            seed (int): The random seed to use. Defaults to 42.
        
        Returns:
            dict: A dictionary of task assignments. The keys are tasks and the values
                are tuples of the start and finish times for the task.
        '''
        
        def resource_gap_finder(resource_schedule: list,
                                min_start: int,
                                gap_duration: int) -> dict:
            """helper function that checks to see if there is a gap in the 
            resource_schedule of at least gap_duration that starts after min_start

            Args:
                resource_schedule (list): The schedule of tasks currently assigned to the resource,
                    sorted by start time. Each element is a dictionary with keys 'start', 'finish', 
                    and 'task'
                min_start (int): The earliest time that the task can start
                gap_duration (int): The minimum gap duration that is needed to fit the task

            Returns:
                dict: A dictionary of the form {Task: (start_time, finish_time)} for the each
                    assigned task
            """            
            for i in range(len(resource_schedule) - 1):
                if resource_schedule[i+1]['start'] > min_start and \
                    resource_schedule[i+1]['start'] - max(min_start, resource_schedule[i]['finish']) >= gap_duration:
                    return max(min_start, resource_schedule[i]['finish']), i+1
            return max(min_start, resource_schedule[-1]['finish']), len(resource_schedule)

        rng = np.random.default_rng(seed)
        resource_schedules = {resource: [] for resource in self.model_data.resources}
        job_schedules = {job: [] for job in self.model_data.jobs}
        last_task_scheduled = {job: -1 for job in self.model_data.jobs}
        task_assignments = {}
        unfinished_jobs = [x for x in self.model_data.jobs]
        remaining_task_times = {job: self.model_data.get_total_job_time(job) for job in self.model_data.jobs}
        unfinished_jobs = np.array([x for x in self.model_data.jobs])
        rng.shuffle(unfinished_jobs)
        not_yet_finished = np.ones(len(unfinished_jobs))
        idx = 0

        while sum(not_yet_finished) > 0:
            job = unfinished_jobs[idx % len(unfinished_jobs)]
            if not_yet_finished[idx % len(unfinished_jobs)] == 0:
                idx += 1
                continue

            if rng.uniform() < skip_probability:
                idx += 1
                continue
            
            task = self.model_data.job_tasks[job][last_task_scheduled[job] + 1]
            resource = task.resource

            if len(job_schedules[job]) == 0:
                min_job_time = 0
            else:
                min_job_time = job_schedules[job][-1]['finish']
            if len(resource_schedules[resource]) == 0:
                min_resource_time = max(0, min_job_time)
                resource_pos = 0
            else: 
                min_resource_time, resource_pos = resource_gap_finder(resource_schedules[resource], min_job_time, task.duration)

            start_time = min_resource_time
            finish_time = task.duration + start_time
            resource_schedules[resource].insert(resource_pos, {'start': start_time, 'finish': finish_time, 'task': task})
            job_schedules[job].append({'start': start_time, 'finish': finish_time, 'task': task})
            task_assignments[task] = (start_time, finish_time)
            last_task_scheduled[job] += 1
            if last_task_scheduled[job] == len(self.model_data.job_tasks[job]) - 1:
                not_yet_finished[idx % len(unfinished_jobs)] = 0
            idx += 1
            remaining_task_times[job] -= task.duration

        return task_assignments