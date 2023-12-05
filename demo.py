# Copyright 2019 D-Wave Systems Inc.
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
import json
import pandas as pd


from src.job_shop_scheduler_cqm import JobShopScheduler
from src.model_data import JobShopData
from src.utils.utils import read_instance


def prep_output_for_dashboard(task_assignments: dict) -> pd.DataFrame:
    """This function takes the output of the JobShopScheduler.solve() function and
    transforms it into a pandas DataFrame that can be used to generate a Gantt chart
    in the dashboard.

    Args:
        task_assignments (dict): A dictionary that maps tasks to their start times.

    Returns:
        pd.DataFrame: A DataFrame that can be used to generate a Gantt chart in the
        dashboard. The DataFrame has the following columns: Task, Start, Finish, and
        Resource.
    """    
    task_data = []
    for task, (start_time, finish_time) in task_assignments.items():
        task_data.append({'Job': task.job, 'Start': start_time, 'Finish': finish_time, 'Resource': task.resource, 'Duration': task.duration})
    df = pd.DataFrame(task_data)
    return df


def run_job_shop_scheduler(job_data: JobShopData, max_time: int = None) -> pd.DataFrame:
    """This function runs the job shop scheduler on the given data.

    Args:
        job_data (JobShopData): A JobShopData object that holds the data for this job shop 
        scheduling problem.
        max_time (int, optional): Upperbound on how long the schedule can be; leave empty to 
        auto-calculate an appropriate value. Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame that can be used to generate a Gantt chart in the
        dashboard. The DataFrame has the following columns: Task, Start, Finish, and
        Resource.
    """    
    cqm = JobShopScheduler(job_data)
    task_assignments = cqm.solve()
    task_assignment_df = prep_output_for_dashboard(task_assignments)
    return task_assignment_df



# Main routine definition
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_filename", type=str, nargs="?",
                        help="Name of the input dataset, in JSON format. This should be a filename in the data folder",
                        default='bakery.json')
    
    parser.add_argument("--max_time", type=int, nargs="?",
                        help="Upperbound on how long the schedule can be; leave empty to auto-calculate an appropriate value",
                        default=None)
    
    args = parser.parse_args()

    input_file = 'input/instance5_5.txt'
    job_data = JobShopData()
    # job_data.load_from_json('data/' + args.data_filename)
    job_dict = read_instance(input_file)
    job_data = JobShopData()
    job_data.load_from_dict(job_dict)
    
    # for job, task_list in jobs.items():
    #     for task in task_list:
    #         job_data.add_task(Task(job, duration=task['duration'], resource=task['resource']))
    
    # cqm = JobShopScheduler(job_data)
    # task_assignments = cqm.solve()
    task_assignment_df = run_job_shop_scheduler(job_data, max_time=args.max_time)
    import pdb
    pdb.set_trace()


if __name__ == "__main__":
    main()

