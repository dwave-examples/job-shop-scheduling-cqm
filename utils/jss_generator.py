import random
import re
import argparse
from collections import defaultdict
from os import path


def generate_random_jss(n_jobs: int, n_machine: int, max_operation_time: int,
                        location: str) -> dict:
    """Generate random job shop problems

    Args:
        n_jobs: number of jobs
        n_machine: number of machines
        max_operation_time: maximum operation time
        location: path to the generate file
    returns:
        job_dict:
    """

    job_dict = defaultdict(list)
    machines = list(range(n_machine))
    for i in range(n_jobs):
        random.shuffle(machines)
        job_dict[i + 1] = [(m, random.randint(0, max_operation_time))
                           for m in machines]
    filename = 'instance' + str(n_jobs) + '_' + str(n_machine)
    if path.exists(location + '/' + filename + '.txt'):
        print(location + '/' + filename + '.txt', "already exist.")
    else:
        with open(location + '/' + filename + '.txt', 'w') as f:
            f.write(str(n_jobs) + '\t' + str(n_machine) + '\n')
            for i in job_dict:
                f.write(re.sub('[^A-Za-z0-9]+', '\t',
                               str(job_dict[i])).strip() + '\n')
        f.close()

    return job_dict


if __name__ == "__main__":
    # Instantiate the parser
    parser = argparse.ArgumentParser(
        description='Job Shop Scheduling Instance Generator')

    parser.add_argument('-n', type=int, help='num of jobs')
    parser.add_argument('-m', type=int, help='num of jobs')
    parser.add_argument('-d', type=int, help='maximum processing duration')
    parser.add_argument('-path', type=str,
                        help='folder location to store generated instance file')
    args = parser.parse_args()
    num_jobs = args.n
    num_machines = args.m
    duration = args.d
    location = args.path
    jobs = generate_random_jss(num_jobs, num_machines, duration, location)
