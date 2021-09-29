import os
import random
import re
from collections import defaultdict

import argparse


def generate_random_jss(n_jobs: int, n_machine: int, max_operation_time: int,
                        location: str) -> dict:
    """Generate random job shop problems

    Args:
        n_jobs: number of jobs
        n_machine: number of machines
        max_operation_time: maximum operation time
        location: path to the generated file
    returns:
        job_dict:
    """

    job_dict = defaultdict(list)
    machines = list(range(n_machine))
    for i in range(n_jobs):
        random.shuffle(machines)
        job_dict[i + 1] = [(m, random.randint(0, max_operation_time))
                           for m in machines]
    filename = location + '/' + 'instance' + str(n_jobs) + '_' + str(n_machine) + '.txt'
    print(filename)
    if os.path.exists(filename):
        print(filename, "already exist.")
    else:
        with open(filename, 'w') as f:
            f.write(str(n_jobs) + '\t' + str(n_machine) + '\n')
            for i in job_dict:
                f.write(re.sub('[^A-Za-z0-9]+', '\t',
                               str(job_dict[i])).strip() + '\n')
        print(f'Saved schedule to '
              f'{os.path.join(os.getcwd(), filename)}')

        f.close()

    return job_dict


if __name__ == "__main__":
    # Instantiate the parser
    parser = argparse.ArgumentParser(
        description='Job Shop Scheduling Instance Generator',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('n', type=int, help='num of jobs')
    parser.add_argument('m', type=int, help='num of machines')
    parser.add_argument('d', type=int, help='maximum processing duration')
    parser.add_argument('-path', type=str,
                        help='folder location to store generated instance file',
                        default='input')
    args = parser.parse_args()
    num_jobs = args.n
    num_machines = args.m
    duration = args.d
    location = args.path
    jobs = generate_random_jss(num_jobs, num_machines, duration, location)
