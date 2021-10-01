import os
import random
from collections import defaultdict

from tabulate import tabulate
import argparse


def generate_random_jss(n_jobs: int, n_machine: int, max_operation_time: int,
                        location: str) -> None:
    """Generate random job shop problems

    Args:
        n_jobs: number of jobs
        n_machine: number of machines
        max_operation_time: maximum operation time
        location: path to the generated file

    """

    job_dict = defaultdict(list)
    machines = list(range(n_machine))
    for i in range(n_jobs):
        random.shuffle(machines)
        job_dict[i] = [(task, m, random.randint(0, max_operation_time))
                       for task, m in enumerate(machines)]
    filename = location + '/' + 'instance' + \
               str(n_jobs) + '_' + str(n_machine) + '.txt'

    print(filename)
    if os.path.exists(filename):
        print(filename, "already exist.")
    else:
        task_header = " " * 10
        for i in range(n_machine):
            task_header += " " * 6 + f'task {i}' + " " * 6

        header = ["job id"]
        for i in range(n_machine):
            header.extend(['machine', 'dur'])

        tasks_info = {j: [j] for j in range(n_jobs)}
        for j, v in job_dict.items():
            for k in v:
                tasks_info[j].extend(k[1:])


        with open(filename, 'w') as f:
            f.write(f'#Num of jobs: {n_jobs} \n')
            f.write(f'#Num of machines: {n_machine} \n')
            f.write(task_header)
            f.write('\n')
            f.write(tabulate([header, *[v for l, v in tasks_info.items()]],
                             headers="firstrow"))
            f.write('\n')

        print(f'Saved schedule to '
              f'{os.path.join(os.getcwd(), filename)}')

        f.close()


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
    generate_random_jss(num_jobs, num_machines, duration, location)
