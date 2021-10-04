import os
from collections import defaultdict

import numpy as np
import argparse
import matplotlib

try:
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib.use("agg")
    import matplotlib.pyplot as plt


def plot_solution(data, solution: dict, location: str = None) -> tuple:
    """Prepare Jss solution for plotting

    Args:
        data: JSS data class
        solution: a dictionary of solution
        location: path for saving scheduling plot

    """
    job_start_time = defaultdict(list)
    processing_time = defaultdict(list)
    for j in range(data.num_jobs):
        job_start_time[j] = [solution[(j, i)][1] for i in
                             range(data.num_machines)]
        processing_time[j] = [
            data.task_duration[j, data.machine_task[(j, i)]] for i in
            range(data.num_machines)]
    if location is not None:
        plot_schedule_core(job_start_time, processing_time, location)
    return job_start_time, processing_time


def read_solution(path: str) -> tuple:
    """This function read a Job shop solution file

    Args:
        path: path to the input solution file

    Returns:
        job_start_time: start time of each job on each machine
        processing_time: processing duration of each job on each machine

    """
    job_start_time = defaultdict(list)
    processing_time = defaultdict(list)

    with open(path) as f:
        f.readline()
        k = -1
        for i, line in enumerate(f):
            if i < 6:
                continue
            k += 1
            lint = list(map(int, line.split()))[1:]
            job_start_time[k] = lint[1::3]
            processing_time[k] = lint[2::3]

    return job_start_time, processing_time


def plot_schedule_core(job_start_time: dict, processing_time: dict,
                       location) -> None:
    """This function plots job shop problem
    Args:
        job_start_time: start time of each job on each machine
        processing_time: processing duration of each job on each machine
        location: path for saving scheduling plot
    """

    sols = np.array(list(job_start_time.values()))
    durs = np.array(list(processing_time.values()))
    solsT = sols.transpose()
    dursT = durs.transpose()
    n, m = sols.shape
    labels = ['machine ' + str(i) for i in range(m)]
    category_names = ['job ' + str(i) for i in range(n)]

    category_colors = plt.get_cmap('RdYlGn')(
        np.linspace(0.15, 0.85, sols.shape[0]))
    fig, ax = plt.subplots()
    ax.invert_yaxis()
    ax.xaxis.set_visible(True)

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = dursT[:, i]
        starts = solsT[:, i]
        ax.barh(labels, widths, left=starts, height=.5,
                label=colname)
        xcenters = starts + widths / 2
        r, g, b, _ = color
        text_color = 'white'
        for y, (x, c) in enumerate(zip(xcenters, widths)):
            if c > 0:
                ax.text(x, y, str(int(c)), ha='center', va='center',
                        color=text_color)
    ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='small')

    plt.savefig(location)

    print(f'Saved plot to {os.path.join(os.getcwd(), location)}')

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Plot a schedule given by a JSS solution file.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('s', type=str,
                        help='path to the input solution file')

    parser.add_argument('-op', type=str,
                        help='path to the output plot file',
                        default="schedule.png")

    args = parser.parse_args()
    input_solution = args.s
    out_solution = args.op
    job_start_time, processing_time = read_solution(input_solution)
    plot_schedule_core(job_start_time, processing_time, out_solution)
    plt.savefig('schedule.png')
    plt.show()
