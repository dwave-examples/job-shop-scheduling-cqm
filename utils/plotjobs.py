import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import sys


def prep_solution_for_plotting(data, solution: dict) -> tuple:
    """prepare Jss solution for plotting

    Args:
        data:
        solution: a

     Returns:
        job_start_time: start time of each job on each machine
        processing_time: processing duration of each job on each machine

    """
    job_start_time = defaultdict(list)
    processing_time = defaultdict(list)
    for j in range(data.num_jobs):
        job_start_time[j] = [solution[(j, i)] for i in
                             range(data.num_machines)]
        processing_time[j] = [
            data.task_duration[j, data.machine_task[(j, i)]] for i in
            range(data.num_machines)]
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
            if '#' in line[0] or len(line) <= 2:
                continue
            k += 1
            lint = list(map(int, line.split()))
            job_start_time[k] = lint[::2]
            processing_time[k] = lint[1::2]

    return job_start_time, processing_time


def plotjssp(job_start_times: dict, processing_time: dict) -> None:
    """This function plots job shop problem 
    Args:
        job_start_times: start time of each job on each machine
        processing_time: processing duration of each job on each machine
    """

    sols = np.array(list(job_start_times.values()))
    durs = np.array(list(processing_time.values()))
    solsT = sols.transpose()
    dursT = durs.transpose()
    n, m = sols.shape
    labels = ['machine' + str(i) for i in range(n)]
    category_names = ['job' + str(i) for i in range(m)]

    category_colors = plt.get_cmap('RdYlGn')(
        np.linspace(0.15, 0.85, sols.shape[1]))
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

    return


if __name__ == "__main__":
    try:
        filename = sys.argv[1]
    except:
        filename = "output/sol3_3.sol"

    sol, dur = read_solution(filename)

    plotjssp(sol, dur)
    plt.savefig('out.png')
    plt.show()
