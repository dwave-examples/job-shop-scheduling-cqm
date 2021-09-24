from utils.utils import read_instance


class Data:
    """ A class that reads and prepare data for job shop scheduling problem."""

    def __init__(self):
        self.num_jobs = 0
        self.num_machines = 0
        # Upper bound for maximum make span
        self.MaxMakeSpan = 0
        self.task_duration = {}
        self.task_machine = {}
        self.machine_task = {}

    def read_input_data(self, filename):
        # Read input file
        jobs = read_instance(filename)

        for j, val in jobs.items():
            for i, (machine, duration) in enumerate(val):
                self.task_machine[(j, i)] = machine
                self.task_duration[(j, i)] = duration
                self.machine_task[(j, machine)] = i

        # Get number of jobs
        self.num_jobs = len(jobs)

        # Get number of machines
        self.num_machines = len(jobs[0])

        # Calculate a trivial upper bound for make span
        self.MaxMakeSpan = sum(self.task_duration[(j, i)]
                               for j in range(self.num_jobs) for i in
                               range(self.num_machines))
