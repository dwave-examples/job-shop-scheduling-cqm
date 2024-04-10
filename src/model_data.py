import json
import sys
from collections.abc import Iterable

sys.path.append('./src')
from utils.utils import read_instance, read_taillard_instance


class Task:
    """This class represents a task in a job shop scheduling problem.

    Attributes:
        job (str): the name of the job requiring the task
        resource (str): the name of the resource to use for the task
        duration (int): how many time periods the task will take to complete
    """    
    def __init__(self, job: str, resource: str, duration: int):
        """Initializes a task in a job shop scheduling problem.

        Args:
            job (str): the name of the job requiring the task
            resource (str): the name of the resource to use for the task
            duration (int): how many time periods the task will take to complete
        """        
        self.job = job
        self.resource = resource
        self.duration = duration

    def __repr__(self) -> str:
        return ("{{job: {job}, resource: {resource}, duration:"
                " {duration}}}").format(**vars(self))


class JobShopData:
    """This class holds and manages the data for a job shop scheduling problem.
    """    

    def __init__(self, jobs: Iterable[str] = [], resources: Iterable[str] = []):
        """Initializes the data for a job shop scheduling problem.

        Args:
            jobs (Iterable[str]): The jobs to be scheduled.
            resources (Iterable[str]): The resources to be used.
        """        
        self._jobs = set(jobs)
        self._resources = set(resources)
        self._job_tasks = {job: [] for job in jobs}

    
    @property
    def jobs(self) -> Iterable[str]:
        """Returns the jobs in the data.

        Returns:
            Iterable[str]: The jobs in the data.
        """
        return self._jobs
    

    @property
    def resources(self) -> Iterable[str]:
        """Returns the resources in the data.

        Returns:
            Iterable[str]: The resources in the data.
        """
        return self._resources
    

    @property
    def job_tasks(self) -> dict:
        """Returns the tasks in the data, grouped by job.

        Returns:
            Iterable[Iterable[Task]]: The tasks in the data, grouped by job.
        """
        return self._job_tasks
    

    def get_tasks(self) -> Iterable[Task]:
        """Returns the tasks in the data.

        Returns:
            Iterable[Task]: The tasks in the data.
        """
        return [task for job_tasks in self._job_tasks.values() for task in job_tasks]
    

    def get_last_tasks(self) -> Iterable[Task]:
        """Returns the last task in each job.

        Returns:
            Iterable[Task]: The last task in each job.
        """
        return [job_tasks[-1] for job_tasks in self._job_tasks.values()]


    def add_job(self, job: str) -> None:
        """Adds a job to the data.

        Args:
            job (str): The job to be added.
        """
        if job in self._jobs:
            raise ValueError(f"Job {job} already in dataset")        
        self._jobs.add(job)
        self._job_tasks[job] = []

    
    def add_resource(self, resource: str) -> None:
        """Adds a resource to the data.

        Args:
            resource (str): The resource to be added.
        """ 
        self._resources.add(resource)

    
    def add_task(self, resource: str, job: str, duration: int, position: int = None) -> None:
        """Adds a task to the dataset.

        Args:
            resource (str): the name of the resource to use for the task
            job (str): the name of the job requiring the task
            duration (int): how many time periods the task will take to
                complete
            position (int): the position of the task in the job's schedule;
                if None, then will put this task at the end

        Raises:
            ValueError: if the resource or job is not in the dataset
        """
        self.add_task(Task(job, resource, duration), position=position)

    
    def add_task(self, task: Task, position: int = None) -> None:
        """Adds a task to the dataset. If the task job or resource is not
        in the current dataset, then it will be added.

        Args:
            task (Task): the task to be added
        
        Raises:
            ValueError: if the resource position or duration is negative
        """
        if task.resource not in self._resources:
            self.add_resource(task.resource)
        
        if task.job not in self._jobs:
            self.add_job(task.job)
        
        if task.duration < 0:
            raise ValueError(f"Duration {task.duration} must be positive")
        
        if position is None:
            position = len(self._job_tasks[task.job])

        if position < 0:
            raise ValueError(f"Position {position} must be non-negative")
        
        self._job_tasks[task.job].insert(position, task)

    
    def remove_task(self, task: Task) -> None:
        """Removes a task from the dataset.

        Args:
            task (Task): the task to be removed
        """
        self._job_tasks[task.job].remove(task)

    
    def remove_job(self, job: str) -> None:
        """Removes a job from the dataset.

        Args:
            job (str): the job to be removed
        """
        self._jobs.remove(job)
        del self._job_tasks[job]

    
    def get_task_position(self, task: Task) -> int:
        """Returns the position of a task in its job's schedule.

        Args:
            task (Task): the task to be checked
        
        Returns:
            int: the position of the task in its job's schedule

        Raises:
            ValueError: if the task is not in the dataset
        """
        if task.job not in self._jobs:
            raise ValueError(f"Job {task.job} not in dataset")

        return self._job_tasks[task.job].index(task)
    

    def get_resource_job_tasks(self, resource: str, job: str) -> Task:
        """Returns the tasks in a job that require a given resource.

        Args:
            resource (str): the resource to be checked
            job (str): the job to be checked
        
        Returns:
            Task: the tasks in the job that require the given resource

        Raises:
            ValueError: if the resource or job is not in the dataset
        """
        if resource not in self._resources:
            raise ValueError(f"Resource {resource} not in dataset")

        if job not in self._jobs:
            raise ValueError(f"Job {job} not in dataset")

        return [task for task in self._job_tasks[job] if task.resource == resource][0]


    def get_total_job_time(self, job: str) -> int:
        """Returns the total time needed to complete a job.

        Args:
            job (str): the job to be checked
        
        Returns:
            int: the total time needed to complete the job

        Raises:
            ValueError: if the job is not in the dataset
        """
        if job not in self._jobs:
            raise ValueError(f"Job {job} not in dataset")

        return sum(task.duration for task in self._job_tasks[job])
    

    def get_maximum_job_time(self) -> int:
        """Returns the maximum time needed to complete any job.

        Returns:
            int: the maximum time needed to complete any job
        """
        return max(self.get_total_job_time(job) for job in self._jobs)
    

    def get_max_makespan(self) -> int:
        """Returns the maximum time needed to complete all jobs

        Returns:
            int: the maximum time needed to complete any job
        """
        return sum(self.get_total_job_time(job) for job in self._jobs)
    

    def get_job_count(self) -> int:
        """Returns the number of jobs in the dataset.

        Returns:
            int: the number of jobs in the dataset
        """
        return len(self._jobs)
    

    def get_resource_count(self) -> int:
        """Returns the number of resources in the dataset.

        Returns:
            int: the number of resources in the dataset
        """
        return len(self._resources)
    

    def get_task_time_bounds(self, task, make_span: int) -> tuple[int, int]:
        """Returns the minimum and maximum possible starting
        times for this task, given that prior tasks for the job
        must be completed first and subsequent tasks for the job
        must be completed after.
        
        Args:
            make_span (int): the maximum time needed to complete all jobs
        """
        job_tasks = [x for x in self.get_tasks() if x.job == task.job]
        task_position = self.get_task_position(task)
        prior_tasks = job_tasks[:task_position]
        subsequent_tasks = job_tasks[task_position + 1:]
        prior_time = sum(x.duration for x in prior_tasks)
        subsequent_time = sum(x.duration for x in subsequent_tasks)
        return (prior_time, make_span - subsequent_time - task.duration)
    

    def get_resource_tasks(self, resource: str) -> list[Task]:
        """This function returns all Tasks that require a given resource.

        Args:
            resource (str): the resource to be checked

        Returns:
            list[Task]: all Tasks that require the given resource
        """        
        return [task for job_tasks in self._job_tasks.values() for task in job_tasks if task.resource == resource]


    def load_from_json(self, json_file: str, resource_names: list=None) -> None:
        """Loads data from a JSON file.

        Args:
            json_file (str): the JSON file to load data from
        """
        self.__init__()        
        jobs = json.load(open(json_file, 'r'))
        self.load_from_dict(jobs, resource_names=resource_names)

    
    def load_from_dict(self, jobs: dict, resource_names: list=None) -> None:
        """Loads data from a dictionary.

        Args:
            jobs (dict): the dictionary to load data from
            resource_names: the names of the resources to be used; if you
                want to change the resource names from the ones in the
                input dictionary. If None, then will use the resource
                names in the input dictionary. If there are more resources
                in the input dictionary than in this list, then the extra
                resources will have a suffix append to the name.
        """
        self.__init__()
        resource_mapping = {}    
        unique_resource_num = 0    
        for job, task_list in jobs.items():
            for (resource, duration) in task_list:        
                if resource_names is not None:
                    if resource not in resource_mapping:
                        quotient, remainder = divmod(unique_resource_num, len(resource_names))
                        resource_name = resource_names[remainder]
                        if quotient > 0:
                            resource_name += '_' + str(quotient)
                        resource_mapping[resource] = resource_name
                        unique_resource_num += 1
                        resource_mapping[resource] = resource_name
                    else:
                        resource_name = resource_mapping[resource]
                else:
                    resource_name = resource
                self.add_task(Task(str(job), duration=duration, resource=resource_name))


    def load_from_file(self, filename: str, resource_names: list=None) -> None:
        """Loads data from a file.

        Args:
            filename (str): the file to load data from
        """
        if 'taillard' in str(filename):
            job_dict = read_taillard_instance(filename)
        else:
            job_dict = read_instance(filename)

        self.load_from_dict(job_dict, resource_names=resource_names)