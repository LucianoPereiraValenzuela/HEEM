from typing import List, Tuple, Union, Optional
import numpy as np
import networkx as nx
import copy
import joblib
import contextlib
from tqdm.auto import tqdm
import time

from qiskit import execute
from qiskit.circuit import QuantumCircuit
from qiskit.providers import JobStatus
from qiskit.providers.ibmq.job import IBMQJob
from qiskit.providers.ibmq import IBMQBackend
from qiskit.opflow.primitive_ops import TaperedPauliSumOp, PauliSumOp


def _question(name: Optional[str] = None, message: Optional[str] = None) -> bool:
    """
    Make a question and return True or False depending on the answer of the user. There is only two possible answers
    y -> yes or	n -> no. If the answer is none of this two the question is repeated until a good answer is given.
    If not message is provided, then the default overwriting file message is printed, with the file name provided.

    Parameter
    ----------
    name: str (optional, default=None)
        Name of the file to overwrite
    message: str (optional, default=None)
        Message to print
    Return
    ------
    (Bool)
        Answer given by the user
    """
    if message is None:
        message = 'Do you want to overwrite the file ({})?'.format(name)

    temp = input(message + '  [y]/n:').lower()  # Ask for an answer by keyword input

    if temp == 'y' or temp == '':
        return True
    elif temp == 'n':
        return False
    else:  # If the answer is not correct
        print('I didn\'t understand your answer.')
        return _question(name=name, message=message)  # The function will repeat until a correct answer if provided


def random_labels(n: int, N: int) -> np.ndarray:
    """
    Return a total of n Pauli strings of N qubits.
    """
    labels = [np.random.randint(4, size=N) for _ in range(n)]
    return np.array(labels)


def string2number(labels: List[str]) -> np.ndarray:
    mapping = {'I': 0, 'X': 1, 'Y': 2, 'Z': 3}

    n = len(labels)  # Number of labels
    N = len(labels[0])  # Number of qubits

    labels_temp = copy.copy(labels)
    labels = np.zeros((n, N), dtype=np.int8)  # Since the indices < 256, use int8 to reduce memory usage
    for i in range(n):
        for j in range(N):
            labels[i, j] = mapping[labels_temp[i][j]]

    return labels


def number2string(labels: np.ndarray) -> List[str]:
    """
    Return the Pauli labels from the number convention , e.g. [0, 2, 1, 3], to the string convention 'IYXZ'.
    """
    mapping = ['I', 'X', 'Y', 'Z']

    # If only one label is provided, then reshape to (1, N)
    shape = np.shape(labels)
    if len(shape) == 2:
        n, N = shape
    else:
        labels = labels.reshape(1, -1)
        n, N = np.shape(labels)

    labels_string = []
    for i in range(n):
        string = ''
        for j in range(N):
            string += mapping[labels[i, j]]
        labels_string.append(string)
    return labels_string


def add_edge(G: nx.Graph, node1: int, node2: int) -> None:
    """
    Add an edge in the graph between node1 and node2 with width equals to one. If the edge already exists, then the
    weight increases by one.
    """
    if node2 < node1:
        node1, node2 = node2, node2  # Ensure that node1 < node2

    if node1 != node2:  # Do not include self edges
        edges = list(G.edges())
        if (node1, node2) in edges:
            last_weight = nx.get_edge_attributes(G, "weight")[(node1, node2)]
        else:
            last_weight = 0

        G.add_edge(node1, node2, weight=last_weight + 1)
    else:
        G.add_node(node1)


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """
    Context manager to patch joblib to report into tqdm progress bar given as argument. Code extracted from the
    stack overflow response: https://stackoverflow.com/a/58936697/8237804
    """

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def _check_status_jobs(running: List[Tuple[int, IBMQJob]], n_batches: int, verbose: Optional[bool] = False) -> Union[
    None, Tuple[bool, int, int]]:
    """
    Check is some job is completed in order to send the next one. If some job has raised an error, then return it so
    is sent again.
    Parameters
    ----------
    running: list(tuple(int, IBMQJob)
        List with the running jobs. Each element is a tuple where the first element is the job index, and the second one
        the job itself.
    n_batches: int
        Total number of jobs to be sent. This number is only used if verbose=True
    verbose: bool (optional, default=None)
        If true, print when a job is completed, an error raised, of the job have been cancelled.
    Returns
    -------
    completed: bool
        If the job have been completed, then completed=True. If the job is cancelled or raised some error,
        completed=False. Otherwise, completed=None.
    job_index: int
        Index of the completed/cancelled job
    queue_index: int
        Index of the completed/cancelled job within the running jobs.
    """
    for j, (index, job) in enumerate(running):
        if verbose:
            print(f'Checking job {index + 1}/{n_batches}')

        status = job.status()
        if status == JobStatus.DONE:
            if verbose:
                print(f'Job {index + 1}/{n_batches} done')
            return True, j, index
        elif status == JobStatus.CANCELLED:
            if verbose:
                print(f'Job {index + 1}/{n_batches} cancelled')
            return False, j, index
        elif status == JobStatus.ERROR:
            if verbose:
                print(f'Job {index + 1}/{n_batches} failed')
            return False, j, index


def _send_job_backend(backend: IBMQBackend, circuits_batch: List[QuantumCircuit], index: Optional[int] = 0,
                      n_batches: Optional[int] = 1, job_tags: Optional[Union[str, List[str]]] = None,
                      verbose: Optional[bool] = True, **kwargs_run: dict) -> IBMQJob:
    """
    Send a job to ibm with a series of circuits.
    Parameters
    ----------
    backend: IBMQBackend
        IBMQ backend to send the job
    circuits_batch: list(QuantumCircuits)
        Circuits to execute in the backend
    index: int (optional, default=0)
        If multiple jobs are going to be sent to IBM, then use the name of the job to write the job index.
    n_batches: int (optional, default=1)
        if multiple jobs are going to be sent to IBM, then use the name of the job to write the total number of jobs.
    kwargs_run: dict (optional, default=None)
        Optional arguments for execute the circuits
    job_tags: str (optional, default=None)
        Job tag.
    verbose: bool (optional, default=True)
        If True, print when the job is sent, together with the job index and the total number of jobs to be sent.

    Returns
    -------
    job: IBMQJob
        Job sent to IBMQ.
    """

    if job_tags is not None and type(job_tags) is not list:
        job_tags = [job_tags]

    job_name = '{}/{}'.format(index + 1, n_batches)

    job = execute(circuits_batch, backend, **{'job_name': job_name, 'job_tags': job_tags}, **kwargs_run)

    if verbose:
        print('Job {}/{} sent to IBMQ'.format(index + 1, n_batches))

    return job


def send_ibmq_parallel(backend: IBMQBackend, circuits: List[QuantumCircuit],
                       job_tags: Optional[Union[str, List[str]]] = None, verbose: Optional[bool] = False,
                       progress_bar: Optional[bool] = False, circuits_batch_size: Optional[int] = 300,
                       n_jobs_parallel: Optional[int] = 5, waiting_time: Optional[int] = 20, **kwargs_run: dict) -> \
        List[IBMQJob]:
    """
    Send to IBMQ backed multiples jobs backends, so they run in parallel. The circuits are divided in multiples batches.

    Parameters
    ----------
    backend: IBMQBackend
        IBMQ backend to send the job
    circuits: list(QuantumCircuit)
        Circuits to be run
    job_tags: str or list(str) (optional, default=None)
        Tag for the job. If multiple tags, they can be added in a list.
    verbose: bool (optional, default=True)
        If True, write the progress as the jobs are sent, completed, cancelled, ...
    progress_bar: bool (optional, default=False)
        If True, print a progress bar as the jobs are completed.
    circuits_batch_size: int (optional, default=300)
        Maximum number of circuits in the batches. This can be limited by the backend options.
    n_jobs_parallel: int (optional, default=5)
        Maximum number of jobs to be run in parallel.
    waiting_time: int (optional, default=20
        Waiting time (in sec) to wait between petitions to IBMQ for job monitor.
    kwargs_run: dict (optional)
        Extra optional arguments for the backend run.

    Returns
    -------
    job_done: list(IBMQJob)
        When all jobs are completed, they are returned in a sorted list. The circuits are located as follows:
        job_1 -> [circuits[0], circuits[1], ..., circuits[circuits_batch_size - 1]]
        job_2 -> [circuits[circuits_batch_size], ..., circuits[2 * circuits_batch_size - 1]]
        .
        .
        .
        job_-1 -> [..., circuits[-1]]
    """

    n_active_jobs = backend.job_limit().active_jobs
    if n_active_jobs != 0:
        delete_jobs = _question(
            message='There are {} active jobs in the backed. Do you want to cancel them?'.format(n_active_jobs))

        if delete_jobs:
            for job in backend.active_jobs(limit=n_active_jobs):
                job.cancel()

    max_experiments = backend.configuration().max_experiments
    circuits_batch_size = min(circuits_batch_size, max_experiments)

    n_circuits = len(circuits)
    n_batches = int(np.ceil(n_circuits / circuits_batch_size))  # Number of circuits batches to sent

    max_jobs = backend.job_limit().maximum_jobs
    if n_jobs_parallel == -1:
        n_jobs_parallel = n_batches

    n_jobs_parallel = min(max_jobs, n_jobs_parallel)

    pbar = tqdm(total=n_batches, desc='Jobs completed', disable=not progress_bar)

    # Indices for the first and last circuit in each batch
    indices = []
    for i in range(n_batches):
        initial = i * circuits_batch_size
        final = min(initial + circuits_batch_size, n_circuits)

        indices.append((initial, final))

    job_done = []
    running_jobs = []

    # Send the initial jobs
    for i, index in enumerate(indices[:n_jobs_parallel]):
        job = _send_job_backend(backend, circuits[index[0]:index[1]], index=i, n_batches=n_batches, job_tags=job_tags,
                                verbose=verbose, **kwargs_run)
        running_jobs.append((i, job))

    max_id_sent = n_jobs_parallel - 1

    while len(running_jobs) > 0:  # Iterate until all jobs are completed
        job_checker = _check_status_jobs(running_jobs, n_batches, verbose=verbose)
        if verbose:
            print('Finish checking')

        if job_checker is not None:  # If some job has been completed or cancelled
            done, running_id, job_id = job_checker
            if done:
                pbar.update()
                job_done.append(running_jobs.pop(running_id))

                if max_id_sent + 1 < n_batches:  # If there is still some job to be sent
                    max_id_sent += 1
                    job = _send_job_backend(backend, circuits[indices[max_id_sent][0]:indices[max_id_sent][1]],
                                            index=max_id_sent, n_batches=n_batches, job_tags=job_tags, verbose=verbose,
                                            **kwargs_run)

                    running_jobs.append((max_id_sent, job))
            else:  # Resend the failed job
                running_jobs.pop(running_id)
                job = _send_job_backend(backend, circuits[indices[job_id][0]:indices[job_id][1]], index=job_id,
                                        n_batches=n_batches, job_tags=job_tags, verbose=verbose, **kwargs_run)
                running_jobs.append((job_id, job))
        else:
            time.sleep(waiting_time)  # Wait some time for the next status, so IBMQ do not detect lots of petitions
    pbar.close()

    job_done = [x[1] for x in sorted(job_done, key=lambda x: x[0])]  # Sort the jobs using their job_id

    return job_done


def extract_paulis(qubit_op: Union[PauliSumOp, TaperedPauliSumOp]) -> Tuple[List[str], List[complex]]:
    """
    Extract the Pauli labels and the coefficients from a given qubit operator.
    Parameters
    ----------
    qubit_op: TaperedPauliSumOp:
        Qubit operator to which extract the information

    Returns
    -------
    labels: list[str]
        Pauli labels in the string convention
    coeffs: list[complex]
        Coefficient for each of the Pauli strings
    """

    qubit_op_data = qubit_op.primitive.to_list()
    labels = [data[0] for data in qubit_op_data]
    coeffs = [data[1] for data in qubit_op_data]

    return labels, coeffs
