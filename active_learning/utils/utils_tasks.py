import multiprocessing as mp
from collections.abc import Iterable
from typing import List

import copy
from numpy import random
import logging
from functools import partial

from omegaconf import DictConfig

log = logging.getLogger(__name__)


CUDA_DEVICES = mp.Queue()
WORKER_CUDA_DEVICES = None


def initialize_worker(num_cuda_devices=1):
    global CUDA_DEVICES
    global WORKER_CUDA_DEVICES
    WORKER_CUDA_DEVICES = []
    for i in range(num_cuda_devices):
        WORKER_CUDA_DEVICES.append(str(CUDA_DEVICES.get()))
    log.info(f"Worker cuda devices: {','.join(WORKER_CUDA_DEVICES)}")


def repeat_tasks(tasks: List[DictConfig]) -> List[DictConfig]:
    rep_tasks: List[DictConfig] = []
    for task in tasks:
        if "seeds" in task:
            seeds = task.seeds
        elif "n_repeats" in tasks and task.n_repeats:
            seeds = random.randint(0, 2147483648, task.n_repeats).tolist()
        else:
            seeds = [random.randint(2147483648)]
        log.info(f"N repeats: {len(seeds)}")
        for i, seed in enumerate(seeds):
            new_task = DictConfig(task)
            new_task.name = f"{new_task.name}_rep{i}"
            new_task.repeat = f"rep{i}"
            new_task.args += f" seed={seed} +fixed_seed=True"
            if "ens_seeds" in task:
                ens_seeds_str = "[" + ",".join(str(e) for e in task.ens_seeds[i]) + "]"
                new_task.args += f" al.ens_seeds={ens_seeds_str}"
            rep_tasks.append(new_task)

    return rep_tasks


def run_tasks(config, f_task):
    global CUDA_QUEUE

    n_cudas_per_run = getattr(config, "cuda_devices_per_run", 1)

    if not isinstance(config.cuda_devices, Iterable):
        cuda_devices = [config.cuda_devices]
    else:
        cuda_devices = config.cuda_devices

    log.info(f"Cuda devices: {cuda_devices}")

    for cuda_device in cuda_devices:
        CUDA_DEVICES.put(cuda_device)

    log.info("All tasks: {}".format(str([t.name for t in config.tasks])))
    if "task_names" in config and config.task_names:
        task_names = config.task_names

        task_index = {t.name: t for t in config.tasks}

        tasks = []
        for task_name in task_names:
            task_name = task_name.split("@")
            if task_name[0] not in task_index:
                raise ValueError(f"Task name: {task_name[0]} is not in config file.")

            task = task_index[task_name[0]]
            if len(task_name) == 2:
                task.n_repeats = int(task_name[1])

            tasks.append(task)
    else:
        tasks = config.tasks

    log.info("Running tasks: {}".format(str([t.name for t in tasks])))
    log.info(f"Cuda devices per run: {n_cudas_per_run}")

    tasks = repeat_tasks(tasks)

    pool = mp.Pool(
        processes=len(cuda_devices) // n_cudas_per_run,
        initializer=partial(initialize_worker, num_cuda_devices=n_cudas_per_run),
    )
    try:
        pool.map(f_task, tasks)

        pool.close()
        pool.join()
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
