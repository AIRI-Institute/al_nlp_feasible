import torch
from torch import optim
import torch.multiprocessing as mp

import dill
import numpy as np
from math import ceil

from tqdm import tqdm
import json
from pathlib import Path
from typing import List, Union
import logging
import os

log = logging.getLogger()


def point_to_device(point, device):
    if isinstance(point, torch.Tensor):
        return point.to(device)
    return {k: v.to(device) for k, v in point.items()}


def calculate_grad(model, optimizer, point):

    optimizer.zero_grad()

    point = point_to_device(point, next(model.parameters()).device)
    point = {k: v for k, v in point.items() if k in model.forward.__code__.co_varnames}
    loss = model(**point)["loss"]

    loss.backward()
    loss_gradient = torch.cat(
        [param.grad.view(-1) for param in model.parameters() if param.grad is not None]
    )
    return loss_gradient


def calculate_tracin_score_one_epoch(model, optimizer, train_point, test_point, nu=1):

    train_grad = calculate_grad(model, optimizer, train_point)
    test_grad = calculate_grad(model, optimizer, test_point)
    return nu * (train_grad @ test_grad)


def load_dataloader(dataloader_path):
    with open(dataloader_path, "rb") as f:
        dataloader = dill.load(f)
    return dataloader


def calculate_tracin_outlier_score_one_model(
    model_load_func,
    args: Union,  # args for model_load function
    weights_path,
    dataloader_path,
    work_dir,
    n_epoch,
    nu=1,
    cuda_device=0,
):

    log.info("Loading model and dataloader")
    model = model_load_func(*args)
    model.load_state_dict(torch.load(weights_path))
    model.cuda(cuda_device)

    dataloader = load_dataloader(dataloader_path)
    log.info("Done with loading model and dataloader")

    optimizer = optim.SGD(model.parameters(), lr=1)
    scores = []

    log.info("Start calculating TracIn scores")
    for i, point in enumerate(tqdm(dataloader)):
        point_to_device(point, 0)

        point_grad = calculate_grad(model, optimizer, point)
        point_epoch_score = nu * (point_grad @ point_grad.T)

        scores.append(float(point_epoch_score.item()))
    log.info("Done with calculating TracIn scores")

    with open(Path(work_dir) / f"scores_epoch_{n_epoch}.json", "w") as f:
        json.dump(scores, f)
    return scores


def load_model(model_path: str or Path) -> torch.nn.Module:
    with open(model_path, "rb") as f:
        model = dill.load(f)
    return model


def calculate_outlier_scores(
    model_path: str or Path,
    weights_paths: List[str or Path],
    dataloader_path: str or Path,
    work_dir: str or Path,
    max_num_processes: int = None,
    model_type: str = "ner",
    nu: float or int = 1,
):
    torch.cuda.empty_cache()
    mp.set_start_method("spawn", force=True)

    processes = []
    cuda_devices = list(map(int, os.environ["CUDA_VISIBLE_DEVICES"].split(",")))
    total_procs = len(weights_paths)
    num_cudas = len(cuda_devices)
    num_procs_per_cuda = min(max_num_processes, ceil(len(weights_paths) / num_cudas))
    num_procs_per_core = ceil(
        total_procs / (max_num_processes * num_cudas)
    )  # ~ num batches

    for n_proc_per_core in range(num_procs_per_core):
        for n_proc_per_cuda in range(num_procs_per_cuda):
            for i_cuda, cuda_device in enumerate(cuda_devices):
                n_epoch = (
                    i_cuda
                    + n_proc_per_cuda * num_cudas
                    + n_proc_per_core * (num_cudas * num_procs_per_cuda)
                )
                if n_epoch >= total_procs:
                    break
                log.info(
                    f"Starting process for epoch {n_epoch} on cuda device {cuda_device}, a process number "
                    f"{n_proc_per_cuda} on this cuda device."
                )
                p = mp.Process(
                    target=calculate_tracin_outlier_score_one_model,
                    args=(
                        load_model,
                        [model_path],
                        weights_paths[n_epoch],
                        dataloader_path,
                        work_dir,
                        n_epoch,
                        nu,
                        i_cuda,
                    ),
                )
                p.start()
                processes.append(p)
                torch.cuda.empty_cache()
        for p in processes:
            p.join()

    log.info(f"TracIn scores calculation done.")
    scores = []
    for n_epoch in range(total_procs):
        with open(Path(work_dir) / f"scores_epoch_{n_epoch}.json") as f:
            scores.append(json.load(f))

    if model_type == "ner":
        dataloader = load_dataloader(dataloader_path)
        lengths = [len(point["idx_first_bpe"][0]) for point in dataloader]
        final_scores = np.mean(scores, axis=0) / lengths
        return final_scores, scores, lengths

    elif model_type == "cls":
        final_scores = np.mean(scores, axis=0)
        return final_scores, scores
    else:
        raise NotImplementedError
