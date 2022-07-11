import yaml
from pathlib import Path
from omegaconf.dictconfig import DictConfig
import json
import logging
import time
from typing import List
from torch import randperm
from collections import defaultdict
import pickle
import numpy as np


log = logging.getLogger()
f_format_data = lambda data: [list(e) for e in zip(*data)]


def create_tmp_directory(storage_path, name):
    dir_path = Path(storage_path) / ("datasets/" + name + "_tmp")
    dir_path.mkdir(exist_ok=True)


def json_dump(obj, path):
    if isinstance(obj, np.ndarray):
        obj = obj.tolist()
    with open(path, "w") as f:
        json.dump(obj, f)


def json_load(path):
    with open(path) as f:
        obj = json.load(f)
    return obj


def pickle_dump(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def pickle_load(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj


def read_yaml(file_path):
    with open(file_path) as f:
        data = yaml.load(f, yaml.Loader)
    return data


def tensor_to_numpy(tensor):
    return tensor.cpu().detach().numpy()


def create_time_dict(time_dict_path, model_name, mode="w"):
    init_dict = {model_name + "_fit": [], model_name + "_predict": []}
    with open(time_dict_path, mode) as f:
        json.dump(init_dict, f)


def add_new_model_to_time_dict(time_dict_path, new_model_name):
    with open(time_dict_path) as f:
        time_dict = json.load(f)
    time_dict.update({new_model_name + "_fit": [], new_model_name + "_predict": []})
    with open(time_dict_path, "w") as f:
        json.dump(time_dict, f)


def get_target_model_checkpoints(config, framework="transformers"):
    checkpoints_path = Path(config.target_model.training.trainer_args.serialization_dir)
    models_paths = [
        x / "pytorch_model.bin"
        for x in checkpoints_path.iterdir()
        if str(x).split("/")[-1].startswith("checkpoint")
    ]
    return models_paths


def get_time_dict_path(config):

    initial_time_dict = {
        "acquisition_fit": [],
        "acquisition_predict": [],
        "successor_fit": [],
        "successor_predict": [],
    }
    if "target_model" in config:
        initial_time_dict["target_fit"] = []
        initial_time_dict["target_predict"] = []

    # Create time dir if it is not created
    Path(config.cache_dir).mkdir(exist_ok=True)

    acquisition_name = config.acquisition_model.name.replace("/", "-")
    successor_name = (
        config.successor_model.name.replace("/", "-")
        if config.successor_model
        else acquisition_name
    )
    seed = config.seed

    if "target_model" not in config:
        file_name = f"time_dict_{acquisition_name}_{successor_name}_{seed}.json"
    else:
        target_name = config.target_model.name.replace("/", "-")
        file_name = (
            f"time_dict_{acquisition_name}_{successor_name}_{target_name}_{seed}.json"
        )

    time_dict_path = Path(config.cache_dir) / file_name
    json_dump(initial_time_dict, time_dict_path)

    return time_dict_path


def get_time_dict_path_full_data(config):

    initial_time_dict = {"model_fit": [], "model_predict": []}

    # Create time dir if it is not created
    Path(config.cache_dir).mkdir(exist_ok=True)

    model_name = config.model.name.replace("/", "-")
    seed = config.seed

    file_name = f"time_dict_{model_name}_{seed}.json"

    time_dict_path = Path(config.cache_dir) / file_name
    json_dump(initial_time_dict, time_dict_path)

    return time_dict_path


def get_config_to_update(dataset_configs_path, dataset_name, model_name):
    dataset_config = read_yaml(f"{dataset_configs_path}/{dataset_name}/dataset.yaml")
    model_config = read_yaml(
        f"{dataset_configs_path}/{dataset_name}/model_configs/{model_name}.yaml"
    )
    if dataset_config is not None:
        dataset_config.update(model_config or {})
    else:
        dataset_config = {}

    return dataset_config


def initialize_metrics_dict(
    model_type: str,  # cls, ner, abs-sum, cv-cls
    work_dir: str or Path,
    model_name: str,
    framework: str = "transformers",
    evaluate_query: bool = False,
):
    """
    Create dict of metrics
    """
    metrics_dict = {}
    if framework == "allennlp":  # If framework == "transformers", dump the empty dict
        if model_type == "cls":
            keys = ["accuracy", "loss"]
        elif model_type == "ner":
            keys = [
                "accuracy",
                "accuracy3",
                "precision-overall",
                "recall-overall",
                "f1-measure-overall",
                "loss",
            ]
        else:  # Hence it is AutoNER
            keys = ["micro", "macro", "weighted"]

        for phase in ["train", "test"]:
            for key in keys:
                metrics_dict[phase + "_" + key] = []

    path_to_dump = Path(work_dir) / f"{model_name}_metrics.json"
    json_dump(metrics_dict, path_to_dump)
    if evaluate_query:
        path_to_dump = Path(work_dir) / f"{model_name}_evaluate_query_metrics.json"
        json_dump(metrics_dict, path_to_dump)


def get_metrics_dict(model_type):
    """
    Create dict of metrics
    """
    if model_type == "cls":
        keys = ["accuracy", "loss"]
    elif model_type == "ner":
        keys = [
            "accuracy",
            "accuracy3",
            "precision-overall",
            "recall-overall",
            "f1-measure-overall",
            "loss",
        ]
    else:  # Hence it is AutoNER
        keys = ["micro", "macro", "weighted"]

    metrics_dict = {}

    for phase in ["train", "test"]:
        for key in keys:
            metrics_dict[phase + "_" + key] = []

    return metrics_dict


def calculate_time_decorator(time_dict_path, step):
    def decorator(function):
        def wrapped(*args, **kwargs):
            start_time = time.time()

            result = function(*args, **kwargs)

            time_work = time.time() - start_time
            time_dict = json.load(time_dict_path)
            time_dict[step].append(time_work)
            json_dump(time_dict, time_dict_path)

            return result

        return wrapped

    return decorator


def log_config(log, conf, num_tabs=0):
    config = to_dict(conf)
    for key in config.keys():
        if isinstance(config[key], dict):
            log.info("\t" * num_tabs + key)
            log_config(log, config[key], num_tabs + 1)
        else:
            log.info("\t" * num_tabs + f"{key}: {config[key]}")


def to_dict(conf):
    config = dict(conf)
    if "hydra" in config.keys():
        del config["hydra"]
    for key in config.keys():
        if isinstance(config[key], DictConfig):
            config[key] = to_dict(config[key])
    config = dict(config)
    return config


def get_balanced_sample_indices(dataset, num_classes, n_per_digit=2) -> List[int]:
    """Given `target_classes` randomly sample `n_per_digit` for each of the `num_classes` classes."""
    permed_indices = randperm(len(dataset))

    if n_per_digit == 0:
        return []

    num_samples_by_class = defaultdict(int)
    initial_samples = []

    for i in range(len(permed_indices)):
        permed_index = int(permed_indices[i])
        label = dataset["label"][permed_index]
        index, target = permed_index, int(label)

        num_target_samples = num_samples_by_class[target]
        if num_target_samples == n_per_digit:
            continue

        initial_samples.append(index)
        num_samples_by_class[target] += 1

        if len(initial_samples) == num_classes * n_per_digit:
            break

    return initial_samples


def random_fixed_length_data_sampler(features):
    target_length = 5056
    indices = randperm(target_length + (-target_length % len(features)))
    indices = (indices[:target_length] % len(features)).tolist()
    new_features = features.select(indices)
    return new_features


class DictWithGetattr(dict):
    def __getattr__(self, item):
        return self.get(item, None)
