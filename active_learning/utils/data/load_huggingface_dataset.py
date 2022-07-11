from datasets import load_dataset
import logging
from pathlib import Path
from copy import deepcopy
import numpy as np

from .preprocessing import (
    preprocess_img,
    _add_id_column_to_datasets,
    _use_test_subset,
    _use_train_subset,
    _filter_quantiles,
    _multiply_data,
)


log = logging.getLogger()


class HuggingFaceDatasetsReader:
    def __init__(self, *dataset_args, cache_dir=None, use_auth_token=None):
        self.dataset = load_dataset(
            *dataset_args, cache_dir=cache_dir, use_auth_token=use_auth_token
        )

    def __call__(self, phase, text_name=None, label_name=None):
        dataset = self.dataset[phase]
        if text_name is not None and label_name is not None:
            dataset = dataset.remove_columns(
                [
                    x
                    for x in dataset.column_names
                    if x not in [text_name, label_name, "id"]
                ]
            )
        setattr(self, phase, dataset)
        return getattr(self, phase)


def load_huggingface_dataset(config, task, cache_dir=None):

    data_cache_dir = Path(cache_dir) / "data" if cache_dir is not None else None
    text_name = config.text_name
    label_name = config.label_name
    kwargs = {
        "cache_dir": data_cache_dir,
        "use_auth_token": config.get("use_auth_token", None),
    }

    hfdreader = (
        HuggingFaceDatasetsReader(config.dataset_name, **kwargs)
        if isinstance(config.dataset_name, str)
        else HuggingFaceDatasetsReader(*list(config.dataset_name), **kwargs)
    )

    if config.get("multiply_data", None) is not None:
        hfdreader = _multiply_data(hfdreader, config.multiply_data)

    train_dataset = hfdreader("train", text_name, label_name)
    test_dataset = None
    if "validation" in hfdreader.dataset.keys():
        dev_dataset = hfdreader("validation", text_name, label_name)
        # Since on GLUE we do not have gold labels for test data
        if "test" in hfdreader.dataset.keys() and ("glue" not in config.dataset_name):
            test_dataset = hfdreader("test", text_name, label_name)
    else:
        dev_dataset = hfdreader("test", text_name, label_name)

    log.info(f"Loaded train size: {len(train_dataset)}")
    log.info(f"Loaded dev size: {len(dev_dataset)}")
    if test_dataset is None:
        log.info("Dev dataset coincides with test dataset")
    else:
        log.info(f"Loaded test size: {len(test_dataset)}")

    if config.labels_to_remove is not None:
        train_dataset = train_dataset.filter(
            lambda x: x[label_name] not in config.labels_to_remove
        )
        dev_dataset = dev_dataset.filter(
            lambda x: x[label_name] not in config.labels_to_remove
        )
        if test_dataset is not None:
            test_dataset = test_dataset.filter(
                lambda x: x[label_name] not in config.labels_to_remove
            )

    if task == "cls" or task == "cnn_cls":
        id2label = {
            i: val for i, val in enumerate(train_dataset.features[label_name].names)
        }
    elif task == "ner":
        id2label = {
            i: val
            for i, val in enumerate(train_dataset.features[label_name].feature.names)
        }
    elif task == "cv_cls":
        id2label = {
            i: val for i, val in enumerate(train_dataset.features[label_name].names)
        }
        train_dataset = train_dataset.map(preprocess_img)
        dev_dataset = dev_dataset.map(preprocess_img)
        if test_dataset is not None:
            test_dataset = test_dataset.map(preprocess_img)
    elif task == "abs-sum":
        id2label = None
    else:
        raise NotImplementedError

    if getattr(config, "filter_quantiles", None) is not None:
        train_dataset = _filter_quantiles(
            train_dataset,
            config.filter_quantiles,
            cache_dir,
            text_name,
            config.tokenizer_name,
        )

    if getattr(config, "use_subset", None) is not None:
        train_dataset = _use_train_subset(
            train_dataset,
            config.use_subset,
            getattr(config, "seed", 42),
            task,
            label_name,
        )

    if ("id" not in train_dataset.column_names) and config.get("add_id_column", True):
        train_dataset, dev_dataset, test_dataset = _add_id_column_to_datasets(
            [train_dataset, dev_dataset, test_dataset]
        )

    if getattr(config, "use_test_subset", False):
        if test_dataset is None:
            test_dataset = dev_dataset
        test_dataset, subsample_idx = _use_test_subset(
            test_dataset,
            config.use_test_subset,
            getattr(config, "seed", 42),
            getattr(config, "subset_fixed_seed", False),
        )
        dev_dataset = dev_dataset.select(
            np.setdiff1d(np.arange(len(dev_dataset)), subsample_idx)
        )

    if test_dataset is None:
        test_dataset = dev_dataset

    return [train_dataset, dev_dataset, test_dataset, id2label]
