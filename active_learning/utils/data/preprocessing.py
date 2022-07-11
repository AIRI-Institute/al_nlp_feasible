from torch import tensor
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from datasets import concatenate_datasets
from pathlib import Path


def preprocess_img(example):
    data = tensor(np.array(example["image"]))
    data = data.float().div(255)
    data = data.sub_(0.1307).div_(0.3081)
    data = list(data.numpy())
    example["image"] = data
    return example


def _add_id_column_to_datasets(datasets):
    """
    :param datasets: should be strictly in order `train - dev - test (if test exists)`
    :return:
    """
    last_id = 0
    for i in range(len(datasets)):
        dataset = datasets[i]
        if dataset is not None:
            datasets[i] = dataset.add_column(
                "id", list(range(last_id, last_id + len(dataset)))
            )
            last_id += len(dataset)
        return datasets


def _use_train_subset(train_dataset, subset_size, seed, task, label_name):
    kwargs = {"random_state": seed}
    if task.endswith("cls"):
        kwargs["stratify"] = train_dataset[label_name]
    random_train_idx = train_test_split(
        range(len(train_dataset)), train_size=subset_size, **kwargs
    )[0]
    return train_dataset.select(random_train_idx)


def _use_test_subset(test_dataset, subset_share_or_size, seed=42, use_seed_42=False):
    if isinstance(subset_share_or_size, float):
        subset_size = round(subset_share_or_size * len(test_dataset))
    else:
        subset_size = subset_share_or_size
    if use_seed_42:
        np.random.seed(42)
    else:
        np.random.seed(seed)
    subsample_idx = np.random.choice(range(len(test_dataset)), subset_size, False)
    return test_dataset.select(subsample_idx), subsample_idx


def _filter_quantiles(train_dataset, quantiles, cache_dir, text_name, tokenizer_name):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        cache_dir=Path(cache_dir) / "tokenizer" if cache_dir is not None else None,
    )
    train_lengths = np.array(
        [
            len(x)
            for x in tokenizer(train_dataset[text_name], truncation=False)["input_ids"]
        ]
    )
    length_quantiles = np.quantile(train_lengths, quantiles)
    ids_satisfy_cond = np.argwhere(
        (train_lengths >= length_quantiles[0]) & (train_lengths <= length_quantiles[1])
    ).ravel()
    return train_dataset.select(ids_satisfy_cond)


def _multiply_data(hfdreader, multiply_coef):
    if type(hfdreader.dataset.num_rows) == int:
        hfdreader.dataset = concatenate_datasets([hfdreader.dataset] * multiply_coef)
    else:
        # we have dataset with several splits
        for key in hfdreader.dataset.num_rows.keys():
            if key != "test":
                hfdreader.dataset[key] = concatenate_datasets(
                    [hfdreader.dataset[key]] * multiply_coef
                )
    return hfdreader
