import wget
import gzip
import os
import json
import numpy as np
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups
import pytreebank

import logging

log = logging.getLogger(__name__)


def load_data_from_url(config, cache_dir):

    LOAD_FUNCS_AND_ARGS = {
        "amazon": (load_amazon_5core, [config, cache_dir]),
        "20newsgroups": (load_20newsgroups, config),
        "sst5": (load_sst5, config),
    }
    load_func, args = LOAD_FUNCS_AND_ARGS[config.dataset_name]
    return load_func(args)


def load_amazon_5core(config, cache_dir=None):
    """Return closest version of Amazon Reviews Sports & Outdoors split from the paper
    Towards More Accurate Uncertainty Estimation In Text Classification.
    """
    texts, targets = [], []
    # get zipped dataset
    url = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Sports_and_Outdoors_5.json.gz"
    save_path = os.path.join(cache_dir, "amazon_5core.json.gz")
    # check if file already exists, load if not
    if not (os.path.isfile(save_path)):
        save_path = wget.download(url, out=save_path)
    # unzip it and extract data to arrays
    with gzip.open(save_path, "rb") as f:
        for line in f.readlines():
            data = json.loads(line)
            texts.append(data["reviewText"])
            targets.append(np.int64(data["overall"]))
    # to shift classes from 1-5 to 0-4
    targets = np.asarray(targets) - 1
    # split on train|val|test
    seed = getattr(config, "seed", 42)
    text_buf, text_eval, targ_buf, targ_eval = train_test_split(
        texts, targets, test_size=0.1, random_state=seed
    )
    text_train, text_val, targ_train, targ_val = train_test_split(
        text_buf, targ_buf, test_size=2.0 / 9.0, random_state=seed
    )
    amazon_train = {"label": targ_train, "text": text_train}
    amazon_eval = {"label": targ_eval, "text": text_eval}
    train_dataset = Dataset.from_dict(amazon_train)
    dev_dataset = test_dataset = Dataset.from_dict(amazon_eval)
    return train_dataset, dev_dataset, test_dataset, None


def load_20newsgroups(config):
    newsgroups_train = fetch_20newsgroups(subset="train")
    newsgroups_train = {
        "label": newsgroups_train["target"],
        "text": newsgroups_train["data"],
    }
    newsgroups_eval = fetch_20newsgroups(subset="test")
    newsgroups_eval = {
        "label": newsgroups_eval["target"],
        "text": newsgroups_eval["data"],
    }
    datasets = DatasetDict(
        {
            "train": Dataset.from_dict(newsgroups_train),
            "validation": Dataset.from_dict(newsgroups_eval),
        }
    )
    train_dataset = Dataset.from_dict(newsgroups_train)
    dev_dataset = test_dataset = Dataset.from_dict(newsgroups_eval)
    return train_dataset, dev_dataset, test_dataset, None


def load_sst5(config):
    dataset = pytreebank.load_sst()
    sst_datasets = {}
    for category in ["train", "test", "dev"]:
        df = {"text": [], "label": []}
        for item in dataset[category]:
            df["text"].append(item.to_labeled_lines()[0][1])
            df["label"].append(item.to_labeled_lines()[0][0])
        cat_name = category if category != "dev" else "validation"
        sst_datasets[cat_name] = Dataset.from_dict(df)
    dataset = DatasetDict(sst_datasets)
    train_dataset = sst_datasets["train"]
    dev_dataset = sst_datasets[cat_name]
    test_dataset = sst_datasets["test"]
    return train_dataset, dev_dataset, test_dataset, None


# def load_twitter_hso(config):
#
#     dataset = load_dataset('hate_speech_offensive', cache_dir=config.cache_dir)
#     df = dataset['train'].to_pandas()
#     annotators_count_cols = ['hate_speech_count', 'offensive_language_count', 'neither_count']
#
#     #split by ambiguity (for test select most ambiguous part by annotators disagreement)
#     df_test = df[df['count'] != df[annotators_count_cols].max(axis=1)].reset_index(drop=True)
#     df_train = df[df['count'] == df[annotators_count_cols].max(axis=1)].reset_index(drop=True)
#
#     train_dataset = {'label': df_train['class'],
#                      'text': df_train['tweet']}
#
#     eval_dataset = {'label': df_test['class'],
#                     'text': df_test['tweet']}
#
#     datasets = DatasetDict({'train': Dataset.from_dict(train_dataset),
#                             'validation': Dataset.from_dict(eval_dataset)})
#
#     return datasets
