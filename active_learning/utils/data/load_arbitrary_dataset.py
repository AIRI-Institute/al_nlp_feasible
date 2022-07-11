from datasets import load_dataset
import logging
import itertools
from pathlib import Path
from copy import deepcopy
from sklearn.model_selection import train_test_split

from ..transformers_dataset import TransformersDataset

log = logging.getLogger()


def load_arbitrary_dataset_for_cls(config):
    """
    At the moment, only datasets splitted by folders are accepted with no train/test division.
    The dataset is splitted inside this function.
    """
    path = Path(config.path) / config.dataset_name
    texts, labels, unique_labels = [], [], []
    for folder in path.iterdir():
        if folder.is_dir():
            label = str(folder).split("/")[-1]
            for file in folder.iterdir():
                with open(file, "r", encoding="utf-8", errors="ignore") as f:
                    text = "".join(list(f))
                texts.append(text)
                labels.append(label)
                if label not in unique_labels:
                    unique_labels.append(label)

    label2id = {k: i for i, k in enumerate(unique_labels)}
    id2label = {v: k for k, v in label2id.items()}

    labels_encoded = [label2id[lab] for lab in labels]
    data = {config.text_name: texts, config.label_name: labels_encoded}
    dataset = TransformersDataset(data)

    splitted_dataset = dataset.train_test_split(
        train_size=config.get("train_size_split", 0.8),
        shuffle=True,
        seed=config.get("seed", 42),
    )
    train_dataset, test_dataset = splitted_dataset["train"], splitted_dataset["test"]
    # Deepcopy to make a dev set as a copy of the test set
    return train_dataset, deepcopy(test_dataset), test_dataset, id2label


def _is_divider(line: str) -> bool:
    empty_line = line.strip() == ""
    if empty_line:
        return True
    else:
        first_token = line.split()[0]
        if first_token == "-DOCSTART-":
            return True
        else:
            return False


def load_arbitrary_dataset_for_ner(config):
    ### Need to have files `train.txt`, `dev.txt` and `test.txt` stored in the `config.path` folder
    file_names = ["train.txt", "dev.txt", "test.txt"]
    path = Path(config.path) / config.dataset_name
    tokens_key = config.text_name
    tags_key = config.label_name
    processed_datasets = []
    unique_labels = {"O"}
    label2id = {"O": 0}

    for file_name in file_names:
        instances = []
        with open(path / file_name) as f:
            log.info("Reading instances from lines in file at: %s", file_name)
            tokens, ner_tags = [], []
            # Group into alternative divider / sentence chunks.
            for is_divider, lines in itertools.groupby(f, _is_divider):
                # Ignore the divider chunks, so that `lines` corresponds to the words
                # of a single sentence.
                if not is_divider:
                    fields = [line.strip().split() for line in lines]
                    fields = [field for field in zip(*fields)]
                    tokens = list(fields[0])
                    if config.tag_index >= 0:
                        ner_tags = fields[1:][config.tag_index]
                        # Update the unique tags
                        unique_labels.update(ner_tags)
                        # Update label2id of necessary
                        if unique_labels != set(label2id.keys()):
                            for key in sorted(unique_labels):
                                if key not in label2id:
                                    label2id[key] = len(label2id)
                        # Encode the tags
                        replacer = label2id.get
                        encoded_tags = [replacer(tag, tag) for tag in ner_tags]
                    else:
                        encoded_tags = [0 for _ in range(len(fields[0]))]
                    # Add the queried instance
                    instances.append({tokens_key: tokens, tags_key: encoded_tags})

        # Construct `id2label` with respect to `tag2idx`
        id2label = {i: tag for tag, i in label2id.items()}
        dataset = TransformersDataset(
            instances,
            text_column_name=tokens_key,
            label_column_name=tags_key,
            task="ner",
            id2label=id2label,
        )
        processed_datasets.append(dataset)
    return processed_datasets + [id2label]


def load_arbitrary_dataset_for_ner_from_ls(text, tag_idx):
    processed_datasets = []
    unique_tags = {"O"}
    tag2idx = {"O": 0}

    instances = []
    # log.info("Reading instances from lines in file at: %s", file_name)
    # Group into alternative divider / sentence chunks.
    for is_divider, lines in itertools.groupby(text.splitlines(), _is_divider):
        # Ignore the divider chunks, so that `lines` corresponds to the words
        # of a single sentence.
        if not is_divider:
            fields = [line.strip().split() for line in lines]
            fields = [field for field in zip(*fields)]
            tokens = list(fields[0])
            if tag_idx >= 0:
                ner_tags = fields[1:][tag_idx]
                # Update the unique tags
                unique_tags.update(ner_tags)
                # Update tag2idx of necessary
                if unique_tags != set(tag2idx.keys()):
                    for key in sorted(unique_tags):
                        if key not in tag2idx:
                            tag2idx[key] = len(tag2idx)
                # Encode the tags
                replacer = tag2idx.get
                encoded_tags = [replacer(tag, tag) for tag in ner_tags]
            else:
                encoded_tags = [0 for _ in range(len(fields[0]))]
            # Add the queried instance
            instances.append({"tokens": tokens, "ner_tags": encoded_tags})

    # Construct `id2label` with respect to `tag2idx`
    id2label = {i: tag for tag, i in tag2idx.items()}
    dataset = TransformersDataset(
        instances,
        text_column_name="tokens",
        label_column_name="ner_tags",
        task="ner",
        id2label=id2label,
    )
    processed_datasets.append(dataset)
    return processed_datasets + [id2label]
