from omegaconf.omegaconf import DictConfig
from pathlib import Path


def construct_model(
    config: DictConfig,
    model_cfg: DictConfig,
    dev_data,
    framework: str = "transformers",
    labels_or_id2label=None,
    name: str = "acquisition",
    time_dict_path: Path or str = None,
    embeddings=None,
    word2idx=None,
):
    from .transformers_api.construct_transformers_wrapper import (
        construct_transformers_wrapper,
    )

    return construct_transformers_wrapper(
        config,
        model_cfg,
        dev_data,
        labels_or_id2label,
        name,
        time_dict_path,
        embeddings=embeddings,
        word2idx=word2idx,
    )
