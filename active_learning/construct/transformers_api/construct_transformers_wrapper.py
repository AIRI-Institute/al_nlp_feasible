from pathlib import Path
from omegaconf.omegaconf import DictConfig
from typing import Union

from .create_transformers_model import create_transformers_model_tokenizer
from ...modal_wrapper.transformers_api.modal_transformers import (
    ModalTransformersWrapper,
)


def construct_transformers_wrapper(
    config: DictConfig,
    model_cfg: DictConfig,
    dev_data,
    id2label=None,
    name: str = "acquisition",
    time_dict_path: Path or str = None,
    default_data_config: Union[dict, DictConfig, None] = None,
    tokenize_dev_data: int = True,
    embeddings=None,
    word2idx=None,
) -> "ModalTransformersWrapper":

    cache_dir = Path(config.cache_dir) if config.cache_model_and_dataset else None
    num_labels = model_cfg.num_labels if id2label is None else len(id2label)
    model, tokenizer = create_transformers_model_tokenizer(
        model_cfg,
        id2label,
        config.seed,
        cache_dir=cache_dir,
        embeddings=embeddings,
        word2idx=word2idx,
    )
    if ("tracin" not in config) or (not config.tracin.use) or (name != "target"):
        num_checkpoints_to_save = 1
    else:
        num_checkpoints_to_save = config.tracin.num_model_checkpoints
    if default_data_config is None:
        default_data_config = getattr(config, "data", None)
    training_cfg = model_cfg.training

    modal_wrapper = ModalTransformersWrapper(
        model=model,
        tokenizer=tokenizer,
        model_config=model_cfg,
        num_labels=num_labels,
        task=model_cfg.type,
        id2label=id2label,
        default_data_config=default_data_config,
        name=name,
        dev_data=dev_data,
        shuffle_dev=training_cfg.shuffle_dev,
        dev_size=training_cfg.dev_size,
        seed=config.seed,
        trainer_kwargs=training_cfg.trainer_args,
        batch_size_kwargs=training_cfg.batch_size_args,
        optimizer_kwargs=training_cfg.optimizer_args,
        scheduler_kwargs=training_cfg.scheduler_args,
        time_dict_path=time_dict_path,
        cache_dir=config.cache_dir,
        cache_model=config.cache_model_and_dataset,
        num_checkpoints_to_save=num_checkpoints_to_save,
        tokenize_dev_data=tokenize_dev_data,
        embeddings=embeddings,
        word2idx=word2idx,
    )

    return modal_wrapper
