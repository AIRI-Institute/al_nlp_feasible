import os
from pathlib import Path
from shutil import rmtree
import hydra
from omegaconf import OmegaConf
import logging

os.environ["WANDB_DISABLED"] = "true"

log = logging.getLogger()

OmegaConf.register_new_resolver(
    "to_string", lambda x: x.replace("/", "_").replace("-", "_")
)
OmegaConf.register_new_resolver(
    "get_patience_value", lambda dev_size: 1000 if dev_size == 0 else 3
)


@hydra.main(
    config_path=os.environ["HYDRA_CONFIG_PATH"],
    config_name=os.environ["HYDRA_CONFIG_NAME"],
)
def main(config):
    # Set the working directory
    auto_generated_dir = os.getcwd()
    os.chdir(hydra.utils.get_original_cwd())
    # Enable offline mode for HuggingFace libraries if necessary
    if config.offline_mode:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        start_word = "Loading cached"
        end_word = "loaded"
    else:
        start_word = "Caching"
        end_word = "cached"
    # Cache everything
    cache_all(config, start_word, end_word)
    # Remove unused directory
    rmtree(auto_generated_dir)


def cache_model_and_tokenizer(task, model_name, cache_dir=None, end_word="cached"):

    from transformers import (
        AutoModelForSequenceClassification,
        AutoModelForTokenClassification,
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
    )

    if cache_dir is not None:
        model_cache_dir = cache_dir / "model"
        tokenizer_cache_dir = cache_dir / "tokenizer"
    else:
        model_cache_dir = None
        tokenizer_cache_dir = None

    if task == "cls":
        AutoModelForSequenceClassification.from_pretrained(
            model_name, cache_dir=model_cache_dir
        )
    elif task == "ner":
        AutoModelForTokenClassification.from_pretrained(
            model_name, cache_dir=model_cache_dir
        )
    elif task == "abs-sum":
        AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=model_cache_dir)
    log.info("=" * 10 + f" Model {model_name} successfully {end_word} " + "=" * 10)
    AutoTokenizer.from_pretrained(model_name, cache_dir=tokenizer_cache_dir)
    log.info("=" * 10 + f" Tokenizer {model_name} successfully {end_word} " + "=" * 10)


def cache_all(config, start_word="Caching", end_word="cached"):

    from datasets import load_dataset, load_metric

    cache_dir = Path(config.cache_dir) if config.cache_model_and_dataset else None
    # Cache models and tokenizers
    log.info("." * 10 + f" {start_word} model and tokenizer " + "." * 10)
    cache_model_and_tokenizer(
        config.acquisition_model.type,
        config.acquisition_model.name,
        cache_dir,
        end_word,
    )
    if getattr(config, "successor_model", None) is not None:
        cache_model_and_tokenizer(
            config.successor_model.type,
            config.successor_model.name,
            cache_dir,
            end_word,
        )
    if getattr(config, "target_model", None) is not None:
        cache_model_and_tokenizer(
            config.target_model.type, config.target_model.name, cache_dir, end_word
        )
    # Cache dataset and dataset metrics
    metrics_cache_dir = cache_dir / "metrics"
    if config.data.path == "datasets":
        if cache_dir is not None:
            data_cache_dir = cache_dir / "data"
        else:
            data_cache_dir = None
            metrics_cache_dir = None
        log.info("." * 10 + f" {start_word} dataset " + "." * 10)
        if isinstance(config.data.dataset_name, str):
            load_dataset(config.data.dataset_name, cache_dir=data_cache_dir)
            log.info(
                "=" * 10
                + f" Dataset {config.data.dataset_name} successfully {end_word} "
                + "=" * 10
            )
            try:
                load_metric(config.data.dataset_name, cache_dir=metrics_cache_dir)
                log.info(
                    "=" * 10
                    + f" Dataset {config.data.dataset_name} metric successfully {end_word} "
                    + "=" * 10
                )
            except:
                pass
        else:
            load_dataset(*list(config.data.dataset_name), cache_dir=data_cache_dir)
            log.info(
                "=" * 10
                + f" Dataset {list(config.data.dataset_name)} successfully {end_word} "
                + "=" * 10
            )
            try:
                load_metric(
                    *list(config.data.dataset_name), cache_dir=metrics_cache_dir
                )
                log.info(
                    "=" * 10
                    + f" Dataset {list(config.data.dataset_name)} metric successfully {end_word} "
                    + "=" * 10
                )
            except:
                pass
    # Cache dataset metrics
    log.info("." * 10 + f" {start_word} metrics " + "." * 10)
    main_metric = (
        "accuracy"
        if config.task == "cls"
        else "seqeval"
        if config.task == "ner"
        else "rouge"
        if config.task == "abs-sum"
        else None
    )
    load_metric(main_metric, cache_dir=metrics_cache_dir)
    log.info(
        "=" * 10 + f" Main metric {main_metric} successfully {end_word} " + "=" * 10
    )
    eval_metrics = config.acquisition_model.training.trainer_args.eval_metrics
    if eval_metrics is not None:
        for metric in eval_metrics:
            load_metric(metric, cache_dir=metrics_cache_dir)
            log.info(
                "=" * 10
                + f" Additional metric {metric} successfully {end_word} "
                + "=" * 10
            )

    if config.al.strategy in ("abs_sum_mc_dropout", "beam_variance"):
        metric = config.al.strategy_kwargs.var_metric
        load_metric(metric, cache_dir=metrics_cache_dir)
        log.info(
            "=" * 10 + f" Strategy metric {metric} successfully {end_word} " + "=" * 10
        )


if __name__ == "__main__":
    main()
