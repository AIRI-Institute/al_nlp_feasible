import json
import yaml
import pandas as pd
from pathlib import Path
import logging
from numpy import nan


log = logging.getLogger()

NUM_QUERIES = 16
QUANTILES = (-1, -1.0, 0.01, 0.025, 0.05, 0.1)
METRIC_NAME = "accuracy"  # f1-measure-overall

METRIC_FILE_NAMES = [
    "target_metrics.json",
    "successor_metrics.json",
    "acquisition_metrics.json",
    "metrics.json",
]

AL_STRATEGIES = ["random", "mahalanobis", "nuq", "logits", "ddu", "margin", "oracle"]


def _json_load_with_unknown_prefix(file_name, prefixes, metric_name):
    with open(file_name) as f:
        metrics = json.load(f)
    for prefix in prefixes:
        try:
            target_metric = metrics[prefix + metric_name]
            return target_metric
        except:
            continue


def _add_observation_to_df(
    df, exp_data, metrics, num_queries, id_experiment, fill_missing=False
):
    row = (
        [id_experiment]
        + [exp_data[x] for x in df.columns[1:11]]
        + metrics[:num_queries]
    )
    if len(row) < df.shape[1] and fill_missing:
        row = row + [None for _ in range(df.shape[1] - len(row))]
    df.loc[len(df)] = row


def extract_info_from_config(config):
    data = {}
    data["strategy"] = config["al"]["strategy"] if "al" in config else "full"
    data["acquisition"] = config["acquisition_model"]["name"]
    data["successor"] = (
        config["successor_model"]["name"]
        if config["successor_model"]
        else data["acquisition"]
    )
    data["target"] = (
        config["target_model"]["name"]
        if "target_model" in config
        else data["successor"]
    )
    data["deleted"] = (
        config["al"]["share_confident_to_save"]
        if "al" in config and "share_confident_to_save" in config["al"]
        else round(1 - config["al"]["plasm_thresh"], 4)
        if "al" in config
        and config["al"].get("plasm_thresh", 1) != 1
        and isinstance(config["al"]["plasm_thresh"], (int, float))
        else config["al"]["plasm_thresh"]
        if "al" in config and isinstance(config["al"].get("plasm_thresh", 1), str)
        else 0
    )
    data["seed"] = config["seed"]
    dataset = config["data"]["dataset_name"]
    data["dataset"] = " ".join(dataset) if isinstance(dataset, list) else dataset
    data["ups"] = (
        "None"
        if "al" not in config
        else "k=1.0, T=0"
        if not config["al"]["sampling_type"]
        else f"k={config['al']['gamma_or_k_confident_to_save']}, T={config['al']['T']}"
    )
    data["framework"] = (
        config["framework"]["name"] if "framework" in config else "allennlp"
    )
    return data


def search_for_experiments(
    path,
    df,
    num_queries=16,
    metric_name="accuracy",
    fill_missing=False,
    prefix: str or None = "test",  # "test" or "eval" or None
    quantiles=QUANTILES,
    log_errors=False,
    metric_file_names=None,
    id_experiment=0,
):
    if prefix is None or prefix == "none":
        metrics_prefix = ""
    else:
        metrics_prefix = prefix + "_"

    if metric_file_names is None:
        metric_file_names = METRIC_FILE_NAMES

    iterdir = list(path.iterdir())
    if (path / "config.yaml" in iterdir) and (
        (path / "acquisition_metrics.json" in iterdir)
        or (path / "metrics.json" in iterdir)
    ):

        try:
            with open(path / "config.yaml") as f:
                config = yaml.load(f, yaml.Loader)

            for file_name in metric_file_names:
                if path / file_name in iterdir:
                    metrics = _json_load_with_unknown_prefix(
                        path / file_name,
                        [metrics_prefix, "test_", "eval_", ""],
                        metric_name,
                    )
                    break

            if "tracin" in config and config["tracin"]["use"]:
                tracin_results = []
                processed_quantiles = []
                for quant in quantiles:
                    try:
                        tracin_results.append(
                            _json_load_with_unknown_prefix(
                                path / f"target_tracin_quantile_{quant}_metrics.json",
                                [metrics_prefix, "test_", "eval_", ""],
                                metric_name,
                            )
                        )
                        processed_quantiles.append(quant)
                    except Exception as e:
                        if log_errors:
                            log.error(e, exc_info=True)
                quantiles = processed_quantiles

            al_type = "one"  # Default value
            if "acquisition_model" not in config:
                config["acquisition_model"] = config["model"]
                config["successor_model"] = None
                metrics = [metrics for _ in range(num_queries)]
                al_type = "full"
            else:
                if "target_model" in config:
                    if config["al"].get("plasm_thresh", 1) != 1:
                        al_type = "plasm_thresh"
                    else:
                        al_type = "plasm"
                elif config["successor_model"]:
                    al_type = "asm"

            exp_data = extract_info_from_config(config)
            exp_data["al_type"] = al_type
            _add_observation_to_df(
                df, exp_data, metrics, num_queries, id_experiment, fill_missing
            )

            if "tracin" in config and config["tracin"]["use"]:
                for i, metrics in enumerate(tracin_results):
                    exp_data["deleted"] = quantiles[i]
                    _add_observation_to_df(
                        df, exp_data, metrics, num_queries, id_experiment, fill_missing
                    )

        except Exception as e:
            if log_errors:
                if (
                    isinstance(e, ValueError)
                    and str(e) == "cannot set a row with mismatched columns"
                ):
                    log.error(
                        f"Path: {str(path)}; num obs: {len(metrics[:num_queries])}; "
                        f"required num obs: {df.shape[1] - len([exp_data[x] for x in df.columns[:10]])}"
                    )
                else:
                    log.error(e, exc_info=True)
            return

    else:
        for file in path.iterdir():
            if file.is_dir():
                search_for_experiments(
                    file,
                    df,
                    num_queries,
                    metric_name,
                    fill_missing=fill_missing,
                    prefix=prefix,
                    quantiles=quantiles,
                    log_errors=log_errors,
                    metric_file_names=metric_file_names,
                    id_experiment=id_experiment,
                )


def collect_data(
    paths,
    num_queries=16,
    task="cls",
    fill_missing=False,
    prefix: str or None = "test",  # "test" or "eval" or None
    quantiles=QUANTILES,
    log_errors=False,
    metric_name=None,
    metric_file_names=None,
):
    if metric_name is None:
        if task == "cls":
            metric_name = "accuracy"
        elif task == "ner":
            metric_name = "overall_f1"
        elif task == "abs-sum":
            metric_name = "rougeL"
        else:
            raise NotImplementedError

    if quantiles is None:
        quantiles = QUANTILES

    df = pd.DataFrame(
        columns=[
            "id_experiment",
            "al_type",
            "strategy",
            "dataset",
            "acquisition",
            "successor",
            "target",
            "deleted",
            "ups",
            "framework",
            "seed",
        ]
        + [f"f1_{i}" for i in range(num_queries)]
    )

    [
        search_for_experiments(
            Path(path),
            df=df,
            num_queries=num_queries,
            metric_name=metric_name,
            fill_missing=fill_missing,
            prefix=prefix,
            quantiles=quantiles,
            log_errors=log_errors,
            metric_file_names=metric_file_names,
            id_experiment=id_experiment,
        )
        for id_experiment, path in enumerate(paths)
    ]
    df.loc[:, "f1_0":] = df.loc[:, "f1_0":].astype(float)
    df = df[~df.duplicated()]
    # Replace values for abs-sum metrics
    if (
        metric_name.startswith("ngram_overlap")
        or metric_name.startswith("novel_ngrams_2")
        or metric_name.startswith("cons_score")
    ):
        df.loc[:, "f1_0":] = df.loc[:, "f1_0":].replace(-1, nan)
    log.info(f"Num observations: {len(df)}")
    return df
