import numpy as np
import json
from pathlib import Path
from copy import deepcopy
from tqdm import tqdm
from typing import List, Union, Tuple
import random
import dill
import logging
import shutil
from torch.utils.data import DataLoader
from omegaconf.omegaconf import ListConfig
from sklearn.model_selection import train_test_split

from datasets import concatenate_datasets
from datasets.arrow_dataset import Dataset as ArrowDataset

from transformers import (
    DataCollatorWithPadding,
    DataCollatorForTokenClassification,
    set_seed,
)

from ..utils.general import (
    json_dump,
    json_load,
    pickle_dump,
    initialize_metrics_dict,
    add_new_model_to_time_dict,
    get_time_dict_path,
    get_target_model_checkpoints,
    get_balanced_sample_indices,
)
from ..utils.plasm import label_data, concatenate_data
from ..utils.calculate_tracin_score import calculate_outlier_scores
from ..utils.transformers_dataset import TransformersDataset
from ..utils.cluster_margin import hierarchial_ac

from ..construct.construct_active_learner import (
    construct_active_learner,
    QUERY_STRATEGIES,
)
from ..construct import construct_model

from .al_strategy_utils import (
    get_query_idx_for_selecting_by_number_of_tokens,
    get_num_query_instances_or_tokens,
)
from .sampling_strategy import random_subsampling


log = logging.getLogger()
f_format_data = lambda data: [list(e) for e in zip(*data)]


def initial_split(
    data,
    config,
    task,
    work_dir,
    tokens_column_name=None,
    labels_column_name=None,
    seed=42,
    include_last=True,
) -> Union[
    Tuple[ArrowDataset, ArrowDataset], Tuple[TransformersDataset, TransformersDataset]
]:
    """
    Create initial split for AL
    :param config: config.al
    """
    init_p_or_n = config.init_p_or_n
    stratify = config.get("initial_split_stratify", False)

    np.random.seed(seed)
    random.seed(seed)

    if task == "ner" and config.split_by_tokens:
        total_num_tokens = sum([len(x[tokens_column_name]) for x in data])
        init_n = (
            init_p_or_n
            if isinstance(init_p_or_n, int)
            else int(init_p_or_n * total_num_tokens)
        )
        # Shuffle data ids to have the data randomly sorted
        sorted_idx = list(range(len(data)))
        random.shuffle(sorted_idx)
        # Get idx of data to label
        init_id = get_query_idx_for_selecting_by_number_of_tokens(
            data, sorted_idx, init_n, tokens_column_name, include_last
        )
        unlabeled_id = np.setdiff1d(sorted_idx, init_id)

    elif task == "cv_cls":
        range_len_data = np.arange(len(data))
        init_id = get_balanced_sample_indices(data, 10, n_per_digit=2)
        unlabeled_id = np.setdiff1d(range_len_data, init_id)

    else:
        strategy = config.get("initial_strategy", "random")
        init_n = (
            init_p_or_n
            if isinstance(init_p_or_n, int)
            else int(init_p_or_n * len(data))
        )
        range_len_data = np.arange(len(data))
        if strategy == "random":
            if stratify:
                init_id = train_test_split(
                    range(len(data)),
                    train_size=init_n,
                    stratify=data[labels_column_name],
                    random_state=seed,
                )[0]
            else:
                init_id = np.random.choice(
                    range_len_data, init_n, replace=False
                ).tolist()
        else:
            query_strategy = QUERY_STRATEGIES[strategy]
            # TODO: make neater
            strategy_kwargs = config.strategy_kwargs
            if config.sampling_type == "random":
                subsample_ratio = len(data) // config.gamma_or_k_confident_to_save
                strategy_kwargs["subsample_ratio"] = subsample_ratio
            init_id, *_ = query_strategy(
                None, data, init_n, None, seed=seed, device="cuda", **strategy_kwargs
            )
        unlabeled_id = np.setdiff1d(range_len_data, init_id)

    initial_data = data.select(init_id)
    unlabeled_data = data.select(unlabeled_id)

    log.info(f"Seeding dataset size: {len(initial_data)}")
    log.info(f"Pool size: {len(unlabeled_data)}")
    json_dump([int(x) for x in init_id], Path(work_dir) / "ids_data_query_0.json")

    return initial_data, unlabeled_data


def update_unlabeled_data_and_uncertainty_estimates(
    unlabeled_data, uncertainty_estimates, query_idx, framework="transformers"
):
    unlabeled_data = unlabeled_data.select(
        np.setdiff1d(range(len(unlabeled_data)), query_idx)
    )
    uncertainty_estimates = np.delete(uncertainty_estimates, query_idx, axis=0)
    return unlabeled_data, uncertainty_estimates


def probably_fit_and_evaluate_model(
    model,
    test_data,
    train_data=None,
    require_fit=True,
    model_name="successor",
    framework="transformers",
):
    if require_fit:
        model.fit(train_data)
    serialization_dir = model._trainer_kwargs["serialization_dir"]
    if serialization_dir is not None and model_name != "target":
        shutil.rmtree(Path(serialization_dir))

    log.info(f"############### Evaluating the {model_name} model. ###############")
    return model.evaluate(test_data)


def log_model(
    log,
    work_dir,
    evaluate_dict,
    model_name="successor",
    idx=None,
    framework="transformers",
):
    if idx is None:
        log.info(f"Initial AL iteration:\n{model_name.title()} model:")
    else:
        log.info(f"AL iteration {idx + 1}:\n{model_name.title()} model:")

    metrics_dict_path = Path(work_dir) / f"{model_name}_metrics.json"
    metrics_dict = json_load(metrics_dict_path)

    for key in evaluate_dict:
        if key not in metrics_dict:
            metrics_dict[key] = [evaluate_dict[key]]
        else:
            metrics_dict[key].append(evaluate_dict[key])
        log.info(f"{key}: {evaluate_dict[key]}")

    json_dump(metrics_dict, metrics_dict_path)


def log_query_meta(log, work_dir, query_meta: dict):
    query_meta_path = Path(work_dir) / f"query_meta.json"
    try:
        all_query_meta = json_load(query_meta_path)
    except Exception as exc:
        log.info(f"Could not load query meta from {query_meta_path}: {exc}")
        all_query_meta = []

    log.info(f"Query meta: {query_meta}")
    log.info(f"Dumping query meta to {query_meta_path}")
    assert isinstance(all_query_meta, list)
    all_query_meta.append(query_meta)

    json_dump(all_query_meta, query_meta_path)


def iteration_with_tracin(
    config,
    target_model,
    pseudo_labeled_data,
    train_data,
    test_data,
    successor_model_quality: float,
    filtered_data_share: float,
    work_dir,
    cache_dir="./cache",
    framework="transformers",
):
    # Check whether we need TracIn
    if len(config.tracin.quantiles) == 1:
        if config.tracin.quantiles[0] == -1:
            if 1 - successor_model_quality - filtered_data_share < 0:
                log.info(
                    f"Skipping TracIn step since the expected filtering share =="
                    f" {1 - successor_model_quality - filtered_data_share:.5f} "
                    f"(Pseudo-labeling model quality == {successor_model_quality:.4f};"
                    f" Filtered by uncertainty share: {filtered_data_share:.4f}.)"
                )
                quantile = config.tracin.quantiles[0]
                metrics_dict_path = (
                    Path(work_dir)
                    / f"{target_model.name}_tracin_quantile_{quantile}_metrics.json"
                )
                target_model_metrics_dict = json_load(
                    Path(work_dir) / "target_metrics.json"
                )
                if not Path(metrics_dict_path).exists():
                    metrics_dict = target_model_metrics_dict
                else:
                    # Dump the metrics dict
                    metrics_dict = json_load(metrics_dict_path)
                    for key in metrics_dict:
                        metrics_dict[key].append(target_model_metrics_dict[key][-1])
                        log.info(f"{key}: {target_model_metrics_dict[key][-1]}")
                json_dump(metrics_dict, metrics_dict_path)
                dict_to_log = {k: v[-1] for k, v in metrics_dict.items()}
                log.info(f"Quantile {quantile}: {dict_to_log}")
                return
    # Dump target model and dataloader
    model_path = Path(cache_dir) / f"tmp_target_model_{config.seed}"
    model_weights_paths = get_target_model_checkpoints(config, framework)
    dataloader_path = Path(cache_dir) / f"dataloader_{config.seed}"

    # Cache attributes that will be modified
    target_serialization_dir = deepcopy(
        config.target_model.training.trainer_args.serialization_dir
    )
    target_name = deepcopy(target_model.name)
    fp16 = deepcopy(config.target_model.training.trainer_args.fp16)

    with open(model_path, "wb") as f:
        dill.dump(target_model.model, f)

    # Senseless line, necessary for compatibility
    pseudo_labeled_data_copy = pseudo_labeled_data
    # Tokenize data
    tokenizer = target_model.tokenizer
    tokenized_data = target_model.tokenize_data(
        tokenizer=tokenizer,
        data=pseudo_labeled_data_copy,
        task=target_model.task,
        text_name=target_model.data_config["text_name"],
        label_name=target_model.data_config["label_name"],
        save_first_bpe_mask=True,
    )

    if target_model.task == "cls":
        data_collator_class = DataCollatorWithPadding
    elif target_model.task == "ner":
        data_collator_class = DataCollatorForTokenClassification
    else:
        raise NotImplementedError
    dataloader = DataLoader(
        tokenized_data,
        batch_size=1,
        pin_memory=True,
        collate_fn=data_collator_class(tokenizer=tokenizer, padding="longest"),
    )

    with open(dataloader_path, "wb") as f:
        dill.dump(dataloader, f)
    # Calculate TracIn score
    scores, *_ = calculate_outlier_scores(
        model_path,
        model_weights_paths,
        dataloader_path,
        work_dir,
        config.tracin.max_num_processes,
        target_model.task,
        config.tracin.nu,
    )
    # Change fp16 dict
    target_model._trainer_kwargs["fp16"] = target_model._trainer_kwargs.get(
        "final_model_fp16", False
    )

    for quantile_name in config.tracin.quantiles:
        quantile = float(quantile_name)
        # Modify the name to avoid overwriting time-file
        target_model.name = target_name + f"_tracin_quantile_{quantile_name}"
        # Add initial empty fit/predict lists of the model to the time dict
        add_new_model_to_time_dict(target_model.time_dict_path, target_model.name)
        # Load key metrics of the target model (f1-score for NER, accuracy for CLS)
        if quantile_name == -1:
            quantile = 1 - successor_model_quality - filtered_data_share
        outliers = np.argwhere(scores > np.quantile(scores, (1 - quantile))).ravel()
        pseudo_labeled_data_to_use = [
            pseudo_labeled_data[i]
            for i in range(len(pseudo_labeled_data))
            if i not in outliers
        ]

        if target_model.task == "ner":
            tokenization_kwargs = {"is_split_into_words": True}
        elif target_model.task == "cls":
            tokenization_kwargs = {}
        else:
            raise NotImplementedError

        pseudo_labeled_data_to_use = TransformersDataset(
            pseudo_labeled_data_to_use,
            text_column_name=config.data.text_name,
            label_column_name=config.data.label_name,
            tokenization_kwargs=tokenization_kwargs,
            task=pseudo_labeled_data.task,
            id2label=pseudo_labeled_data.id2label,
            label_smoothing=pseudo_labeled_data.label_smoothing,
        )
        labeled_weight = target_model.model_config.training.get("labeled_weight", 1.0)
        all_data_for_model = concatenate_data(
            train_data, pseudo_labeled_data_to_use, framework, labeled_weight
        )
        model_evaluate_dict = probably_fit_and_evaluate_model(
            target_model,
            test_data,
            all_data_for_model,
            True,
            f"target_{quantile_name}",
            framework,
        )
        # Cast from np.float to python float to dump in json format
        model_evaluate_dict = {
            key: float(model_evaluate_dict[key]) for key in model_evaluate_dict
        }
        # Create the initial dict if it does not exist
        metrics_dict_path = Path(work_dir) / f"{target_model.name}_metrics.json"
        if not Path(metrics_dict_path).exists():
            metrics_dict = {k: [v] for k, v in model_evaluate_dict.items()}
            [
                log.info(f"{key}: {model_evaluate_dict[key]}")
                for key in model_evaluate_dict
            ]
        else:
            # Dump the metrics dict
            metrics_dict = json_load(metrics_dict_path)
            for key in model_evaluate_dict:
                metrics_dict[key].append(model_evaluate_dict[key])
                log.info(f"{key}: {model_evaluate_dict[key]}")

        json_dump(metrics_dict, metrics_dict_path)
        log.info(f"Quantile {quantile_name}: {model_evaluate_dict}")

    # Return the original name, serialization_dir, and fp16
    target_model._trainer_kwargs["serialization_dir"] = target_serialization_dir
    target_model.name = target_name
    target_model._trainer_kwargs["fp16"] = fp16
    # Remove the serialization dir if it exists
    if Path(target_serialization_dir).exists():
        shutil.rmtree(Path(target_serialization_dir))


def probably_fit_and_evaluate_and_log_models(
    models: List,
    model_names: List[str],
    require_fit_list: List[bool],  # i-th el says whether models[i] should be fitted
    log,
    work_dir,
    idx,
    test_data,
    train_data=None,
    framework="transformers",
):
    for model, model_name, require_fit in zip(models, model_names, require_fit_list):
        evaluate_dict = probably_fit_and_evaluate_model(
            model, test_data, train_data, require_fit, model_name, framework
        )
        log_model(log, work_dir, evaluate_dict, model_name, idx, framework)


def create_models(
    framework,
    config,
    dev_instances,
    labels_or_id2label,
    time_dict_path,
    work_dir,
    id_first_iteration: int = 0,
    embeddings=None,
    word2idx=None,
):

    # Create acquisition model
    log.info("Constructing the acquisition model...")
    acquisition_model = construct_model(
        config,
        config.acquisition_model,
        dev_instances,
        framework,
        labels_or_id2label,
        "acquisition",
        time_dict_path,
        embeddings=embeddings,
        word2idx=word2idx,
    )
    log.info("Done with constructing the acquisition model.")

    if id_first_iteration == 0:
        initialize_metrics_dict(
            acquisition_model.task,
            work_dir,
            "acquisition",
            framework,
            config.al.evaluate_query,
        )

    models = [acquisition_model]
    model_names = ["acquisition"]
    require_fit_list = [False]

    # If successor differs from acquisition, create successor model
    if config.successor_model:
        log.info("Constructing the successor model...")
        successor_model = construct_model(
            config,
            config.successor_model,
            dev_instances,
            framework,
            labels_or_id2label,
            "successor",
            time_dict_path,
            embeddings=embeddings,
            word2idx=word2idx,
        )
        log.info("Done with constructing the successor model.")

        if id_first_iteration == 0:
            initialize_metrics_dict(
                successor_model.task, work_dir, "successor", framework
            )

        models.append(successor_model)
        model_names.append("successor")
        require_fit_list.append(True)

    # If target model exists (i.e. dealing with PLASM), construct it as well
    if "target_model" in config:
        log.info("Constructing the target model...")
        target_model = construct_model(
            config,
            config.target_model,
            dev_instances,
            framework,
            labels_or_id2label,
            "target",
            time_dict_path,
            embeddings=embeddings,
            word2idx=word2idx,
        )
        log.info("Done with constructing the target model.")

        if id_first_iteration == 0:
            initialize_metrics_dict(target_model.task, work_dir, "target", framework)
    else:
        target_model = None

    return models, model_names, require_fit_list, target_model


def al_loop(
    config,
    work_dir,
    initial_data,
    dev_data,
    test_data,
    unlabeled_data,
    labels_or_id2label=None,
    id_first_iteration: int = 0,
    embeddings=None,
    word2idx=None,
):
    """
    Function launches a loop of AL: according to selected strategy, query the most "important"
    instances from unlabeled data, label them, add to train sample and retrain the model (from scratch).
    """
    # Determine the task
    task = config.task
    # Whether to use label_smoothing for pseudo labeled data; True -> use classic ls, 'natural' -> use returned probas
    pl_label_smoothing = None
    pl_label_smoothing_is_adaptive = False
    if config.get("target_model", None) is not None:
        pl_label_smoothing = config.target_model.training.get(
            "pseudo_labeled_label_smoothing", False
        )
        if pl_label_smoothing == "adaptive":
            pl_label_smoothing_is_adaptive = True
            log.info("Adaptive label smoothing is used")
        # If we want to increase the weight of the "real labeled data" compared to pseudo labeled data
        labeled_weight = config.target_model.training.get("labeled_weight", 1)
        log.info(
            f"Pseudo-labeling: using weight {labeled_weight} for the true labeled data."
        )
    # Whether pseudo-labeling will be performed over a subsample of the data
    use_subsample_for_pl = getattr(config.al, "use_subsample_for_pl", False)
    # Uncertainty threshold
    unc_threshold = getattr(config.al, "plasm_thresh", None)
    unc_threshold_is_adaptive = False
    if unc_threshold == "adaptive":
        unc_threshold_is_adaptive = True
        log.info("Adaptive uncertainty threshold is used")
    # Set framework
    framework = config.framework.name
    # Set time dict path
    time_dict_path = get_time_dict_path(config)
    # Set cache dir
    cache_dir = (
        config.cache_dir
        if config.cache_dir is not None
        else Path(config.output_dir) / "cache"
    )
    # Calculate the required number of instances/tokens to select
    num_query_instances_or_tokens = get_num_query_instances_or_tokens(
        config.al, initial_data, unlabeled_data, framework, config.data.text_name
    )
    # Whether UPS will be used
    use_ups = config.al.sampling_type is not None
    # Iterations on which we need to recalculate the uncertainty estimates for the whole unlabeled dataset
    iters_to_recalc_scores = (
        range(config.al.num_queries)
        if (not use_ups) or config.al.iters_to_recalc_scores == "all"
        else []
        if config.al.iters_to_recalc_scores == "no"
        else config.al.iters_to_recalc_scores
    )
    # If successor_model is not provided, acquisition model is used as a successor as well
    # If target_model is not provided, variable `target_model` equals None
    models, model_names, require_fit_list, target_model = create_models(
        framework,
        config,
        dev_data,
        labels_or_id2label,
        time_dict_path,
        work_dir,
        id_first_iteration,
        embeddings=embeddings,
        word2idx=word2idx,
    )
    # Construct Active Learner
    learner = construct_active_learner(
        models[0], config.al, initial_data, work_dir, framework
    )
    # If we are continuing AL or solving task of AS, we do not need these steps
    if id_first_iteration == 0:
        # Evaluate acquisition and successor models (fit in advance if necessary) on the seeding dataset (zero iteration)
        probably_fit_and_evaluate_and_log_models(
            models,
            model_names,
            require_fit_list,
            log,
            work_dir,
            None,
            test_data,
            learner.train_data,
            framework,
        )
        # After train calculate clusters
        if config.al.strategy == "cluster_margin":
            clusters = hierarchial_ac(
                models[0], unlabeled_data, learner.train_data, config
            )
            unlabeled_data.clusters = clusters
        # Deal with target_model
        if target_model is not None:
            successor_model_quality = getattr(models[-1], "best_metric", 1.0)
            # Pseudo-label unlabeled data
            if pl_label_smoothing_is_adaptive:
                pl_label_smoothing = successor_model_quality
                log.info(f"Using pseudo-labeling parameter {pl_label_smoothing:.5f}.")
            # Uncertainty filtering for unlabeled data
            if unc_threshold_is_adaptive:
                unc_threshold = successor_model_quality
                log.info(f"Using uncertainty parameter {unc_threshold:.5f}.")
            unc_filt_by_quant = unc_threshold_is_adaptive or isinstance(
                unc_threshold, (ListConfig, list)
            )
            # Set data for pseudo-labeling
            if use_subsample_for_pl:
                subsample_idx = random_subsampling(
                    np.ones(len(unlabeled_data)), config.al.gamma_or_k_confident_to_save
                )
                instances_to_query_from = unlabeled_data.select(subsample_idx)
            else:
                instances_to_query_from = unlabeled_data
            pseudo_labeled_data, filtered_data_share = label_data(
                instances_to_query_from,
                models[-1],
                framework,
                unc_threshold,
                unc_filt_by_quant,
                pl_label_smoothing,
            )
            # Concatenate the labeled-by-oracle and pseudo-labeled data
            target_train_data = concatenate_data(
                learner.train_data, pseudo_labeled_data, framework, labeled_weight
            )
            # Fit and evaluate the target model
            probably_fit_and_evaluate_and_log_models(
                [target_model],
                ["target"],
                [True],
                log,
                work_dir,
                None,
                test_data,
                target_train_data,
                framework,
            )
            # Use TracIn if necessary
            if config.tracin.use:
                Path(cache_dir).mkdir(exist_ok=True)
                iteration_with_tracin(
                    config,
                    target_model,
                    pseudo_labeled_data,
                    learner.train_data,
                    test_data,
                    successor_model_quality,
                    filtered_data_share,
                    work_dir,
                    cache_dir,
                    framework,
                )
        if config.get("dump_train_data", False):
            pickle_dump(
                learner.train_data,
                Path(work_dir) / f"train_data_{id_first_iteration}.pkl",
            )

    # Just an initial value to prevent errors
    uncertainty_estimates = np.ones(len(unlabeled_data)) * 1000
    for idx in tqdm(
        range(id_first_iteration, config.al.num_queries), desc="AL queries done"
    ):
        log.info(f"=================AL iteration #{idx+1} started.=================")
        # Whether need to recalculate uncertainty estimates on the current iteration. If UPS is not used,
        # need to update on each iteration
        require_update_all_probabilities = idx in iters_to_recalc_scores
        ### Query the idx of the most important instances
        # If require update all probas on the current iter, we need to update
        # the uncertainty estimates for all the unlabeled data
        if require_update_all_probabilities:
            (
                query_idx,
                query_instance,
                uncertainty_estimates,
                *query_meta,
            ) = learner.query(unlabeled_data, n_instances=num_query_instances_or_tokens)
        # Else we need to sample instances from the unlabeled data and update
        # the uncertainty estimates only for these instances
        else:
            sampled_idx, instances_to_query_from = learner.sample(
                unlabeled_data, uncertainty_estimates
            )
            (
                query_idx,
                query_instance,
                sampled_uncertainty_estimates,
                *query_meta,
            ) = learner.query(
                instances_to_query_from, n_instances=num_query_instances_or_tokens
            )
            # Update the uncertainty estimates
            uncertainty_estimates[sampled_idx] = sampled_uncertainty_estimates
            # `query_idx` points at the idx of the queried instances in `instances_to_query_from`
            # array. To make it point at the idx of the instances in `unlabeled_data`, we need to
            # take their original idx
            query_idx = sampled_idx[query_idx]

        log_query_meta(log, work_dir, query_meta)

        # Log the uncertainty of the queries instances
        if require_update_all_probabilities:
            sampled_uncertainty_estimates = uncertainty_estimates
        log.info("### All uncertainty estimates ###")
        log.info(
            ", ".join(map(str, sorted(sampled_uncertainty_estimates.round(3))[::-1]))
        )
        log.info("### Uncertainties of the queries ###")
        log.info(", ".join(map(str, uncertainty_estimates[query_idx].round(5))))
        # ¡¡¡For debugging purposes - must be removed in real active learning!!!
        if config.al.evaluate_query:
            log.info(
                f"############### Evaluating the query by the acquisition model. ###############"
            )
            evaluate_dict = learner.estimator.evaluate(query_instance)
            log_model(
                log,
                work_dir,
                evaluate_dict,
                "acquisition_evaluate_query",
                idx,
                framework,
            )
        # Retrain the model, using the queried instances
        learner.teach(query_instance)
        # Evaluate the model (fitting it beforehand if the successor model differs from the acquisition)
        # and log the results
        probably_fit_and_evaluate_and_log_models(
            models,
            model_names,
            require_fit_list,
            log,
            work_dir,
            idx,
            test_data,
            learner.train_data,
            framework,
        )
        # Remove the queried instances from the unlabeled data and their uncertainty estimates
        (
            unlabeled_data,
            uncertainty_estimates,
        ) = update_unlabeled_data_and_uncertainty_estimates(
            unlabeled_data, uncertainty_estimates, query_idx, framework
        )
        # In case of using cluster-margin algorithm, also update clusters
        if config.al.strategy == "cluster_margin":
            clusters = np.array(
                [clusters[idx] for idx in range(len(clusters)) if idx not in query_idx]
            )
            unlabeled_data.clusters = clusters
        # Deal with target_model
        if target_model is not None:
            successor_model_quality = getattr(models[-1], "best_metric", 1.0)
            # Pseudo-label unlabeled data
            if pl_label_smoothing_is_adaptive:
                pl_label_smoothing = successor_model_quality
                log.info(f"Using pseudo-labeling parameter {pl_label_smoothing:.5f}.")
            # Uncertainty filtering for unlabeled data
            if unc_threshold_is_adaptive:
                unc_threshold = successor_model_quality
                log.info(f"Using uncertainty parameter {unc_threshold:.5f}.")
            # Pseudo-label unlabeled data
            unc_filt_by_quant = unc_threshold_is_adaptive or isinstance(
                unc_threshold, (ListConfig, list)
            )
            # Set data for pseudo-labeling
            if not use_subsample_for_pl or require_update_all_probabilities:
                instances_to_query_from = unlabeled_data
            pseudo_labeled_data, filtered_data_share = label_data(
                instances_to_query_from,
                models[-1],
                framework,
                unc_threshold,
                unc_filt_by_quant,
                pl_label_smoothing,
            )
            # Concatenate the labeled-by-oracle and pseudo-labeled data
            target_train_data = concatenate_data(
                learner.train_data, pseudo_labeled_data, framework, labeled_weight
            )
            # Fit and evaluate the target model
            probably_fit_and_evaluate_and_log_models(
                [target_model],
                ["target"],
                [True],
                log,
                work_dir,
                idx,
                test_data,
                target_train_data,
                framework,
            )
            # Use TracIn if necessary
            if config.tracin.use:
                iteration_with_tracin(
                    config,
                    target_model,
                    pseudo_labeled_data,
                    learner.train_data,
                    test_data,
                    successor_model_quality,
                    filtered_data_share,
                    work_dir,
                    cache_dir,
                    framework,
                )
        # Dump the idx of queries
        json_dump(query_idx.tolist(), Path(work_dir) / f"ids_data_query_{idx + 1}.json")
        if config.get("dump_train_data", False):
            pickle_dump(
                learner.train_data, Path(work_dir) / f"train_data_{idx + 1}.pkl"
            )

    # Total logging
    for model_name in model_names:
        log.info(f"{model_name.title()} model metrics:")
        metrics_dict = json_load(Path(work_dir) / f"{model_name}_metrics.json")
        [log.info(f"{key}: {metrics_dict[key]}") for key in metrics_dict]

    if target_model is not None:
        return models + [target_model, learner]
    return models + [learner]
