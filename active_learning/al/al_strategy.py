import numpy as np
from datasets.arrow_dataset import Dataset
from typing import Union, Optional, Tuple
import torch
from datasets import load_metric
from tqdm import tqdm
from math import ceil
from omegaconf.dictconfig import DictConfig
import logging
import yaml
from pathlib import Path
import sys

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from torch.nn.functional import normalize

from .al_strategy_utils import (
    get_query_idx_for_selecting_by_number_of_tokens,
    get_X_pool_subsample,
    get_similarities,
    filter_by_uncertainty,
    filter_by_metric,
    calculate_bleuvar_scores,
    assign_ue_scores_for_unlabeled_data,
    calculate_mnlp_score,
    calculate_bald_score_ner,
    calculate_bald_score_cls,
    calculate_alps_scores,
    var_ratio,
    mean_entropy,
    sampled_max_prob,
    probability_variance,
    mean_entropy_ner,
    probability_variance_ner,
    take_idx,
    calculate_unicentroid_mahalanobis_distance,
    calculate_mahalanobis_distance,
    calculate_mahalanobis_triplet_scores,
    calculate_mahalanobis_filtering_scores,
    calculate_triplet_scores,
    calculate_ddu_scores,
    calculate_ddu_scores_cv,
    calculate_cal_scores,
    calculate_badge_scores,
    choose_cm_samples,
    _get_similarities_from_cache_or_from_scratch,
)
from .strategy_utils.batchbald.batchbald import get_batchbald_batch
from .strategy_utils.batchbald.consistent_dropout import make_dropouts_consistent

from ..utils.transformers_dataset import TransformersDataset
from ..utils.get_embeddings import get_embeddings, get_model_without_cls_layer
from ..utils.cluster_utils import badge, kmeans
from ..construct.transformers_api.construct_transformers_wrapper import (
    construct_transformers_wrapper,
)


log = logging.getLogger()


def random_sampling(
    model,
    X_pool,
    n_instances,
    *args,
    select_by_number_of_tokens: bool = False,
    **kwargs,
):
    if select_by_number_of_tokens:
        sorted_idx = np.arange(len(X_pool))
        np.random.shuffle(sorted_idx)
        query_idx = get_query_idx_for_selecting_by_number_of_tokens(
            X_pool, sorted_idx, n_instances
        )
    else:
        query_idx = np.random.choice(range(len(X_pool)), n_instances, replace=False)

    query = take_idx(X_pool, query_idx)
    # Uncertainty estimates are not defined with random sampling
    uncertainty_estimates = np.zeros(len(X_pool))
    return query_idx, query, uncertainty_estimates

def lc_sampling(
    model,
    X_pool: Union[np.ndarray, Dataset, TransformersDataset],
    n_instances: int,
    **kwargs,
):
    """
    uncertainty estimates: the larger the value, the less confident the model is
    :param classifier:
    :param X_pool:
    :param n_instances:
    :param kwargs:
    :return:
    """
    probas = model.predict_proba(X_pool)
    uncertainty_estimates = 1 - np.max(probas, axis=1)
    argsort = np.argsort(-uncertainty_estimates)
    query_idx = argsort[:n_instances]
    query = take_idx(X_pool, query_idx)

    return query_idx, query, uncertainty_estimates


def entropy_sampling(
    model,
    X_pool: Union[np.ndarray, Dataset, TransformersDataset],
    n_instances: int,
    **kwargs,
):
    """
    uncertainty estimates: the larger the value, the less confident the model is
    :param classifier:
    :param X_pool:
    :param n_instances:
    :param kwargs:
    :return:
    """
    probas = model.predict_proba(X_pool)
    uncertainty_estimates = np.sum(-probas * np.log(probas), axis=1)
    argsort = np.argsort(-uncertainty_estimates)
    query_idx = argsort[:n_instances]
    query = take_idx(X_pool, query_idx)

    return query_idx, query, uncertainty_estimates


def mnlp_sampling(
    model,
    X_pool: Union[np.ndarray, Dataset, TransformersDataset],
    n_instances: int,
    select_by_number_of_tokens=True,
    **kwargs,
):
    probas = model.predict_proba(X_pool, remove_padding=True)
    np.array([-np.sum(np.log(np.max(i, axis=1))) / len(i) for i in probas])

    uncertainty_estimates = calculate_mnlp_score(probas)
    argsort = np.argsort(-uncertainty_estimates)

    if select_by_number_of_tokens:
        query_idx = get_query_idx_for_selecting_by_number_of_tokens(
            X_pool, argsort, n_instances
        )
    else:
        query_idx = argsort[:n_instances]

    query = take_idx(X_pool, query_idx)
    return query_idx, query, uncertainty_estimates


def old_mahalanobis_sampling(
    model,
    X_pool,
    n_instances,
    **mahalanobis_kwargs,
):
    kwargs = dict(
        # Necessary
        model_wrapper=model,
        data=X_pool,
        # General
        data_is_tokenized=False,
        data_config=None,
        batch_size=model._batch_size_kwargs.eval_batch_size,
        to_numpy=True,
        # Mahalanobis
        use_v2=mahalanobis_kwargs["use_v2"],
        use_activation=mahalanobis_kwargs["use_activation"],
        use_spectralnorm=mahalanobis_kwargs["use_spectralnorm"],
    )
    dists = calculate_mahalanobis_distance(**kwargs)
    uncertainty_estimates = np.min(dists, axis=-1)
    argsort = np.argpartition(-uncertainty_estimates, kth=n_instances)
    query_idx = argsort[:n_instances]
    query = take_idx(X_pool, query_idx)

    return query_idx, query, uncertainty_estimates


def mahalanobis_sampling(
    model,
    X_pool: Union[Dataset, TransformersDataset],
    n_instances: int,
    X_train: Union[Dataset, TransformersDataset],
    **mahalanobis_kwargs,
):
    batched = mahalanobis_kwargs.get("mahalanobis_batched", False)
    substrategy = mahalanobis_kwargs.get("mahalanobis_substrategy", "lc")
    use_class_probas = mahalanobis_kwargs.get("mahalanobis_use_class_probas", False)
    kwargs = dict(
        # Necessary
        model_wrapper=model,
        train_data=X_train,
        unlabeled_data=X_pool,
        # General
        data_is_tokenized=False,
        data_config=None,
        batch_size=model._batch_size_kwargs.eval_batch_size,
        to_numpy=True,
        # Mahalanobis
        classwise=True,  # use an own centroid for each class
        batched=batched,
        use_v2=mahalanobis_kwargs.get("mahalanobis_use_v2", False),
        use_activation=mahalanobis_kwargs.get("mahalanobis_use_activation", False),
        use_spectralnorm=mahalanobis_kwargs.get("mahalanobis_use_spectralnorm", True),
    )
    if not batched:
        dists = calculate_mahalanobis_distance(**kwargs)
        if use_class_probas:
            log.info("Using probas for mahalanobis")
            label_name = model.data_config["label_name"]
            class_probas = np.bincount(
                X_train[label_name], minlength=model.num_labels
            ) / len(X_train)
            dists = dists * class_probas
        else:
            log.info("Not using probas for mahalanobis))")
        if substrategy == "lc":
            uncertainty_estimates = np.min(dists, axis=-1)
        elif substrategy == "margin":
            dists.sort(axis=1)
            max_dists = dists[:, 0]
            second_max_dists = dists[:, 1]
            uncertainty_estimates = (
                max_dists - second_max_dists
            )  # the greater the value, the more uncertain we are
        else:
            raise NotImplementedError
        argsort = np.argpartition(-uncertainty_estimates, kth=n_instances)
        query_idx = argsort[:n_instances]
    else:
        query_idx, uncertainty_estimates = calculate_mahalanobis_distance(
            n_instances=n_instances, **kwargs
        )
    query = take_idx(X_pool, query_idx)
    return query_idx, query, uncertainty_estimates


def mahalanobis_triplet_sampling(
    model,
    X_pool: Union[Dataset, TransformersDataset],
    n_instances: int,
    X_train: Union[Dataset, TransformersDataset],
    **mahalanobis_triplet_kwargs,
):
    batched = mahalanobis_triplet_kwargs.get("mahalanobis_batched", False)
    kwargs = dict(
        # Necessary
        model_wrapper=model,
        train_data=X_train,
        unlabeled_data=X_pool,
        # General
        data_is_tokenized=False,
        data_config=None,
        batch_size=model._batch_size_kwargs.eval_batch_size,
        to_numpy=True,
        # Mahalanobis Triplet
        batched=batched,
        lamb=mahalanobis_triplet_kwargs.get("mahalanobis_triplet_lamb", 0.25),
        use_finetuned=mahalanobis_triplet_kwargs.get("mahalanobis_use_finetuned", True),
        use_v2=mahalanobis_triplet_kwargs.get("mahalanobis_use_v2", False),
        use_activation=mahalanobis_triplet_kwargs.get(
            "mahalanobis_use_activation", False
        ),
        use_spectralnorm=mahalanobis_triplet_kwargs.get(
            "mahalanobis_use_spectralnorm", True
        ),
    )
    if not batched:
        uncertainty_estimates = calculate_mahalanobis_triplet_scores(**kwargs)
        argsort = np.argpartition(-uncertainty_estimates, kth=n_instances)
        query_idx = argsort[:n_instances]
    else:
        query_idx, uncertainty_estimates = calculate_mahalanobis_triplet_scores(
            n_instances=n_instances, **kwargs
        )

    query = take_idx(X_pool, query_idx)
    return query_idx, query, uncertainty_estimates


def triplet_sampling(
    model,
    X_pool: Union[Dataset, TransformersDataset],
    n_instances: int,
    X_train: Union[Dataset, TransformersDataset],
    **triplet_kwargs,
):
    kwargs = dict(
        # Necessary
        model_wrapper=model,
        train_data=X_train,
        unlabeled_data=X_pool,
        # General
        data_is_tokenized=False,
        data_config=None,
        batch_size=model._batch_size_kwargs.eval_batch_size,
        # Mahalanobis Triplet
        lamb=triplet_kwargs.get("triplet_lamb", 0.25),
        strategy=triplet_kwargs.get("uncertainty_strategy", "lc"),
        use_finetuned=triplet_kwargs.get("mahalanobis_use_finetuned", True),
        scale_distances=triplet_kwargs.get("scale_distances", True),
        use_activation=triplet_kwargs.get("mahalanobis_use_activation", False),
        use_spectralnorm=triplet_kwargs.get("mahalanobis_use_spectralnorm", True),
    )
    uncertainty_estimates = calculate_triplet_scores(**kwargs)
    argsort = np.argpartition(-uncertainty_estimates, kth=n_instances)
    query_idx = argsort[:n_instances]

    query = take_idx(X_pool, query_idx)
    return query_idx, query, uncertainty_estimates


def mahalanobis_filtering_sampling(
    model,
    X_pool: Union[Dataset, TransformersDataset],
    n_instances: int,
    X_train: Union[Dataset, TransformersDataset],
    **filtering_kwargs,
):
    batched = False
    if "mahalanobis_batched" in filtering_kwargs:
        batched = filtering_kwargs.pop("mahalanobis_batched")
    kwargs = dict(
        # Necessary
        model_wrapper=model,
        train_data=X_train,
        unlabeled_data=X_pool,
        # General
        data_is_tokenized=False,
        data_config=None,
        batch_size=model._batch_size_kwargs.eval_batch_size,
        # Mahalanobis Filtering
        filtering_share=filtering_kwargs.get("filtering_share", 0.01),
        strategy=filtering_kwargs.get("uncertainty_strategy", "lc"),
        batched=batched,
        n_instances=n_instances,
        use_finetuned=filtering_kwargs.get("mahalanobis_use_finetuned", True),
        use_activation=filtering_kwargs.get("mahalanobis_use_activation", False),
        use_spectralnorm=filtering_kwargs.get("mahalanobis_use_spectralnorm", True),
    )
    if batched:
        query_idx, uncertainty_estimates = calculate_mahalanobis_filtering_scores(
            **kwargs
        )
    else:
        uncertainty_estimates = calculate_mahalanobis_filtering_scores(**kwargs)
        argsort = np.argpartition(-uncertainty_estimates, kth=n_instances)
        query_idx = argsort[:n_instances]

    query = take_idx(X_pool, query_idx)
    return query_idx, query, uncertainty_estimates


def badge_sampling(
    model,
    X_pool: Union[Dataset, TransformersDataset],
    n_instances: int,
    X_train: Union[Dataset, TransformersDataset],
    **badge_kwargs,
):
    logits = model.predict_logits(X_pool)

    kwargs = dict(
        # Necessary
        model_wrapper=model,
        data_test=X_pool,
        # General
        data_is_tokenized=False,
        data_config=None,
        batch_size=model._batch_size_kwargs.eval_batch_size,
        to_numpy=True,
        logits=logits,
        # train_probas=train_logits,
    )

    scores_or_vectors = calculate_badge_scores(**kwargs)

    # cluster-based sampling method like BADGE and ALPS
    vectors = normalize(scores_or_vectors)
    # centers = _sampled.tolist()

    clustering = badge

    query_idx = np.array(clustering(vectors, k=n_instances))

    query = take_idx(X_pool, query_idx)

    # TODO
    # Define Uncertainty estimates for BADGE sampling
    uncertainty_estimates = np.zeros(len(X_pool))

    return query_idx, query, uncertainty_estimates


def alps_sampling(
    model,
    X_pool: Union[Dataset, TransformersDataset],
    n_instances: int,
    X_train: Union[Dataset, TransformersDataset],
    **alps_kwargs,
):
    logits = model.predict_logits(X_pool)

    uncertainty_estimates = 1 - np.max(logits, axis=1)
    kwargs = dict(
        # Necessary
        model_wr=model,
        dataloader_or_data=X_pool,
        # General
        data_is_tokenized=False,
        # data_config=None,
        batch_size=model._batch_size_kwargs.eval_batch_size,
        to_numpy=True,
        logits=logits,
        # train_probas=train_logits,
        # data_is_tokenized=data_is_tokenized,
        tokenizer=model.tokenizer,
        task=model.task,
        # text_name=data_config["text_name"],
        # label_name=data_config["label_name"],
    )
    scores_or_vectors = calculate_alps_scores(**kwargs)
    # cluster-based sampling method like BADGE and ALPS
    vectors = normalize(scores_or_vectors)
    clustering = kmeans

    query_idx = np.array(
        clustering(
            # vectors[unsampled], k = args.query_size
            vectors,
            k=n_instances,
        )
    )
    # add new samples to previously sampled list
    query = take_idx(X_pool, query_idx)

    return query_idx, query, uncertainty_estimates


def cal_sampling(
    model,
    X_pool: Union[Dataset, TransformersDataset],
    n_instances: int,
    X_train: Union[Dataset, TransformersDataset],
    **cal_kwargs,
):
    logits = model.predict_logits(X_pool)
    train_logits = model.predict_logits(X_train)
    kwargs = dict(
        # Necessary
        model_wrapper=model,
        data_train=X_train,
        data_test=X_pool,
        # General
        data_is_tokenized=False,
        data_config=None,
        batch_size=model._batch_size_kwargs.eval_batch_size,
        to_numpy=True,
        probas=logits,
        train_probas=train_logits,
        num_nei=cal_kwargs["num_nei"],
    )

    uncertainty_estimates = calculate_cal_scores(**kwargs)
    # argsort = np.argsort(-uncertainty_estimates)
    # query_idx = argsort[:n_instances]
    query_idx = np.argpartition(uncertainty_estimates, -n_instances)[-n_instances:]
    query = take_idx(X_pool, query_idx)
    return query_idx, query, uncertainty_estimates


def ddu_sampling(
    model,
    X_pool: Union[Dataset, TransformersDataset],
    n_instances: int,
    X_train: Union[Dataset, TransformersDataset],
    **ddu_kwargs,
):
    kwargs = dict(
        # Necessary
        model_wrapper=model,
        data_train=X_train,
        data_test=X_pool,
        # General
        data_is_tokenized=False,
        data_config=None,
        batch_size=model._batch_size_kwargs.eval_batch_size,
        to_numpy=True,
        # DDU
        use_activation=ddu_kwargs["use_activation"],
        use_spectralnorm=ddu_kwargs["use_spectralnorm"],
    )
    uncertainty_estimates = calculate_ddu_scores(**kwargs)
    argsort = np.argpartition(uncertainty_estimates, kth=n_instances)
    query_idx = argsort[:n_instances]
    query = take_idx(X_pool, query_idx)

    return query_idx, query, -uncertainty_estimates


def logits_lc_sampling(
    model,
    X_pool: Union[np.ndarray, Dataset, TransformersDataset],
    n_instances: int,
    **kwargs,
):
    probas = model.predict_logits(X_pool)
    uncertainty_estimates = 1 - np.max(probas, axis=1)
    argsort = np.argsort(-uncertainty_estimates)
    query_idx = argsort[:n_instances]
    query = take_idx(X_pool, query_idx)

    return query_idx, query, uncertainty_estimates


def margin_sampling(
    model,
    X_pool: Union[np.ndarray, Dataset, TransformersDataset],
    n_instances: int,
    **kwargs,
):
    probas = model.predict_proba(X_pool)
    # To get second max probas, need to sort the array since `.sort` modifies the array
    probas.sort(axis=1)
    max_probas = probas[:, -1]
    second_max_probas = probas[:, -2]
    uncertainty_estimates = 1 + second_max_probas - max_probas
    argsort = np.argsort(-uncertainty_estimates)
    query_idx = argsort[:n_instances]
    query = X_pool.select(query_idx)

    return query_idx, query, uncertainty_estimates


def oracle_sampling(
    model,
    X_pool: Union[np.ndarray, Dataset, TransformersDataset],
    n_instances: int,
    **kwargs,
):
    """
    ¡NB! Works only with `transformers` framework
    """
    probas = model.predict_proba(X_pool)
    labels = X_pool["label"]
    uncertainty_estimates = -np.log([pr[lab] for pr, lab in zip(probas, labels)])
    argsort = np.argsort(-uncertainty_estimates)
    query_idx = argsort[:n_instances]
    query = take_idx(X_pool, query_idx)

    return query_idx, query, uncertainty_estimates


def bald_sampling(
    model,
    X_pool: Union[np.ndarray, Dataset, TransformersDataset],
    n_instances: int,
    select_by_number_of_tokens: bool = False,
    **bald_kwargs,
):
    mc_iterations = bald_kwargs.get("mc_iterations", 10)
    use_stable_dropout = bald_kwargs.get("use_stable_dropout", 10)

    # TODO: realize whether is required or not
    # Make dropout consistent inside huggingface model
    if use_stable_dropout:
        make_dropouts_consistent(model.model)
    else:
        model.enable_dropout()

    if bald_kwargs.get("only_head_dropout", False):
        raise NotImplementedError
        # log_probs_N_K_C = model.predict_proba(X_pool, mc_dropout=True, mc_iterations=mc_iterations)
    else:
        # Stable dropout
        probas = []
        for _ in range(mc_iterations):
            if use_stable_dropout:
                # Reset masks
                model.enable_dropout()
                model.disable_dropout()
            probas_iter = model.predict_proba(
                X_pool, use_predict_loop=True, to_eval_mode=False
            )
            probas.append(probas_iter)

        probas_N_K_C = np.stack(probas, -2)

    uncertainty_estimates = calculate_bald_score_cls(probas_N_K_C)
    # The larger the score, the more confident the model is
    argsort = np.argsort(-uncertainty_estimates)

    if select_by_number_of_tokens:
        query_idx = get_query_idx_for_selecting_by_number_of_tokens(
            X_pool, argsort, n_instances
        )
    else:
        query_idx = argsort[:n_instances]
    query = take_idx(X_pool, query_idx)

    return query_idx, query, uncertainty_estimates


def batchbald_sampling(
    model,
    X_pool: Union[np.ndarray, Dataset, TransformersDataset],
    n_instances: int,
    **bald_kwargs,
):

    mc_iterations = bald_kwargs.get("mc_iterations", 10)
    max_num_samples = bald_kwargs.get("max_num_samples", int(1e4))
    # requires enormous amount of memory
    device = "cpu"  # list(model.model.parameters())[0].device

    # Make dropout consistent inside huggingface model
    make_dropouts_consistent(model.model)

    if bald_kwargs.get("only_head_dropout", False):
        raise NotImplementedError
        # log_probas_N_K_C = model.predict_proba(X_pool, mc_dropout=True, mc_iterations=mc_iterations)
    else:
        probas = []
        for _ in range(mc_iterations):
            # Reset masks
            model.enable_dropout()
            model.disable_dropout()
            probas_iter = model.predict_proba(
                X_pool, use_predict_loop=True, to_numpy=False, to_eval_mode=False
            )
            probas.append(probas_iter)

        log_probas_N_K_C = torch.stack(probas, -2).log().to(device)

    query_idx, uncertainty_estimates = get_batchbald_batch(
        log_probas_N_K_C, n_instances, max_num_samples, device=device
    )
    query = take_idx(X_pool, query_idx)

    return query_idx, query, uncertainty_estimates


def cluster_margin(classifier, X_pool, n_instances, **kwargs):
    instances_multiplier = kwargs["instances_multiplier"]
    random_query = kwargs["random_query"]
    probas = classifier.predict_proba(X_pool)
    # reuse code from margin sampling
    # To get second max probas, need to sort the array since `.sort` modifies the array
    probas.sort(axis=1)
    max_probas = probas[:, -1]
    second_max_probas = probas[:, -2]
    uncertainty_estimates = 1 + second_max_probas - max_probas
    argsort = np.argsort(-uncertainty_estimates)
    # we have to choose n_instances with round-robin algorithm, so for this we
    # firstly choose 2 * n_instances
    query_idx = argsort[: int(instances_multiplier * n_instances)]
    cluster_labels = np.array(X_pool.clusters)[query_idx]
    # X_pool either transformer dataset or np array
    query_idx, samples_idx = choose_cm_samples(
        cluster_labels, query_idx, n_instances, random_query
    )
    query = take_idx(X_pool, query_idx)
    return query_idx, query, uncertainty_estimates


# def de_ue_cls(
#     classifiers,
#     X_pool,
#     n_instances,
#     gamma_or_k_confident_to_save=0.25,
#     T=None,
#     sampling_type=None,
#     require_sampling=True,
#     method="bald",
#     **kwargs,
# ):
#     prob_arr = np.stack(
#         [classifier.predict_proba(X_pool) for classifier in classifiers], axis=1
#     )
#     if method == "bald":
#         scores = calculate_bald_score_cls(prob_arr)
#     elif method == "vr":
#         scores = var_ratio(prob_arr)
#     elif method == "mean_ent":
#         scores = mean_entropy(prob_arr)
#     elif method == "max_prob":
#         scores = sampled_max_prob(prob_arr)
#     elif method == "prob_var":
#         scores = probability_variance(prob_arr)
#     else:
#         raise NotImplementedError
#
#     argsort = np.argsort(-scores)
#     query_idx = argsort[:n_instances]
#     query = (
#         X_pool[query_idx]
#         if isinstance(X_pool, np.ndarray)
#         else X_pool.select(query_idx)
#     )
#
#     if (sampling_type is None) or (not require_sampling):
#         return query_idx, query
#
#     elif sampling_type == "naive":
#         k_confident_idx = argsort[
#             n_instances : int(gamma_or_k_confident_to_save * len(argsort)) + n_instances
#         ]
#     elif sampling_type == "ups":
#         k_confident_idx = sample_ups(
#             argsort, query_idx, gamma_or_k_confident_to_save, T
#         )
#     elif sampling_type == "random":
#         idx_to_select_from = np.setdiff1d(argsort, query_idx)
#         share_to_sample = gamma_or_k_confident_to_save
#         k_confident_idx = np.random.choice(
#             idx_to_select_from,
#             int(share_to_sample * len(idx_to_select_from)),
#             replace=False,
#         )
#     else:
#         raise NotImplementedError
#
#     k_confident_query = (
#         X_pool[k_confident_idx]
#         if isinstance(X_pool, np.ndarray)
#         else X_pool.select(k_confident_idx)
#     )
#     return query_idx, query, k_confident_idx, k_confident_query
#
#
# def de_ue_ner(
#     classifiers,
#     X_pool: Union[np.ndarray, Dataset, TransformersDataset],
#     n_instances: int,
#     gamma_or_k_confident_to_save: float = 0.25,
#     T: float = None,
#     sampling_type: str = None,
#     select_by_number_of_tokens=True,
#     require_sampling: bool = True,
#     method="bald",
#     **kwargs,
# ):
#     prob_arr = np.stack(
#         [classifier.predict_proba(X_pool) for classifier in classifiers], axis=1
#     )
#     if method == "bald":
#         scores = calculate_bald_score_ner(prob_arr)
#     # elif method == "vr":
#     #     scores = var_ratio(prob_arr)
#     elif method == "mean_ent":
#         scores = mean_entropy_ner(prob_arr)
#     # elif method == "max_prob":
#     #     scores = sampled_max_prob(prob_arr)
#     elif method == "prob_var":
#         scores = probability_variance_ner(prob_arr)
#     else:
#         raise NotImplementedError
#
#     argsort = np.argsort(-scores)
#     # How we select instances from the pool: with `__getitem__` (for np.ndarray) or with `select` (for Dataset)
#     method_name = "__getitem__" if isinstance(X_pool, np.ndarray) else "select"
#     select_fn = getattr(X_pool, method_name)
#
#     if select_by_number_of_tokens:
#         query_idx = get_query_idx_for_selecting_by_number_of_tokens(
#             X_pool, argsort, n_instances
#         )
#     else:
#         query_idx = argsort[:n_instances]
#
#     query = select_fn(query_idx)
#
#     if (sampling_type is None) or (not require_sampling):
#         return query_idx, query
#
#     elif sampling_type == "naive" and select_by_number_of_tokens:
#         ### Get cumulative sum of tokens; Determine how many we need to query;
#         ### Since a part of them will be queried anyway, take as many tokens, as the query will contain;
#         ### Find idx of the last instance that need to be queried; Query (including the last one).
#         cumsum_tokens = np.cumsum([len(x["tokens"].tokens) for x in select_fn(argsort)])
#         num_tokens_to_select = gamma_or_k_confident_to_save * cumsum_tokens[-1]
#         tokens_bound = num_tokens_to_select + sum(
#             [len(x["tokens"].tokens) for x in select_fn(query_idx)]
#         )
#         idx_last_plus_one = np.argwhere(cumsum_tokens > tokens_bound)[0][0] + 1
#         k_confident_idx = argsort[len(query_idx) : idx_last_plus_one]
#     elif sampling_type == "naive":  # not select by tokens
#         k_confident_idx = argsort[
#             n_instances : int(gamma_or_k_confident_to_save * len(argsort)) + n_instances
#         ]
#     elif sampling_type == "ups":
#         k_confident_idx = sample_ups(
#             argsort, query_idx, gamma_or_k_confident_to_save, T
#         )
#     elif sampling_type == "random":
#         idx_to_select_from = np.setdiff1d(argsort, query_idx)
#         share_to_sample = gamma_or_k_confident_to_save
#         k_confident_idx = np.random.choice(
#             idx_to_select_from,
#             int(share_to_sample * len(idx_to_select_from)),
#             replace=False,
#         )
#     else:
#         raise NotImplementedError
#
#     k_confident_query = select_fn(k_confident_idx)
#     return query_idx, query, k_confident_idx, k_confident_query
#
#
# # TODO: refactor
# def bald_sampling(
#     model,
#     X_pool,
#     n_instances,
#     gamma_or_k_confident_to_save=0.25,
#     T=None,
#     sampling_type=None,
#     select_by_number_of_tokens=True,
#     require_sampling=True,
#     only_head_dropout=True,
#     mc_iterations=10,
#     **kwargs,
# ):
#     if only_head_dropout:
#         prob_arr = np.array(
#             model.predict_proba(X_pool, mc_dropout=True, mc_iterations=mc_iterations)
#         )
#     else:
#         model.enable_dropout()
#         prob_arr = np.stack(
#             [model.predict_proba(X_pool) for _ in range(mc_iterations)], axis=0
#         )
#
#     scores = calculate_bald_score_ner(np.array(prob_arr))
#     argsort = np.argsort(-scores)
#
#     # How we select instances from the pool: with `__getitem__` (for np.ndarray) or with `select` (for Dataset)
#     method_name = "__getitem__" if isinstance(X_pool, np.ndarray) else "select"
#     select_fn = getattr(X_pool, method_name)
#
#     if select_by_number_of_tokens:
#         query_idx = get_query_idx_for_selecting_by_number_of_tokens(
#             X_pool, argsort, n_instances
#         )
#     else:
#         query_idx = argsort[:n_instances]
#
#     query = select_fn(query_idx)
#
#     if (sampling_type is None) or (not require_sampling):
#         return query_idx, query
#
#     elif sampling_type == "naive" and select_by_number_of_tokens:
#         ### Get cumulative sum of tokens; Determine how many we need to query;
#         ### Since a part of them will be queried anyway, take as many tokens, as the query will contain;
#         ### Find idx of the last instance that need to be queried; Query (including the last one).
#         cumsum_tokens = np.cumsum([len(x["tokens"].tokens) for x in select_fn(argsort)])
#         num_tokens_to_select = gamma_or_k_confident_to_save * cumsum_tokens[-1]
#         tokens_bound = num_tokens_to_select + sum(
#             [len(x["tokens"].tokens) for x in select_fn(query_idx)]
#         )
#         idx_last_plus_one = np.argwhere(cumsum_tokens > tokens_bound)[0][0] + 1
#         k_confident_idx = argsort[len(query_idx) : idx_last_plus_one]
#     elif sampling_type == "naive":  # not select by tokens
#         k_confident_idx = argsort[
#             n_instances : int(gamma_or_k_confident_to_save * len(argsort)) + n_instances
#         ]
#     elif sampling_type == "ups":
#         k_confident_idx = sample_ups(
#             argsort, query_idx, gamma_or_k_confident_to_save, T
#         )
#     elif sampling_type == "random":
#         idx_to_select_from = np.setdiff1d(argsort, query_idx)
#         share_to_sample = gamma_or_k_confident_to_save
#         k_confident_idx = np.random.choice(
#             idx_to_select_from,
#             int(share_to_sample * len(idx_to_select_from)),
#             replace=False,
#         )
#     else:
#         raise NotImplementedError
#
#     k_confident_query = select_fn(k_confident_idx)
#     return query_idx, query, k_confident_idx, k_confident_query


def ddu_sampling_cv(
    classifier,
    X_pool: Union[Dataset, TransformersDataset],
    n_instances: int,
    X_train: Union[Dataset, TransformersDataset],
    **ddu_kwargs,
):
    kwargs = dict(
        # Necessary
        model_wrapper=classifier,
        data_train=X_train,
        data_test=X_pool,
        # General
        data_is_tokenized=False,
        data_config=None,
        batch_size=classifier._batch_size_kwargs.eval_batch_size,
        # DDU
        use_activation=ddu_kwargs["use_activation"],
        use_spectralnorm=ddu_kwargs["use_spectralnorm"],
        to_numpy=True,
    )
    scores = calculate_ddu_scores_cv(**kwargs)
    uncertainty_estimates = 1 - scores
    argsort = np.argpartition(scores, kth=n_instances)
    query_idx = argsort[:n_instances]
    query = take_idx(X_pool, query_idx)

    return query_idx, query, uncertainty_estimates


def get_top_k_scorers(scores_N, batch_size, uncertainty=True):
    N = len(scores_N)
    batch_size = min(batch_size, N)
    candidate_scores, candidate_indices = torch.topk(
        scores_N, batch_size, largest=uncertainty
    )
    return candidate_scores.tolist(), candidate_indices.tolist()
