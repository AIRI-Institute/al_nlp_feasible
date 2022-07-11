from copy import deepcopy
from functools import partial
from pathlib import Path

from ..al.learner import ActiveLearner
from ..al.al_strategy import (
    random_sampling,
    lc_sampling,
    entropy_sampling,
    mnlp_sampling,
    mahalanobis_sampling,
    mahalanobis_triplet_sampling,
    triplet_sampling,
    mahalanobis_filtering_sampling,
    ddu_sampling,
    logits_lc_sampling,
    margin_sampling,
    oracle_sampling,
    bald_sampling,
    batchbald_sampling,
    alps_sampling,
    # de_ue_cls,
    # de_ue_ner,
    cal_sampling,
    cluster_margin,
    ddu_sampling_cv,
)

from ..al.sampling_strategy import (
    ups_subsampling,
    random_subsampling,
    naive_subsampling,
)


QUERY_STRATEGIES = {
    # Classification strategies
    "random": partial(random_sampling, select_by_number_of_tokens=False),
    "entropy": entropy_sampling,
    "lc": lc_sampling,
    "logits_lc": logits_lc_sampling,
    "mahalanobis": mahalanobis_sampling,
    "mahalanobis_triplet": mahalanobis_triplet_sampling,
    "mahalanobis_filtering": mahalanobis_filtering_sampling,
    "triplet": triplet_sampling,
    "ddu": ddu_sampling,
    "margin": margin_sampling,
    "cal": cal_sampling,
    "oracle": oracle_sampling,
    "cluster_margin": cluster_margin,
    "alps": alps_sampling,
    # Classification Ensemble strategies
    # "bald_de_cls": partial(de_ue_cls, method="bald"),
    # "vr_de_cls": partial(de_ue_cls, method="vr"),
    # "mean_ent_de_cls": partial(de_ue_cls, method="mean_ent"),
    # "max_prob_de_cls": partial(de_ue_cls, method="max_prob"),
    # "prob_var_de_cls": partial(de_ue_cls, method="prob_var"),
    # NER Ensemble strategies
    # "bald_de_ner_tokens": partial(
    #     de_ue_ner, select_by_number_of_tokens=True, method="bald"
    # ),
    # "prob_var_de_ner_tokens": partial(
    #     de_ue_ner, select_by_number_of_tokens=True, method="prob_var"
    # ),
    # "mean_ent_de_ner_tokens": partial(
    #     de_ue_ner, select_by_number_of_tokens=True, method="mean_ent"
    # ),
    # NER strategies
    "mnlp_tokens": partial(mnlp_sampling, select_by_number_of_tokens=True),
    "mnlp_samples": partial(mnlp_sampling, select_by_number_of_tokens=False),
    "random_tokens": partial(random_sampling, select_by_number_of_tokens=True),
    "random_samples": partial(random_sampling, select_by_number_of_tokens=False),
    # BALD
    "bald": partial(
        bald_sampling, select_by_number_of_tokens=False, only_head_dropout=False
    ),
    "bald_head": partial(
        bald_sampling, select_by_number_of_tokens=False, only_head_dropout=True
    ),
    "bald_tokens": partial(
        bald_sampling, select_by_number_of_tokens=True, only_head_dropout=False
    ),
    "bald_samples": partial(
        bald_sampling, select_by_number_of_tokens=False, only_head_dropout=False
    ),
    "bald_tokens_head": partial(
        bald_sampling, select_by_number_of_tokens=True, only_head_dropout=True
    ),
    "bald_samples_head": partial(
        bald_sampling, select_by_number_of_tokens=False, only_head_dropout=True
    ),
    # BatchBald
    "batchbald": partial(
        batchbald_sampling, select_by_number_of_tokens=False, only_head_dropout=False
    ),
    "batchbald_head": partial(
        batchbald_sampling, select_by_number_of_tokens=False, only_head_dropout=True
    ),
    "batchbald_tokens": partial(
        batchbald_sampling, select_by_number_of_tokens=True, only_head_dropout=False
    ),
    "batchbald_samples": partial(
        batchbald_sampling, select_by_number_of_tokens=False, only_head_dropout=False
    ),
    "batchbald_tokens_head": partial(
        batchbald_sampling, select_by_number_of_tokens=True, only_head_dropout=True
    ),
    "batchbald_samples_head": partial(
        batchbald_sampling, select_by_number_of_tokens=False, only_head_dropout=True
    ),
    # CV strategies
    "ddu_cv": ddu_sampling_cv,
}

sampling_strategies = {
    "ups": ups_subsampling,
    "random": random_subsampling,
    "naive": naive_subsampling,
}


def construct_active_learner(
    model, config, initial_data, log_dir: str or Path, framework: str = "transformers"
):

    # TODO: rewrite using `split_by_tokens` as `strategy_kwargs`
    initial_data_copy = deepcopy(initial_data)
    use_ups = config.sampling_type is not None
    postfix = ""
    if ("split_by_tokens" in config) and (config.split_by_tokens):
        postfix += "_tokens"
    elif "split_by_tokens" in config:  # avoid adding "_samples" for classification
        postfix += "_samples"

    if config.strategy == "bald" and getattr(config, "head_only_dropout", False):
        postfix += "_head"

    query_strategy = QUERY_STRATEGIES[f"{config.strategy}{postfix}"]
    sampling_strategy = sampling_strategies[config.sampling_type] if use_ups else None
    sampling_kwargs = {
        "gamma_or_k_confident_to_save": config.gamma_or_k_confident_to_save,
        "T": config.T,
    }
    strategy_kwargs = config.strategy_kwargs

    learner = ActiveLearner(
        estimator=model,
        query_strategy=query_strategy,
        train_data=initial_data_copy,
        strategy_kwargs=strategy_kwargs,
        sampling_strategy=sampling_strategy,
        sampling_kwargs=sampling_kwargs,
        framework=framework,
        log_dir=log_dir,
    )

    return learner
