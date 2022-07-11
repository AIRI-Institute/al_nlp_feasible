import numpy as np

from .al_strategy_utils import get_ups_sampling_probas, sample_idxs


def ups_subsampling(uncertainty_estimates, gamma_or_k_confident_to_save, T):
    if isinstance(gamma_or_k_confident_to_save, int):
        gamma_or_k_confident_to_save /= len(uncertainty_estimates)
    argsort = np.argsort(-uncertainty_estimates)
    sampling_probas = get_ups_sampling_probas(argsort, gamma_or_k_confident_to_save, T)
    return sample_idxs(sampling_probas)


def random_subsampling(uncertainty_estimates, gamma_or_k_confident_to_save, **kwargs):
    length = len(uncertainty_estimates)
    if isinstance(gamma_or_k_confident_to_save, float):
        gamma_or_k_confident_to_save = int(gamma_or_k_confident_to_save * length)
    if gamma_or_k_confident_to_save >= length:
        return np.arange(length)
    return np.random.choice(
        np.arange(length), gamma_or_k_confident_to_save, replace=False
    )


def naive_subsampling(uncertainty_estimates, gamma_or_k_confident_to_save, **kwargs):
    if isinstance(gamma_or_k_confident_to_save, float):
        gamma_or_k_confident_to_save = int(
            gamma_or_k_confident_to_save * len(uncertainty_estimates)
        )
    argsort = np.argsort(-uncertainty_estimates)
    return argsort[:gamma_or_k_confident_to_save]
