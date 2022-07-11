import torch
from tqdm.auto import tqdm
from typing import Tuple
import numpy as np

from .compute_entropy import compute_conditional_entropy
from .joint_entropy import DynamicJointEntropy


def get_batchbald_batch(
    log_probs_N_K_C: torch.Tensor,
    batch_size: int,
    max_num_samples: int,
    dtype=None,
    device=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """

    :param log_probs_N_K_C:
    :param batch_size:
    :param max_num_samples: how many queried samples to use to calculate the joint entropy
    :param calculation_batch_size: size of batch for calculation on `compute_batch` step
    :param dtype:
    :param device:
    :return:
    """
    N, K, C = log_probs_N_K_C.shape

    batch_size = min(batch_size, N)

    candidate_indices = []
    candidate_scores = []

    conditional_entropies_N = compute_conditional_entropy(log_probs_N_K_C)

    batch_joint_entropy = DynamicJointEntropy(
        max_num_samples, batch_size - 1, K, C, dtype=dtype, device=device
    )

    # We always keep these on the CPU.
    scores_N = torch.empty(N, dtype=torch.double, pin_memory=torch.cuda.is_available())

    for i in tqdm(range(batch_size), desc="BatchBALD", leave=False):
        if i > 0:
            latest_index = candidate_indices[-1]
            batch_joint_entropy.add_variables(
                log_probs_N_K_C[latest_index : latest_index + 1]
            )

        shared_conditinal_entropies = conditional_entropies_N[candidate_indices].sum()

        batch_joint_entropy.compute_batch(log_probs_N_K_C, output_entropies_B=scores_N)

        scores_N -= conditional_entropies_N + shared_conditinal_entropies
        scores_N[candidate_indices] = -float("inf")

        candidate_score, candidate_index = scores_N.max(dim=0)

        candidate_indices.append(candidate_index.item())
        candidate_scores.append(candidate_score.item())

    final_iter_scores = scores_N.numpy()
    final_iter_scores[candidate_indices] = candidate_scores

    return np.array(candidate_indices), final_iter_scores
