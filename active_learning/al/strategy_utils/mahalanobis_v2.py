import torch
from tqdm import tqdm

JITTERS = [10**exp for exp in range(-15, 0, 1)]


def compute_inv_covariance_v2(centroids, train_features, train_labels, jitters=None):
    jitter = 0
    jitter_eps = None
    if jitters is None:
        jitters = JITTERS

    cov = torch.zeros(
        len(centroids),
        centroids.shape[1],
        centroids.shape[1],
        device=centroids.device,
        dtype=torch.float32,
    ).float()
    for c, mu_c in tqdm(enumerate(centroids)):
        features_of_class_c = train_features[train_labels == c]
        for x in features_of_class_c:
            d = (x - mu_c).unsqueeze(1)
            cov[c] += d @ d.T
        cov[c] /= len(features_of_class_c)

    for i, jitter_eps in enumerate(jitters):
        jitter = jitter_eps * torch.eye(cov.shape[1], device=cov.device)
        cov_update = cov + jitter
        eigenvalues = torch.symeig(cov_update).eigenvalues
        if (eigenvalues >= 0).all():
            break
    cov = cov + jitter
    cov_inv = torch.empty(cov.shape, dtype=torch.float32, device=cov.device)
    for i in range(len(centroids)):
        cov_inv[i].copy_(torch.inverse(cov[i].to(torch.float64)).float())

    return cov_inv, jitter_eps


def mahalanobis_distance_with_known_centroids_sigma_inv_v2(
    centroids, centroids_mask, sigmas_inv, eval_features
):
    device = eval_features.device
    diffs = eval_features.unsqueeze(1) - centroids.unsqueeze(
        0
    )  # bs (b), num_labels (c / l), dim (d / a)
    dists = (
        torch.sqrt(torch.einsum("bld,lda,bla->bl", diffs, sigmas_inv, diffs))
        .squeeze(-1)
        .squeeze(-1)
        .cpu()
    )
    dists = dists.masked_fill_(centroids_mask, float("inf")).to(device)
    return dists
