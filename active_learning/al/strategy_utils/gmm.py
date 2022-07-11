import torch
from tqdm import tqdm
from torch import nn

DOUBLE_INFO = torch.finfo(torch.double)
JITTERS = [0, DOUBLE_INFO.tiny] + [10**exp for exp in range(-10, 0, 1)]


def centered_cov(x):
    return x.T @ x / (len(x) - 1)


def compute_density(log_logits, label_probs):
    return torch.sum((torch.exp(log_logits / 768) * label_probs), dim=1)


def get_gmm_log_probs(gaussians_model, embeddings):
    return gaussians_model.log_prob(embeddings[:, None, :])


def gmm_fit(embeddings, labels):
    num_classes = len(set(labels))
    with torch.no_grad():
        centroids = torch.stack(
            [torch.mean(embeddings[labels == c], dim=0) for c in range(num_classes)]
        )
        cov_matrix = torch.stack(
            [
                centered_cov(embeddings[labels == c] - centroids[c])
                for c in range(num_classes)
            ]
        )

    with torch.no_grad():
        for jitter_eps in JITTERS:
            try:
                jitter = jitter_eps * torch.eye(
                    cov_matrix.shape[1],
                    device=cov_matrix.device,
                ).unsqueeze(0)
                gmm = torch.distributions.MultivariateNormal(
                    loc=centroids,
                    covariance_matrix=(cov_matrix + jitter),
                )
            except RuntimeError as e:
                if "cholesky" in str(e):
                    continue
            except ValueError as e:
                if "The parameter covariance_matrix has invalid values" in str(e):
                    continue
            break

    return gmm, jitter_eps


def centered_cov_torch(x):
    n = x.shape[0]
    res = 1 / (n - 1) * x.t().mm(x)
    return res


def compute_density_(logits, class_probs):
    return torch.sum((torch.exp(logits) * class_probs), dim=1)


def class_probs(data_loader):
    num_classes = 10
    class_n = len(data_loader.dataset)
    class_count = torch.zeros(num_classes)
    for batch in data_loader:
        label = batch["labels"]
        class_count += torch.Tensor([torch.sum(label == c) for c in range(num_classes)])

    class_prob = class_count / class_n
    return class_prob


def gmm_evaluate(net, gaussians_model, loader, device, num_classes, storage_device):

    num_samples = len(loader.dataset)
    logits_N_C = torch.empty(
        (num_samples, num_classes), dtype=torch.float, device=storage_device
    )
    labels_N = torch.empty(num_samples, dtype=torch.int, device=storage_device)

    with torch.no_grad():
        start = 0
        for batch in tqdm(loader):
            data = batch["image"].to(device)  # torch.Size([128, 1, 28, 28])
            label = batch["labels"].to(device)

            logit_B_C = gmm_forward(
                net, gaussians_model, data, label
            )  # torch.Size([128, 10])

            end = start + len(data)
            logits_N_C[start:end].copy_(logit_B_C, non_blocking=True)
            labels_N[start:end].copy_(label, non_blocking=True)
            start = end

    return logits_N_C, labels_N


def gmm_forward(net, gaussians_model, data_B_X, label):  # torch.Size([128, 1, 28, 28])

    if isinstance(net, nn.DataParallel):
        features_B_Z = net.module(data_B_X, label)
        features_B_Z = net.module.feature
    else:
        features_B_Z = net(data_B_X, label)
        features_B_Z = net.feature  # torch.Size([128, 512])

    log_probs_B_Y = gaussians_model.log_prob(
        features_B_Z[:, None, :]
    )  # torch.Size([128, 10])

    return log_probs_B_Y


def gmm_fit_(embeddings, labels, num_classes=10):
    with torch.no_grad():
        classwise_mean_features = torch.stack(
            [torch.mean(embeddings[labels == c], dim=0) for c in range(num_classes)]
        )
        classwise_cov_features = torch.stack(
            [
                centered_cov_torch(embeddings[labels == c] - classwise_mean_features[c])
                for c in range(num_classes)
            ]
        )

    with torch.no_grad():
        for jitter_eps in JITTERS:
            try:
                jitter = jitter_eps * torch.eye(
                    classwise_cov_features.shape[1],
                    device=classwise_cov_features.device,
                ).unsqueeze(0)
                gmm = torch.distributions.MultivariateNormal(
                    loc=classwise_mean_features,
                    covariance_matrix=(classwise_cov_features + jitter),
                )
            except RuntimeError as e:
                if "cholesky" in str(e):
                    continue
            except ValueError as e:
                if (
                    "The parameter covariance_matrix has invalid values"
                    or "Expected parameter covariance_matrix" in str(e)
                ):
                    continue
            except ValueError as e:
                if "Expected parameter covariance_matrix" in str(e):
                    continue
            break

    return gmm, jitter_eps
