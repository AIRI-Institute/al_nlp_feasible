import numpy as np
from sklearn.cluster import AgglomerativeClustering
from collections import Counter
from .get_embeddings import get_embeddings
from ..al.al_strategy_utils import take_idx
import random
import logging
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import cdist
import fastcluster


log = logging.getLogger()


def hierarchial_ac(classifier, X_pool, X_train, config):
    """Wrapper for AgglomerativeClustering from scipy and fastcluster."""
    # get data embeddings
    kwargs = dict(
        # General
        prepare_model=True,
        use_activation=False,
        use_spectralnorm=False,
        data_is_tokenized=False,
        batch_size=classifier._batch_size_kwargs.eval_batch_size,
        to_numpy=True,
        # Tokenization
        tokenizer=classifier.tokenizer,
        task=classifier.task,
        text_name=classifier.data_config["text_name"],
        label_name=classifier.data_config["label_name"],
    )
    distance = config.al.strategy_kwargs["distance"]
    fast = config.al.strategy_kwargs["fast"]
    if fast:
        # in this case, we calc clusters only for part of data, and after calc centroids for these clusters
        # train_data = X_pool.select(np.random.choice(np.arange(len(X_pool)), int(0.1 * len(X_pool))))
        train_data = X_train
        embeddings = get_embeddings(classifier.model, train_data, **kwargs)
        n_clusters = config.al.strategy_kwargs["n_clusters"]
        log.info("Started clustering")
        linkage_matrix = fastcluster.linkage(
            embeddings, method="average", metric=distance
        )
        clusters = fcluster(linkage_matrix, t=n_clusters, criterion="maxclust")
        log.info("Done with clustering on train")
        centroids = {}
        unique_clusters = np.unique(clusters)
        for cluster in unique_clusters:
            # calc centroid for this cluster
            centroids[cluster] = np.average(
                embeddings[np.where(clusters == cluster)], axis=0
            )
        # after assign all embeddings to nearest cluster
        embeddings = get_embeddings(classifier.model, X_pool, **kwargs)
        centroids_values = np.array(list(centroids.values()))
        distances = cdist(centroids_values, embeddings, metric=distance)
        cluster_ids = np.argmin(distances, axis=0)
        clusters = np.array(list(centroids.keys()))[cluster_ids.astype(int)].astype(int)
        del embeddings, distances
        log.info("Done with clustering")
    else:
        train_data = X_pool
        embeddings = get_embeddings(classifier.model, train_data, **kwargs)
        n_clusters = config.al.strategy_kwargs["n_clusters"]
        log.info("Started clustering")
        linkage_matrix = fastcluster.linkage(
            embeddings, method="average", metric=distance
        )
        del embeddings
        clusters = fcluster(linkage_matrix, t=n_clusters, criterion="maxclust")
        log.info("Done with clustering")
    return clusters


"""
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
    query_idx = argsort[:int(instances_multiplier * n_instances)]
    cluster_labels = np.array(X_pool.clusters)[query_idx]
    # X_pool either transformer dataset or np array
    query_idx, samples_idx = choose_samples(cluster_labels, query_idx, n_instances, random_query)
    query = take_idx(X_pool, query_idx)
    return query_idx, query, uncertainty_estimates


def choose_samples(cluster_labels, query_idx, n_instances, random_query):
    # assume that we added cluster labels as coulmn in dataset
    # so we get samples, scores and clusters indices
    cluster_sizes = Counter(cluster_labels)
    new_query_idx = []
    samples_idx = []
    # split all query_idx to array by cluster label
    query_idx_by_clusters = {idx: list(query_idx[np.where(cluster_labels == idx)]) for idx in list(cluster_sizes.keys())}

    # try faster approach - collect all indices that we will sample
    samples_per_cluster = []
    avg_sample_per_cluster = np.ceil(n_instances / len(cluster_sizes))

    sorted_cluster = [el[1] for el in sorted([(v, k) for k, v in cluster_sizes.items()])]
    # else find smaller cluster, and sample its size, after subtract n_clusters * min_size and do same
    curr_idx = 0
    while sum(cluster_sizes.values()) > 0 and len(new_query_idx) < n_instances:
        # sample data from each cluster
        # sorted_cluster - array with cluster numbers in ascending order by size
        # cluster_labels - dict with cluster sizes as values
        curr_cluster = sorted_cluster[curr_idx]
        if cluster_sizes[curr_cluster] == 0:
            curr_idx = (curr_idx + 1) % len(sorted_cluster)
            continue
        # randomly sample from data with curr_cluster labels
        if random_query:
            sample_idx = random.choice(np.arange(len(query_idx_by_clusters[curr_cluster])))
        else:
            sample_idx = 0
        samples_idx.append(sample_idx)
        new_query_idx.append(query_idx_by_clusters[curr_cluster][sample_idx])
        # and remove this sample from data
        query_idx_by_clusters[curr_cluster] = np.delete(query_idx_by_clusters[curr_cluster], sample_idx)
        # after subtract 1 from this cluster
        cluster_sizes[curr_cluster] -= 1
        curr_idx = (curr_idx + 1) % len(sorted_cluster)
    return np.array(new_query_idx), samples_idx
"""
