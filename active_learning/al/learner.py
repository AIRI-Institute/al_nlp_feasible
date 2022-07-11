import numpy as np
from datasets import concatenate_datasets
from datasets.arrow_dataset import Dataset as ArrowDataset
from time import time
from pathlib import Path

from ..utils.general import json_dump
from ..utils.transformers_dataset import TransformersDataset
from .al_strategy_utils import take_idx, concatenate_data


class ActiveLearner:
    def __init__(
        self,
        estimator,
        query_strategy,
        train_data,
        strategy_kwargs=None,
        sampling_strategy=None,
        sampling_kwargs=None,
        framework="transformers",
        log_dir="logs/",
    ):
        self.estimator = estimator
        self.query_strategy = query_strategy
        self.train_data = train_data
        self.strategy_kwargs = strategy_kwargs if strategy_kwargs is not None else {}
        self.sampling_strategy = sampling_strategy
        self.sampling_kwargs = sampling_kwargs if sampling_kwargs is not None else {}
        self.framework = framework

        self._create_time_dict(log_dir)

        self._fit_estimator()

    def _fit_estimator(self, data=None):
        start_time = time()
        if data is not None:
            self.estimator.fit(data)
        else:
            self.estimator.fit(self.train_data)
        fit_time = time() - start_time
        self._add_obs_to_time_dict("fit", fit_time)

    def _add_data(self, new_data):
        if self.framework == "transformers":
            self.train_data = concatenate_data(self.train_data, new_data)
        else:  # framework == "allennlp"
            self.train_data = np.r_[self.train_data, new_data]

    # TODO: use decorator
    def query(self, X_pool, n_instances, **kwargs):
        strategy_kwargs = dict(self.strategy_kwargs)
        strategy_kwargs["X_train"] = self.train_data
        strategy_kwargs.update(kwargs)

        start_time = time()
        output = self.query_strategy(
            self.estimator, X_pool, n_instances, **strategy_kwargs
        )
        query_time = time() - start_time
        self._add_obs_to_time_dict("query", query_time)

        return output

    # TODO: use decorator
    def teach(self, data, fit_only_new=False):
        self._add_data(data)
        if fit_only_new:
            self._fit_estimator(data)
        else:
            self._fit_estimator()

    def sample(self, X_pool, uncertainty_estimates, gamma_or_k_confident_to_save=None):
        sampling_kwargs = dict(self.sampling_kwargs)
        if gamma_or_k_confident_to_save is not None:
            sampling_kwargs[
                "gamma_or_k_confident_to_save"
            ] = gamma_or_k_confident_to_save

        sample_idx = self.sampling_strategy(uncertainty_estimates, **sampling_kwargs)
        sample_instances = take_idx(X_pool, sample_idx)
        return sample_idx, sample_instances

    def _create_time_dict(self, log_dir: str or Path):
        self._time_dict = {"fit": [], "query": []}
        self._time_dict_path = Path(log_dir) / "al_learner_time_dict.json"
        if not Path(log_dir).exists():
            Path(log_dir).mkdir()
        json_dump(self._time_dict, self._time_dict_path)

    def _add_obs_to_time_dict(self, step: str, value: float):
        self._time_dict[step].append(value)
        json_dump(self._time_dict, self._time_dict_path)
