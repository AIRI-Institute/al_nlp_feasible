import gc
import logging
import sys
import time
from pathlib import Path
from typing import Union, Tuple, List
from tqdm import tqdm
import os
import mlflow
import nltk
import numpy as np
from datasets import load_metric
from omegaconf.omegaconf import DictConfig
from collections import defaultdict

from nltk.stem import porter
from rouge_score import tokenize

import torch
from torch.nn.functional import softmax
from torch.utils.data import DataLoader

import transformers
from transformers import (
    set_seed,
    DataCollatorWithPadding,
    DataCollatorForTokenClassification,
    DataCollatorForSeq2Seq,
    Trainer,
    Seq2SeqTrainer,
    TrainingArguments,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback,
)

from .trainer_for_pseudo_labeled import TrainerForPseudoLabeled
from ...construct.transformers_api import create_transformers_model_tokenizer
from ...utils.general import (
    create_time_dict,
    json_dump,
    json_load,
    tensor_to_numpy,
    random_fixed_length_data_sampler,
    DictWithGetattr,
)
from ...utils.get_train_constants import get_train_constants
from ...utils.token_classification import align_labels_with_tokens

log = logging.getLogger()
transformers.logging.set_verbosity_info()

DATA_COLLATOR_CLASSES = {
    "cls": DataCollatorWithPadding,
    "ner": DataCollatorForTokenClassification,
    "abs-sum": DataCollatorForSeq2Seq,
    "cv_cls": None,
}

MAIN_METRIC = {
    "cls": "test_accuracy",
    "ner": "test_overall_f1",
    "abs-sum": "test_rougeL",
    "cv_cls": "test_accuracy",
}


class TrainingMetricsLogger(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        # Get train loss
        log.info(f"Epoch {metrics['epoch']}\nEvaluation Metrics:")
        log.info(
            f"Train Loss: {state.log_history[-2]['loss']}"
        )  # since [-1] is the dict with evaluation metrics
        [
            log.info(f"{k.replace('_', ' ').title()}: {v}")
            for k, v in metrics.items()
            if k != "epoch"
        ]


class ModalTransformersWrapper:  # TODO: add interface
    def __init__(
        self,
        model,
        tokenizer,
        model_config,
        num_labels: int,
        task: str = "cls",  # 'cls' or 'ner'
        id2label: dict = None,
        default_data_config=None,
        name: str = "acquisition",
        dev_data=None,
        shuffle_dev: bool = False,
        dev_size: float = 0.0,
        seed: int = 42,
        trainer_kwargs=None,
        batch_size_kwargs=None,
        optimizer_kwargs=None,
        scheduler_kwargs=None,
        time_dict_path: Path or str = None,
        cache_dir: Path or str = None,
        cache_model: bool = False,
        num_checkpoints_to_save: int = 1,
        tokenize_dev_data: int = True,
        embeddings=None,
        word2idx=None,
        *args,
        **kwargs,
    ):
        # super().__init__(*args, **kwargs)
        self.model = model
        self.tokenizer = tokenizer
        self.model_config = model_config

        self.task = task
        self.num_labels = num_labels
        self.id2label = id2label
        self.name = name

        if default_data_config is None:
            default_data_config = {
                "dataset_name": None,
                "text_name": "text",
                "label_name": "label",
            }
        self.data_config = dict(default_data_config)
        # TODO: make as an argument
        self.data_config["display_tag_stats"] = "f1"

        self._optimizer_kwargs = optimizer_kwargs
        self._trainer_kwargs = trainer_kwargs
        self._scheduler_kwargs = scheduler_kwargs
        self._batch_size_kwargs = batch_size_kwargs

        self.dev_data = dev_data
        self._dev_kwargs = DictConfig(
            {
                "size": dev_size,
                "shuffle": shuffle_dev,
            }
        )
        if self.dev_data is not None:
            if tokenize_dev_data:
                self.tokenized_dev_data = self.tokenize_data(
                    tokenizer,
                    dev_data,
                    self.task,
                    self.data_config["text_name"],
                    self.data_config["label_name"],
                )
            else:
                self.tokenized_dev_data = None

        self.seed = seed

        if self._trainer_kwargs["serialization_dir"] is None:
            self._trainer_kwargs["serialization_dir"] = "./output/"
        self._num_checkpoints_to_save = num_checkpoints_to_save
        self.use_own_trainer = (
            model_config.training.get("pseudo_labeled_label_smoothing", False) != False
            or model_config.training.get("labeled_weight", 1.0) != 1.0
        )

        self.cache_dir = (
            Path(cache_dir) if cache_dir is not None else Path(f"workdir/cache_{seed}")
        )
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_model = cache_model
        if time_dict_path is None:
            time_dict_path = self.cache_dir / f"time_dict_{name}.json"
            create_time_dict(time_dict_path, name)
        self.time_dict_path = time_dict_path
        self.embeddings = embeddings
        self.word2idx = word2idx

    def fit(self, train_data, from_scratch=True, is_tokenized=False, data_config=None):

        # TODO: find a better way
        if self._dev_kwargs["size"] > 0:
            splitted = train_data.train_test_split(
                train_size=self._dev_kwargs[
                    "size"
                ],  # this is done to have the sample in reversed way
                shuffle=self._dev_kwargs["shuffle"],
                seed=self.seed,
            )
            # this is done to have the sample in reversed way
            train_data = splitted["test"]
            dev_data = splitted["train"]

            if not is_tokenized:
                data_config = (
                    data_config if data_config is not None else self.data_config
                )
                dev_data = self.tokenize_data(
                    tokenizer=self.tokenizer,
                    data=dev_data,
                    task=self.task,
                    text_name=data_config["text_name"],
                    label_name=data_config["label_name"],
                )
        else:
            dev_data = self.tokenized_dev_data
            if dev_data is None:
                data_config = (
                    data_config if data_config is not None else self.data_config
                )
                dev_data = self.tokenize_data(
                    tokenizer=self.tokenizer,
                    data=dev_data,
                    task=self.task,
                    text_name=data_config["text_name"],
                    label_name=data_config["label_name"],
                )
                self.tokenized_dev_data = dev_data

        log.info(f"Training dataset size: {len(train_data)}")
        log.info(
            f"Validation dataset size: {len(dev_data) if dev_data is not None else 0}"
        )

        (
            batch_size,
            num_epochs,
            steps_per_epoch,
            scheduler_warmup_steps,
        ) = get_train_constants(
            len(train_data),
            self._trainer_kwargs.num_epochs,
            self._batch_size_kwargs.batch_size,
            self._batch_size_kwargs.adjust_batch_size,
            self._batch_size_kwargs.adjust_num_epochs,
            self._batch_size_kwargs.min_num_gradient_steps,
            self._scheduler_kwargs.warmup_steps_factor,
            self._batch_size_kwargs.min_batch_size,
            self._dev_kwargs["size"],
        )

        if from_scratch:
            cache_dir = self.cache_dir if self.cache_model else None
            self.model, self.tokenizer = create_transformers_model_tokenizer(
                self.model_config,
                self.id2label,
                self.seed,
                cache_dir,
                embeddings=self.embeddings,
                word2idx=self.word2idx,
            )

        data_config = data_config if data_config is not None else self.data_config
        if not is_tokenized:
            train_data = self.tokenize_data(
                tokenizer=self.tokenizer,
                data=train_data,
                task=self.task,
                text_name=data_config["text_name"],
                label_name=data_config["label_name"],
                is_train=True,
            )

        data_collator = DATA_COLLATOR_CLASSES[self.task]
        if data_collator is not None:
            data_collator_kwargs = dict(tokenizer=self.tokenizer, padding="longest")
            if (
                self.task == "abs-sum"
                and self._trainer_kwargs.get("label_smoothing_factor", None) is not None
            ):
                data_collator_kwargs["model"] = self.model
            data_collator = data_collator(**data_collator_kwargs)

        # Only if validation sample is "dynamic"
        load_best = self._dev_kwargs.size > 0 or self._trainer_kwargs.get(
            "load_best_at_end", False
        )
        logging_and_evaluation_strategy = (
            self._trainer_kwargs.evaluation_strategy
            if (not load_best) or (self._trainer_kwargs.evaluation_strategy != "no")
            else "epoch"
        )
        gradient_accumulation_steps = (
            self._trainer_kwargs.accumulation.gradient_accumulation_steps
        )
        save_strategy = (
            "no"
            if (not load_best and self.name != "target")
            else logging_and_evaluation_strategy
        )
        save_total_limit = self._num_checkpoints_to_save
        validation_metric = "eval_" + self._trainer_kwargs.validation_metric
        other_kwargs = {}
        if self._trainer_kwargs.accumulation.eval_accumulation_steps is not None:
            other_kwargs[
                "eval_accumulation_steps"
            ] = self._trainer_kwargs.accumulation.eval_accumulation_steps
        if logging_and_evaluation_strategy == "steps":
            other_kwargs["logging_steps"] = other_kwargs[
                "save_steps"
            ] = self._trainer_kwargs.get("logging_steps", 100)

            other_kwargs["max_steps"] = self._trainer_kwargs.get(
                "max_steps", self._batch_size_kwargs.min_num_gradient_steps
            )
        log.info(f"Load best at end: {load_best}")

        if self.task in ["cls", "ner", "cv_cls"]:
            training_args_class = TrainingArguments
        elif self.task == "abs-sum":
            training_args_class = Seq2SeqTrainingArguments

            generation_max_length = (
                self._get_generation_max_length(train_data)
                if self._trainer_kwargs.generation_max_length is None
                else self._trainer_kwargs.generation_max_length
            )
            log.info(f"Using generation max length == {generation_max_length}")
            generation_num_beams = self._trainer_kwargs.generation_num_beams
            other_kwargs["predict_with_generate"] = True
            other_kwargs["generation_max_length"] = generation_max_length
            other_kwargs["generation_num_beams"] = generation_num_beams
        else:
            raise NotImplementedError

        training_args = training_args_class(
            output_dir=self._trainer_kwargs["serialization_dir"]
            or "output",  # output directory
            # Batch size args
            num_train_epochs=num_epochs,  # total number of training epochs
            per_device_train_batch_size=batch_size,  # batch size per device during training
            per_device_eval_batch_size=self._batch_size_kwargs.eval_batch_size,  # batch size for evaluation
            # Optimizer args
            adafactor=self._scheduler_kwargs.use_adafactor,
            learning_rate=self._optimizer_kwargs.lr,
            weight_decay=self._optimizer_kwargs.weight_decay,  # strength of weight decay
            # Gradient args
            max_grad_norm=self._trainer_kwargs.grad_clipping,
            label_smoothing_factor=self._trainer_kwargs.get(
                "label_smoothing_factor", 0.0
            ),
            # Scheduler args
            warmup_ratio=self._scheduler_kwargs.warmup_steps_factor,
            # fp16 args
            fp16=self._trainer_kwargs.fp16.training,
            fp16_full_eval=self._trainer_kwargs.fp16.evaluation,
            # Accumulation args
            gradient_accumulation_steps=gradient_accumulation_steps,
            # Evaluation args
            metric_for_best_model=validation_metric,
            load_best_model_at_end=load_best,
            evaluation_strategy=logging_and_evaluation_strategy,  # Evaluation is done at the end of each epoch
            logging_strategy=logging_and_evaluation_strategy,
            save_strategy=save_strategy,
            save_total_limit=save_total_limit,  # limit the total amount of checkpoints. Deletes the older checkpoints
            # Disable any integrations
            report_to="none",
            # Other args
            seed=self.seed,
            **other_kwargs,
        )
        log.info(training_args)

        callbacks = [TrainingMetricsLogger()]
        if load_best:
            callbacks.append(EarlyStoppingCallback(self._trainer_kwargs.patience))

        compute_metrics = self._get_compute_metrics_fn(
            self.task,
            self._trainer_kwargs.eval_metrics,
            self.num_labels,
            self.data_config["dataset_name"],
            self.id2label,
            self.data_config["display_tag_stats"],
            tokenizer=self.tokenizer,
            metrics_cache_dir=self.cache_dir / "metrics",
        )

        set_seed(self.seed)
        trainer_class = (
            TrainerForPseudoLabeled
            if self.use_own_trainer
            else Trainer
            if self.task in ["cls", "ner", "cv_cls"]
            else Seq2SeqTrainer
            if self.task == "abs-sum"
            else None
        )
        # https://discuss.huggingface.co/t/trainer-vs-seq2seqtrainer/3145
        # seq2seq trainer has some advantages over vanilla trainer
        # e.g., it allows to compute generative metrics (such as ROUGE)
        # during evaluation loop
        trainer = trainer_class(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_data,
            eval_dataset=dev_data,
            callbacks=callbacks,
            compute_metrics=compute_metrics,
        )
        # Prevent parallelization of small models
        if (self.name == "acquisition" or self.name == "successor") and len(
            os.environ.get("CUDA_VISIBLE_DEVICES").split(",")
        ) > 1:
            trainer.args._n_gpu = 1
        self.trainer = trainer

        log.info(f"Starting training...")
        start_time = time.time()

        trainer.train()

        self._calculate_time(start_time, phase="fit")

        mlflow.end_run()
        self.trainer.model.eval()
        self.best_metric = getattr(trainer.state, "best_metric", 1.0)

    def get_predictions(
        self,
        data,
        is_tokenized=False,
        data_config=None,
        calculate_time=False,
        remove_padding=False,
        use_predict_loop: bool = False,
        calculate_loss: bool = False,
        **predict_loop_kwargs,
    ):
        """
        :param remove_padding: ignored if self.task != "ner"
        """

        if data_config is None:
            data_config = self.data_config

        text_name = data_config["text_name"]
        label_name = data_config["label_name"]
        dataset_name = data_config["dataset_name"]

        save_first_bpe_mask = remove_padding and self.task == "ner"
        if not is_tokenized:
            data = self.tokenize_data(
                tokenizer=self.tokenizer,
                data=data,
                task=self.task,
                text_name=text_name,
                label_name=label_name,
                save_first_bpe_mask=save_first_bpe_mask,
            )

        if save_first_bpe_mask:
            self._idx_tmp_first_bpe = data["idx_first_bpe"]
            data = data.remove_columns("idx_first_bpe")

        start_time = time.time()

        if getattr(self, "trainer", None) is not None and not use_predict_loop:
            predictions = self.trainer.predict(data)
        else:
            result = self._model_predict_loop(data, **predict_loop_kwargs)
            loss, logits, extra_data = (
                result["loss"],
                result["logits"],
                result["extra_data"],
            )

            if calculate_loss and loss is not None:
                compute_metrics_fn = self._get_compute_metrics_fn(
                    self.task,
                    eval_metrics=None,
                    num_labels=self.num_labels,
                    dataset_name=dataset_name,
                    metrics_cache_dir=self.cache_dir / "metrics",
                    tokenizer=self.tokenizer,
                )
                labels = data["labels"]
                metrics_dict = compute_metrics_fn([logits, labels])
                metrics_dict["loss"] = loss
                metrics_dict = {k: v for k, v in metrics_dict.items()}

                predictions = DictWithGetattr(
                    {
                        "predictions": logits,
                        "metrics": metrics_dict,
                        "extra_data": extra_data,
                    }
                )
            else:
                predictions = DictWithGetattr(
                    {"predictions": logits, "extra_data": extra_data}
                )

        if calculate_time:
            self._calculate_time(start_time, phase="predict")
        torch.cuda.empty_cache()
        gc.collect()

        return predictions

    def predict_logits(self, data, **kwargs):
        predictions = self.get_predictions(data, **kwargs)
        return predictions.predictions

    def predict_proba(
        self,
        data,
        is_tokenized: bool = False,
        data_config=None,
        to_numpy: bool = True,
        remove_padding: bool = False,
        use_predict_loop: bool = False,
        calculate_loss: bool = False,
        **predict_loop_kwargs,
    ):
        logits = self.predict_logits(
            data,
            is_tokenized=is_tokenized,
            data_config=data_config,
            remove_padding=remove_padding,
            use_predict_loop=use_predict_loop,
            calculate_loss=calculate_loss,
            **predict_loop_kwargs,
        )
        probas = softmax(torch.Tensor(logits).to(self.model.device), dim=-1)

        if self.task == "ner" and remove_padding:
            probas = self._remove_padding(probas, self._idx_tmp_first_bpe)
        # If padding is removed, transformation to numpy has already been done
        elif to_numpy:
            return tensor_to_numpy(probas)
        return probas

    # TODO: add an option to remove padding automatically
    def generate(
        self,
        data,
        return_scores=True,
        return_decoded_preds=True,
        is_tokenized=False,
        data_config=None,
        to_numpy: bool = False,
        to_eval_mode: bool = True,
        batch_size: int = None,
        calculate_time=True,
        generation_max_length: int = None,
        **kwargs,
    ):
        """
        Only implemented for seq2seq models (up to 01.0 only for Abs-sum)
        :param data:
        :param is_tokenized:
        :param data_config:
        :param remove_padding:
        :param to_numpy:
        :param calculate_time:
        :return: sequences of ids of most probable tokens & scores of each sequence (sum of log probs)
        """
        if data_config is None:
            data_config = self.data_config

        if not is_tokenized:
            data = self.tokenize_data(
                tokenizer=self.tokenizer,
                data=data,
                task=self.task,
                text_name=data_config["text_name"],
                label_name=data_config["label_name"],
            )
            if "labels" in data[0]:
                data = data.remove_columns(["labels"])

        start_time = time.time()
        if getattr(self, "trainer", None) is not None:
            log.warning("Model is not fine-tuned! Be careful applying it to new data.")
        output = self._model_generate_loop(
            data, batch_size, to_eval_mode, generation_max_length, **kwargs
        )

        if calculate_time:
            self._calculate_time(start_time, phase="predict")

        if to_numpy:
            output = {k: tensor_to_numpy(v) for k, v in output.items()}

        if return_decoded_preds:
            output["predictions"] = self.tokenizer.batch_decode(
                output["sequences"], skip_special_tokens=True
            )
            if kwargs.get("num_return_sequences", 1) > 1:
                num_ret_seq = kwargs.get("num_return_sequences", 1)
                preds = output["predictions"]
                output["predictions"] = [
                    preds[i * num_ret_seq : (i + 1) * num_ret_seq]
                    for i in range(len(preds) // num_ret_seq)
                ]
                output["sequences"] = output["sequences"].reshape(
                    -1, num_ret_seq, output["sequences"].shape[1]
                )
                output["sequences_scores"] = output["sequences_scores"].reshape(
                    -1, num_ret_seq
                )

        if not return_scores:
            output.pop("sequences_scores")
        torch.cuda.empty_cache()
        gc.collect()

        return output

    def evaluate(self, data, is_tokenized=False, data_config=None, *args, **kwargs):

        predictions = self.get_predictions(
            data, is_tokenized, data_config, calculate_time=False
        )
        return predictions.metrics

    @staticmethod
    def tokenize_data(
        tokenizer,
        data,
        task="cls",
        text_name="text",
        label_name="labels",
        save_first_bpe_mask=False,
        is_train=False,
        **kwargs,
    ):
        # TODO: either remove or expand this method in TransformersDataset for all the tasks
        # if isinstance(data, TransformersDataset):
        #     return data.tokenize_data(
        #         tokenizer,
        #         text_name,
        #         label_name,
        #         save_first_bpe_mask=save_first_bpe_mask,
        #         **kwargs,
        #     )
        if task == "cls":
            tokenize_function = ModalTransformersWrapper._get_tokenize_fn_for_cls(
                tokenizer, text_name, label_name, **kwargs
            )
        elif task == "ner":
            # `labels` need to correspond to "padded" tags, therefore we'll get the error in `tokenize_function`
            if label_name == "labels":
                label_name = "tags"
                data = data.rename_column("labels", "tags")
            tokenize_function = ModalTransformersWrapper._get_tokenize_fn_for_ner(
                tokenizer, text_name, label_name, save_first_bpe_mask, **kwargs
            )
        elif task == "abs-sum":
            tokenize_function = ModalTransformersWrapper._get_tokenize_fn_for_abs_sum(
                tokenizer, text_name, label_name, **kwargs
            )
        elif task == "cv_cls":
            if is_train:
                return random_fixed_length_data_sampler(data)
            else:
                return data
        else:
            raise NotImplementedError

        columns_to_remove = [
            x for x in data.features.keys() if x not in ["labels", "weight"]
        ]
        batched = task == "cls" or task == "ner" or task == "abs-sum"
        # The last two arguments is a temporary fix for Kristofari
        return data.map(
            tokenize_function,
            batched=batched,
            remove_columns=columns_to_remove,
            load_from_cache_file=False,
            cache_file_name=f"tmp.data",
        )

    @staticmethod
    def _get_tokenize_fn_for_cls(
        tokenizer, text_name="text", label_name="label", **kwargs
    ):
        if label_name == "labels":

            def tokenize_function(instances):
                encoding = tokenizer(instances[text_name], truncation=True, **kwargs)
                return encoding

        else:

            def tokenize_function(instances):
                encoding = tokenizer(instances[text_name], truncation=True, **kwargs)
                encoding["labels"] = instances[label_name]
                return encoding

        return tokenize_function

    @staticmethod
    def _get_tokenize_fn_for_ner(
        tokenizer,
        tokens_name="tokens",
        tags_name="ner_tags",
        save_first_bpe_mask=False,
        **kwargs,
    ):
        def tokenize_function(instances):
            tokenized_inputs = tokenizer(
                instances[tokens_name],
                truncation=True,
                is_split_into_words=True,
                **kwargs,
            )
            all_labels = instances[tags_name]
            new_labels, idx_first_bpe = [], []
            for i, labels in enumerate(all_labels):
                word_ids = tokenized_inputs.word_ids(i)
                new_labels.append(align_labels_with_tokens(labels, word_ids))
                # Save ids of the first BPEs if necessary
                if save_first_bpe_mask:
                    idx_first_bpe.append(
                        [
                            i
                            for (i, x) in enumerate(word_ids[1:], 1)
                            if ((x != word_ids[i - 1]) and (x is not None))
                        ]
                    )
            tokenized_inputs["labels"] = new_labels
            # Save ids of the first BPEs if necessary
            if save_first_bpe_mask:
                tokenized_inputs["idx_first_bpe"] = idx_first_bpe
            return tokenized_inputs

        return tokenize_function

    @staticmethod
    def _get_compute_metrics_fn(
        task,
        eval_metrics=None,
        num_labels=2,
        dataset_name=None,
        id2label=None,
        display_tag_stats=None,
        metrics_cache_dir=None,
        tokenizer=None,
    ):

        if task == "cls" or task == "cv_cls":
            load_func = lambda task: load_metric(
                "accuracy", cache_dir=metrics_cache_dir
            )
        elif task == "ner":
            load_func = lambda task: load_metric("seqeval", cache_dir=metrics_cache_dir)
        elif task == "abs-sum":
            load_func = lambda task: load_metric("rouge", cache_dir=metrics_cache_dir)
        else:
            raise NotImplementedError

        if dataset_name is not None:
            # since metrics are not implemented for all the datasets
            try:
                metric = (
                    load_metric(dataset_name, cache_dir=metrics_cache_dir)
                    if isinstance(dataset_name, str)
                    else load_metric(*dataset_name, cache_dir=metrics_cache_dir)
                )
            except:
                metric = load_func(task)
        else:
            metric = load_func(task)

        if eval_metrics is not None:
            additional_metrics = [
                load_metric(add_metric, cache_dir=metrics_cache_dir)
                for add_metric in eval_metrics
            ]
            if (
                all((metric.name != "accuracy" for metric in additional_metrics))
                and task != "abs-sum"
            ):
                additional_metrics.append(
                    load_metric("accuracy", cache_dir=metrics_cache_dir)
                )
        elif task == "cls":
            additional_metrics = [load_metric("f1", cache_dir=metrics_cache_dir)]
            if metric.name != "accuracy":
                additional_metrics.append(
                    load_metric("accuracy", cache_dir=metrics_cache_dir)
                )
        elif task == "ner":
            additional_metrics = (
                [load_metric("seqeval", cache_dir=metrics_cache_dir)]
                if metric.name != "seqeval"
                else []
            )
        else:
            raise NotImplementedError

        if task == "cls" or task == "cv_cls":
            return ModalTransformersWrapper._get_compute_metrics_fn_for_cls(
                metric, additional_metrics, num_labels
            )
        elif task == "ner":
            return ModalTransformersWrapper._get_compute_metrics_fn_for_ner(
                metric, id2label, additional_metrics, num_labels, display_tag_stats
            )

    @staticmethod
    def _get_compute_metrics_fn_for_cls(
        metric, additional_metrics: Union[tuple, list] = tuple(), num_labels=2
    ):
        def compute_metrics(eval_preds):
            logits, labels, *inputs = eval_preds
            preds = logits.argmax(axis=-1)
            metrics_dict = metric.compute(predictions=preds, references=labels)
            for add_metric in additional_metrics:
                if add_metric.name == "f1" and num_labels > 2:
                    for average in ["micro", "macro", "weighted"]:
                        add_metric_dict = add_metric.compute(
                            predictions=preds, references=labels, average=average
                        )
                        metrics_dict.update({f"f1_{average}": add_metric_dict["f1"]})
                else:
                    add_metric_dict = add_metric.compute(
                        predictions=preds, references=labels
                    )
                    metrics_dict.update(add_metric_dict)

            return metrics_dict

        return compute_metrics

    @staticmethod
    def _get_compute_metrics_fn_for_ner(
        metric,
        id2label,
        additional_metrics: Union[Tuple, List] = tuple(),
        num_labels=2,
        display_tag_stats="f1",
        padding_idx=-100,
    ):
        def compute_metrics(eval_preds):
            logits_ids, labels_ids, *inputs = eval_preds
            preds_ids = logits_ids.argmax(axis=-1)
            # Convert ids to tags
            idx_first_bpe = np.array(labels_ids) != padding_idx
            # Remove padding
            labels_without_padding = ModalTransformersWrapper._remove_padding(
                labels_ids, idx_first_bpe
            )
            preds_without_padding = ModalTransformersWrapper._remove_padding(
                preds_ids, idx_first_bpe
            )
            # Convert idx to tokens
            labels = ModalTransformersWrapper._convert_id2label(
                labels_without_padding, id2label
            )
            preds = ModalTransformersWrapper._convert_id2label(
                preds_without_padding, id2label
            )
            # Compute metrics
            metrics_dict = metric.compute(predictions=preds, references=labels)

            for add_metric in additional_metrics:
                if add_metric.name == "f1" and num_labels > 2:
                    for average in ["micro", "macro", "weighted"]:
                        add_metric_dict = add_metric.compute(
                            predictions=preds, references=labels, average=average
                        )
                        metrics_dict.update({f"f1_{average}": add_metric_dict["f1"]})
                else:
                    add_metric_dict = add_metric.compute(
                        predictions=preds, references=labels
                    )
                    metrics_dict.update(add_metric_dict)
            # Individual entity scores are presented as dicts now. Need to "straighten" them
            output_dict = ModalTransformersWrapper._construct_metrics_dict_for_ner(
                metrics_dict, display_tag_stats
            )
            return output_dict

        return compute_metrics

    def _model_predict_loop(
        self,
        data,
        evaluate=True,
        to_eval_mode: bool = True,
        eval_batch_size: Union[int, None] = None,
        extra_keys: Union[List[str], Tuple[str], None] = None,
    ):
        """
        not implemented for abstractive summarization
        Args:
            data:
            evaluate: if True, assume labels are given in the data. Otherwise, loss is not aclculated

        Returns: dict with loss and logits
        """
        if eval_batch_size is None:
            eval_batch_size = self._batch_size_kwargs.eval_batch_size
        data_collator = DATA_COLLATOR_CLASSES[self.task](
            tokenizer=self.tokenizer, padding="longest"
        )
        dataloader = DataLoader(
            data,
            batch_size=eval_batch_size,
            collate_fn=data_collator,
        )

        loss = 0
        logits = []
        extra_data = defaultdict(list)
        start = 0
        if extra_keys is not None and "encoder_last_hidden_state" in extra_keys:
            extra_data["encoder_last_hidden_state"] = torch.zeros(
                len(data),
                list(self.model.parameters())[0].shape[1],
                dtype=torch.float32,
                device="cpu",
            )

        device = self.model.device
        if to_eval_mode:
            self.model.eval()

        with torch.no_grad():
            for batch in tqdm(dataloader):
                batch = {k: v.to(device) for k, v in batch.items()}
                predictions = self.model(**batch)
                if evaluate:
                    loss += predictions.loss.item() * len(predictions.logits)
                logits.append(predictions.logits.cpu())
                if extra_keys is not None:
                    for key in extra_keys:
                        key_data = getattr(predictions, key)
                        if isinstance(key_data, torch.Tensor):
                            key_data = key_data.cpu()
                        if key == "encoder_last_hidden_state":
                            end = start + len(batch["input_ids"])
                            key_data = torch.stack(
                                [
                                    hid_st[:num_tokens, :].mean(dim=0)
                                    for hid_st, num_tokens in zip(
                                        key_data,
                                        batch["attention_mask"]
                                        .sum(dim=1)
                                        .cpu()
                                        .numpy(),
                                    )
                                ]
                            )
                            extra_data[key][start:end, :].copy_(
                                key_data, non_blocking=True
                            )
                            start = end
                        else:
                            extra_data[key].append(key_data)

        # TODO: make this `if` neater
        if self.task != "abs-sum":
            logits = torch.cat(logits, dim=0)
        else:
            logits = extra_data.get("encoder_last_hidden_state", None)
        if evaluate:
            loss /= len(data)
        else:
            loss = None
        if extra_keys is not None:
            for key in extra_keys:
                if key != "encoder_last_hidden_state":
                    extra_data[key] = torch.cat(extra_data[key], dim=0)

        return DictWithGetattr(
            {"loss": loss, "logits": logits, "extra_data": extra_data}
        )

    def _model_generate_loop(
        self,
        data,
        batch_size=None,
        to_eval_mode=True,
        generation_max_length: int = None,
        **kwargs,
    ):
        """
        Implemented for abstractive summarization
        Args:
            data:

        Returns: dict with sequences of ids and their scores

        """
        torch.cuda.empty_cache()
        if batch_size is None:
            batch_size = (
                self._batch_size_kwargs.get("pseudo_labeling_eval_batch_size", None)
                or self._batch_size_kwargs.eval_batch_size
            )

        dataloader = DataLoader(
            data,
            batch_size=batch_size,
            collate_fn=DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer, padding="longest"
            ),
        )

        if generation_max_length is not None:
            max_length = sequences_length = generation_max_length
        elif getattr(self, "trainer", None) is not None:
            max_length = sequences_length = self.trainer.args.generation_max_length
        else:
            max_length = None
            sequences_length = 128
        num_return_sequences = kwargs.get("num_return_sequences", 1)

        if to_eval_mode:
            self.model.eval()
        device = self.model.device
        sequences_X_shape = num_return_sequences * len(data)
        sequences = (
            torch.zeros(
                (sequences_X_shape, sequences_length), dtype=torch.int64, device=device
            )
            + self.tokenizer.pad_token_id
        )
        scores = torch.empty(sequences_X_shape, dtype=torch.float32, device=device)

        with torch.no_grad():
            start = 0
            for batch in tqdm(dataloader):
                batch = {k: v.to(device) for k, v in batch.items()}
                output = self.model.generate(
                    **batch,
                    max_length=max_length,
                    min_length=3,  # To avoid empty summaries. 3 == <BOS> + at least one token + <EOS>
                    output_scores=True,
                    return_dict_in_generate=True,
                    **kwargs,
                )
                end = start + (len(batch["input_ids"]) * num_return_sequences)
                sequences[start:end, : output.sequences.shape[1]].copy_(
                    output.sequences, non_blocking=True
                )
                scores[start:end].copy_(output.sequences_scores, non_blocking=True)
                start = end

        return {"sequences": sequences, "sequences_scores": scores}

    @staticmethod
    def _convert_id2label(array_idx, id2label):
        array_tag = []
        for instance_idx in array_idx:
            instance_tag = []
            for idx in instance_idx:
                instance_tag.append(id2label[idx])
            array_tag.append(instance_tag)
        return array_tag

    @staticmethod
    def _construct_metrics_dict_for_ner(
        metrics_dict, display_tag_stats: str or bool = "f1"
    ):
        output_dict = {}
        for key, val in metrics_dict.items():
            if isinstance(val, dict):
                if display_tag_stats is False or display_tag_stats is None:
                    continue
                elif display_tag_stats == "f1":
                    output_dict[key + "_f1"] = val["f1"]
                else:
                    for key_2, val_2 in val.items():
                        output_dict[key + "_" + key_2] = val_2
            else:
                output_dict[key] = val

        return output_dict

    @staticmethod
    def _remove_padding(predictions, idx_first_bpe):
        preds = []
        for pred, cond in zip(predictions, idx_first_bpe):
            if isinstance(pred, torch.Tensor):
                pred = pred.cpu().detach().numpy()
            pred_without_padding = pred[cond]
            preds.append(pred_without_padding)

        return np.array(preds)

    def enable_dropout(self):
        """Function to enable the dropout layers during test-time"""
        dropout_found = False
        for m in self.model.modules():
            if m.__class__.__name__.startswith("Dropout"):
                dropout_found = True
                m.train()

        if not dropout_found:
            self.model.train()  # e.g., for BART

    def disable_dropout(self):
        """Function to disable the dropout layers during test-time"""
        for m in self.model.modules():
            if m.__class__.__name__.startswith("Dropout"):
                m.eval()

    def _get_generation_max_length(self, data=None, data_config=None):
        if data is None:
            data = self.tokenized_dev_data
            if data is None:
                data_config = (
                    data_config if data_config is not None else self.data_config
                )
                data = self.tokenize_data(
                    tokenizer=self.tokenizer,
                    data=data,
                    task=self.task,
                    text_name=data_config["text_name"],
                    label_name=data_config["label_name"],
                )
                self.tokenized_dev_data = data
        gen_max_length = min(
            # Using quantile instead of `max` since `max` can lead to enormously long generation
            round(np.quantile([len(x) for x in data["labels"]], 0.95) + 1),
            self.tokenizer.model_max_length,
        )
        return gen_max_length

    def _calculate_time(self, start_time, phase="fit"):
        try:
            time_work = time.time() - start_time
            time_dict = json_load(self.time_dict_path)
            time_dict[self.name + f"_{phase}"].append(time_work)
            json_dump(time_dict, self.time_dict_path)
            log.info(f"Done with the model {phase}.")
        except:
            pass
