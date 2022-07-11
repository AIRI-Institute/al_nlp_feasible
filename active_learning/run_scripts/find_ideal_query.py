import os

if os.environ.get("OFFLINE_MODE", "false") == "true":
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

import transformers as ts
from datasets import load_metric, load_dataset, concatenate_datasets
import numpy as np
from tqdm.notebook import tqdm
import torch
from copy import deepcopy
from sklearn.model_selection import train_test_split
import nltk
import gc
import logging
import sys

GPU_ID = os.environ.get("CUDA_VISIBLE_DEVICES", -1)

# Set logger
if not os.path.exists("workdir"):
    os.mkdir("workdir")
fileh = logging.FileHandler(f"workdir/log_{GPU_ID}.txt", "a")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
fileh.setFormatter(formatter)

log = logging.getLogger()  # root logger
for hndlr in log.handlers[:]:  # remove all old handlers
    log.removeHandler(hndlr)
log.addHandler(fileh)
sys.stdout.write = log.info


INIT_TRAIN_SIZE = "auto"
MODEL_CHECKPOINT = os.environ.get("MODEL_CHECKPOINT", "roberta-base")
NUM_QUERIES = int(os.environ.get("NUM_QUERIES", 100))
SEED = int(os.environ.get("SEED", 42))
VALID_SUBSAMPLE_SIZE = int(os.environ.get("VALID_SUBSAMPLE_SIZE", 500))
QUERY_SUBSAMPLE_SIZE = int(os.environ.get("QUERY_SUBSAMPLE_SIZE", 1000))
TRAIN_BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE", 16))
DATASET = os.environ.get("DATASET_NAME", "ag_news")
TEXT_COLUMN_NAME = os.environ.get("TEXT_COLUMN_NAME", "text")
LABEL_COLUMN_NAME = os.environ.get("LABEL_COLUMN_NAME", "label")

if __name__ == "__main__":

    try:
        if " " in DATASET:
            data = load_dataset(*DATASET.split(), cache_dir="cache/data")
        else:
            data = load_dataset(DATASET, cache_dir="cache/data")
        tokenizer = ts.AutoTokenizer.from_pretrained(
            MODEL_CHECKPOINT, cache_dir="cache/tokenizer"
        )
        model = ts.AutoModelForSequenceClassification.from_pretrained(
            MODEL_CHECKPOINT,
            cache_dir="cache/model",
            num_labels=len(set(data["train"][LABEL_COLUMN_NAME])),
        )

        def tokenize_function(instances):
            encoding = tokenizer(instances[TEXT_COLUMN_NAME], truncation=True)
            if LABEL_COLUMN_NAME != "labels":
                encoding["labels"] = instances[LABEL_COLUMN_NAME]
            return encoding

        valid_data = data["validation"] if "validation" in data else data["test"]
        if VALID_SUBSAMPLE_SIZE == -1:
            valid_subsample = valid_data
        else:
            subsample_idx, _ = train_test_split(
                range(len(valid_data)),
                train_size=VALID_SUBSAMPLE_SIZE,
                stratify=valid_data[LABEL_COLUMN_NAME],
                random_state=SEED,
            )
            valid_subsample = valid_data.select(subsample_idx)
        valid_subsample = valid_subsample.map(
            tokenize_function,
            batched=True,
            remove_columns=[
                x for x in valid_data.features.keys() if x not in ["labels", "id"]
            ],
        )

        train_data = data["train"].map(
            tokenize_function,
            batched=True,
            remove_columns=[
                x for x in data["train"].features.keys() if x not in ["labels", "id"]
            ],
        )
        if INIT_TRAIN_SIZE == "auto":
            INIT_TRAIN_SIZE = len(set(data["train"][LABEL_COLUMN_NAME]))
        init_train_idx, _ = train_test_split(
            range(len(train_data)),
            train_size=INIT_TRAIN_SIZE,
            stratify=train_data["labels"],
            random_state=SEED,
        )
        train_sample = train_data.select(init_train_idx)
        unlabeled_data = train_data.select(
            np.setdiff1d(range(len(train_data)), init_train_idx)
        )

        data_collator = ts.DataCollatorWithPadding(
            tokenizer=tokenizer, padding="longest"
        )

        metric = load_metric("accuracy", cache_dir="cache/metric")
        additional_metrics = [load_metric("f1", cache_dir="cache/metric")]

        def compute_metrics(eval_preds):
            logits, labels = eval_preds
            preds = logits.argmax(axis=-1)
            metrics_dict = metric.compute(predictions=preds, references=labels)
            for add_metric in additional_metrics:
                if add_metric.name == "f1":
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

        def model_init():
            return deepcopy(model)

        if VALID_SUBSAMPLE_SIZE > 0:
            eval_batch_size = VALID_SUBSAMPLE_SIZE
        else:
            if DATASET == "ag_news":
                eval_batch_size = 500
            else:
                eval_batch_size = 100
        training_args = ts.TrainingArguments(
            output_dir=f"workdir/{DATASET}_model_output_{SEED}",
            # Steps & Batch size args
            max_steps=200,
            per_device_train_batch_size=TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=eval_batch_size,
            # Optimizer args
            learning_rate=2e-5,
            weight_decay=0.03,
            max_grad_norm=0.3,
            # Scheduler args
            warmup_ratio=0.0,
            # Eval args
            metric_for_best_model="accuracy",
            greater_is_better=True,
            load_best_model_at_end=True,
            evaluation_strategy="steps",
            logging_strategy="steps",
            save_strategy="steps",
            eval_steps=10,
            save_steps=10,
            save_total_limit=1,
            # WANDB args
            report_to="none",  # enable logging to W&B
            # General args
            seed=SEED,
            fp16=True,
            fp16_full_eval=False,
        )
        callbacks = [ts.EarlyStoppingCallback(early_stopping_patience=3)]

        best_metrics = []
        log.info("Starting ideal query search")
        for i_query in tqdm(range(NUM_QUERIES)):
            log.info(f"Query {i_query}")
            np.random.seed(i_query + SEED)
            subsample_idx = np.random.choice(
                range(len(unlabeled_data)), QUERY_SUBSAMPLE_SIZE, False
            )
            metrics = []

            for idx in subsample_idx:
                query = unlabeled_data.select([idx])
                train_data_with_query = concatenate_datasets(
                    [train_sample, query], info=train_data.info
                )
                training_args.eval_steps = training_args.save_steps = (
                    1 + i_query // training_args.per_device_train_batch_size
                ) * 5

                trainer = ts.Trainer(
                    model_init=model_init,
                    args=training_args,
                    train_dataset=train_data_with_query,
                    eval_dataset=valid_subsample,
                    compute_metrics=compute_metrics,
                    callbacks=callbacks,
                    data_collator=data_collator,
                )
                trainer.train()
                metrics.append(trainer.evaluate()["eval_accuracy"])

                del trainer.model
                torch.cuda.empty_cache()
                log.info(
                    f"Instance with text: {tokenizer.decode(query['input_ids'][0])}\n and label: {query['labels'][0]}"
                )
                log.info(f"Accuracy: {metrics[-1]}")
                gc.collect()

            query_true_idx = subsample_idx[np.argmax(metrics)]
            query = unlabeled_data.select([query_true_idx])
            train_sample = concatenate_datasets(
                [train_sample, query], info=train_data.info
            )
            unlabeled_data = unlabeled_data.select(
                np.setdiff1d(range(len(unlabeled_data)), query_true_idx)
            )
            best_metrics.append(np.max(metrics))

            log.info(
                f"Iteration {i_query};\n best instance text: {tokenizer.decode(query['input_ids'][0])};\n "
                f"best instance label: {query['labels'][0]};\n "
                f"Accuracy: {metrics[-1]}"
            )

            torch.save(
                tokenizer.batch_decode(
                    unlabeled_data.select(subsample_idx)["input_ids"],
                    skip_special_tokens=True,
                ),
                f"workdir/{DATASET}_candidate_query_documents_{i_query}",
            )
            torch.save(
                unlabeled_data.select(subsample_idx)["labels"],
                f"workdir/{DATASET}_candidate_query_labels_{i_query}",
            )
            torch.save(metrics, f"workdir/{DATASET}_metrics_{i_query}")

            torch.save(train_sample, f"workdir/{DATASET}_train_sample_{i_query}")
            with open(f"workdir/{DATASET}_best_metrics.yaml", "w") as f:
                for val in best_metrics:
                    f.write(str(val) + "\n")

        torch.save(train_sample, f"workdir/{DATASET}_final_query")
    except Exception as e:
        log.error(e, exc_info=True)
