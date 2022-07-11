from datasets.arrow_dataset import Dataset as ArrowDataset
from torch.utils.data import Dataset
from typing import Union, Dict, List
from copy import deepcopy
import numpy as np
from sklearn.model_selection import train_test_split
import logging
from tqdm.notebook import tqdm
from transformers.tokenization_utils_base import BatchEncoding
from transformers import PreTrainedTokenizerFast
from math import ceil

from .token_classification import align_labels_with_tokens

log = logging.getLogger()


class TransformersDataset(Dataset):
    def __init__(
        self,
        instances: Union[list, ArrowDataset, Dict[str, List[Union[list, str, int]]]],
        text_column_name: str = "text",
        label_column_name: str = "label",
        tokenizer: PreTrainedTokenizerFast or None = None,
        tokenization_kwargs: dict or None = None,
        task: str = "cls",  # or "ner"
        id2label: Dict[int, str] = None,
        label_smoothing: bool = False,  # whether labels are smoothed
    ):
        """
        Class, immitating ArrowDataset from HuggingFace datasets
        :param instances: values of the dataset
        :param text_column_name:
        :param label_column_name:
        :param tokenizer:
        :param tokenization_kwargs:
        :param task:
        :param id2label:
        :param label_smoothing:
        """
        super().__init__()
        if isinstance(instances, (dict, BatchEncoding)):
            self._init_from_dict_with_instances(instances)
        else:
            self._init_from_list_of_instances_or_arrow_dataset(instances)

        self.text_column_name = text_column_name
        self.label_column_name = label_column_name
        # For compatibility with ArrowDataset
        self.features = {k: None for k in self.instances[0]}

        self.tokenizer = tokenizer
        self.tokenization_kwargs = (
            tokenization_kwargs if tokenization_kwargs is not None else {}
        )
        if task == "ner":
            self.tokenization_kwargs["is_split_into_words"] = True
        self.tokenized_data = None

        self.task = task
        self.id2label = id2label
        self.label_smoothing = label_smoothing

    def __len__(self):
        return len(self.instances)

    def __getitem__(
        self, key: Union[int, slice, list, str]
    ) -> Dict[Union[str, int], list]:

        if isinstance(key, str):
            return self.columns_dict[key].tolist()
        elif isinstance(key, int):
            return self._getitem_int(key)
        return self._getitem_list_or_slice(key)

    def select(self, key: Union[int, slice, list]) -> "TransformersDataset":
        if isinstance(key, int):
            instances = [self[key]]
        else:
            instances = self[key]

        return TransformersDataset(
            instances,
            text_column_name=self.text_column_name,
            label_column_name=self.label_column_name,
            tokenizer=self.tokenizer,
            tokenization_kwargs=self.tokenization_kwargs,
            task=self.task,
            id2label=self.id2label,
        )

    def add(
        self,
        instance_or_instances: Union[list, dict, "TransformersDataset"],
        inplace=True,
    ) -> Union[None, "TransformersDataset"]:

        if isinstance(instance_or_instances, dict):
            instance_or_instances = [instance_or_instances]
        elif isinstance(instance_or_instances, TransformersDataset):
            instance_or_instances = instance_or_instances.instances

        columns = self.columns
        # Need to sort columns the same way they are done in `instances` to prevent cases [0, 1] != [1, 0]
        if sorted(columns) == sorted(instance_or_instances[0].keys()):
            columns = list(instance_or_instances[0].keys())
        else:
            raise RuntimeError(
                "Instance columns do not match with the Dataset columns!"
            )
        if inplace:
            self.instances = self.instances + list(instance_or_instances)
            # First convert back to list and then will convert back to np.ndarray to accelerate the process
            for column in columns:
                self.columns_dict[column] = list(self.columns_dict[column])
            for inst in instance_or_instances:
                assert (
                    list(inst.keys()) == columns
                ), "Instance columns do not match with the Dataset columns!"
                for column in columns:
                    self.columns_dict[column].append(inst[column])
            # Convert back to np.ndarray
            for column in columns:
                self.columns_dict[column] = np.array(self.columns_dict[column])
        else:
            dataset = deepcopy(self)
            dataset.instances = dataset.instances + list(instance_or_instances)
            # First convert back to list and then will convert back to np.ndarray to accelerate the process
            for column in columns:
                dataset.columns_dict[column] = list(dataset.columns_dict[column])
            for inst in instance_or_instances:
                assert (
                    list(inst.keys()) == columns
                ), "Instance columns do not match with the Dataset columns!"
                for column in columns:
                    dataset.columns_dict[column].append(inst[column])
            # Convert back to np.ndarray
            for column in columns:
                dataset.columns_dict[column] = np.array(dataset.columns_dict[column])

            return dataset

    def add_column(self, column_name: str, column_data: Union[list, np.ndarray]):
        for i, new_column_data in enumerate(column_data):
            self.instances[i][column_name] = new_column_data
        self.columns.append(column_name)
        self.columns_dict[column_name] = new_column_data

    def _getitem_int(self, idx: int) -> Dict[Union[str, int], list]:
        instance = self.instances[idx]
        if self.tokenizer is None:
            return instance

        encoded = self.tokenizer(
            instance[self.text_column_name], **self.tokenization_kwargs
        )
        encoded.update(instance)
        return encoded

    def _getitem_list_or_slice(
        self, list_or_slice_idx: Union[List[int], slice]
    ) -> Dict[Union[str, int], list]:

        instances_dict = self._get_return_dict_for_list_or_slice(list_or_slice_idx)

        if self.tokenizer is None:
            return instances_dict

        # If there is a tokenizer, tokenize the text data and add it to the return dict
        texts = self.columns_dict[self.text_column_name]
        texts_idx = texts[list_or_slice_idx].tolist()
        encoded = self.tokenizer(texts_idx, **self.tokenization_kwargs)
        encoded.update(instances_dict)
        return encoded

    def _get_return_dict_for_list_or_slice(
        self, list_or_slice_idx: Union[List[int], slice]
    ) -> Dict[Union[str, int], list]:
        return {
            column: self.columns_dict[column][list_or_slice_idx].tolist()
            for column in self.columns
        }

    # def tokenize_data(
    #     self,
    #     tokenizer: PreTrainedTokenizerFast or None = None,
    #     text_column_name: str or None = None,
    #     label_column_name: str or None = None,
    #     save_first_bpe_mask: bool = False,
    #     ignore_tokenized_data: bool = True,
    #     **kwargs,
    # ):
    #     if self.tokenized_data is not None and not ignore_tokenized_data:
    #         return self.tokenized_data
    #
    #     tokenizer = tokenizer if tokenizer is not None else self.tokenizer
    #     tokenization_kwargs = deepcopy(self.tokenization_kwargs)
    #     tokenization_kwargs.update(kwargs)
    #     # Add truncation if it were not specified that is must be disabled
    #     if "truncation" not in tokenization_kwargs:
    #         tokenization_kwargs["truncation"] = True
    #
    #     text_column_name = (
    #         text_column_name if text_column_name is not None else self.text_column_name
    #     )
    #     label_column_name = (
    #         label_column_name
    #         if label_column_name is not None
    #         else self.label_column_name
    #     )
    #
    #     texts = list(self.columns_dict[text_column_name])
    #     labels = list(self.columns_dict[label_column_name])
    #     # TODO: comb this part
    #     if len(self) < 150_000 and not self.task.startswith("cv"):
    #         if self.task == "cls":
    #             batch_encoding = tokenizer(texts, **tokenization_kwargs)
    #         elif self.task == "ner":
    #             for redundant_key in [
    #                 "is_split_into_words",
    #                 "return_offsets_mapping",
    #             ]:
    #                 if redundant_key in tokenization_kwargs:
    #                     del tokenization_kwargs[redundant_key]
    #
    #             batch_encoding = tokenizer(
    #                 texts,
    #                 is_split_into_words=True,
    #                 **tokenization_kwargs,
    #             )
    #             tags, data_idx_first_bpe = [], []
    #             for i in range(len(batch_encoding.encodings)):
    #                 encoding_labels = align_labels_with_tokens(
    #                     labels[i], batch_encoding.word_ids(i)
    #                 )
    #                 tags.append(encoding_labels)
    #                 # Save ids of the first BPEs if necessary
    #                 if save_first_bpe_mask:
    #                     word_ids = batch_encoding.word_ids(i)
    #                     idx_first_bpe = [
    #                         i
    #                         for (i, x) in enumerate(word_ids[1:], 1)
    #                         if ((x != word_ids[i - 1]) and (x is not None))
    #                     ]
    #                     data_idx_first_bpe.append(idx_first_bpe)
    #             # Dummy step to make the next step equal for both tasks
    #             labels = tags
    #         else:
    #             raise NotImplementedError
    #         # Since we get batch encoding instead of list of tokenized instances, need to reformat it
    #         tokenized_data = [
    #             {
    #                 "input_ids": inst.ids,
    #                 "attention_mask": inst.attention_mask,
    #                 "labels": labels[i],
    #             }
    #             for i, inst in enumerate(batch_encoding.encodings)
    #         ]
    #         if save_first_bpe_mask and self.task == "ner":
    #             [
    #                 tokenized_data[i].update({"idx_first_bpe": data_idx_first_bpe[i]})
    #                 for i in range(len(tokenized_data))
    #             ]
    #
    #         self.tokenized_data = tokenized_data
    #     elif self.task.startswith("cv"):
    #         self.tokenized_data = [
    #             {"features": text, "labels": label}
    #             for text, label in zip(texts, labels)
    #         ]
    #     else:
    #         raise NotImplementedError("Not implemented for more than 150k instances")
    #
    #     return TransformersDataset(
    #         tokenized_data,
    #         text_column_name=self.text_column_name,
    #         label_column_name=self.label_column_name,
    #         tokenizer=self.tokenizer,
    #         tokenization_kwargs=self.tokenization_kwargs,
    #         task=self.task,
    #         id2label=self.id2label,
    #     )

    def __repr__(self):
        return f"TransformersDataset with num rows = {len(self)} and columns = {self.columns}"

    def train_test_split(self, train_size=None, test_size=None, shuffle=False, seed=42):

        if train_size is None and test_size is None:
            test_size = 0.2
        elif train_size is not None and test_size is None:
            test_size = 1 - train_size
        elif train_size is not None and test_size is not None:
            log.warning(
                "Both `train_size` and `test_size` are provided. Ignoring `train_size`."
            )

        ids = np.arange(len(self))
        train_ids, test_ids = train_test_split(
            ids, test_size=test_size, shuffle=shuffle, random_state=seed
        )

        train_data = self.select(train_ids)
        test_data = self.select(test_ids)

        return {"train": train_data, "test": test_data}

    def _init_from_list_of_instances_or_arrow_dataset(
        self,
        instances: Union[list, ArrowDataset],
    ):
        columns = instances[0].keys()
        self.columns = list(columns)
        # Check all instances have the same columns
        assert all(
            [inst.keys() == columns for inst in instances]
        ), "All the instances must have the same keys!"

        self.instances = list(instances)
        self.columns_dict = {}
        for column in self.columns:
            self.columns_dict[column] = np.array([inst[column] for inst in instances])

    def _init_from_dict_with_instances(
        self,
        instances: Dict[str, List[Union[list, str, int]]],
    ):
        columns = instances.keys()
        self.columns = list(columns)
        # Check all columns have the same length
        first_column_length = len(instances[self.columns[0]])
        assert all(
            [
                len(instances[column]) == first_column_length
                for column in self.columns[1:]
            ]
        ), "All columns must have the same length!"

        self.instances = [
            {column: instances[column][i] for column in self.columns}
            for i in range(len(instances[self.columns[0]]))
        ]
        self.columns_dict = {}
        for column in columns:
            self.columns_dict[column] = np.array(instances[column])

    def __delitem__(self, key):
        if isinstance(key, str):
            del self.columns_dict[key]
            self.columns.remove(key)
            [inst.pop(key) for inst in self.instances]
        elif isinstance(key, int):
            self.instances.pop(key)
            for col in self.columns_dict:
                self.columns_dict[col] = np.delete(self.columns_dict[col], key)
        elif isinstance(key, (list, tuple, np.ndarray)) and isinstance(key[0], int):
            for i in key:
                self.__delitem__(i)
        else:
            raise NotImplementedError

    def remove_columns_(self, columns):
        if isinstance(columns, str):
            self.__delitem__(columns)
        else:
            for column in columns:
                self.__delitem__(column)

    def remove_columns(self, columns):
        if isinstance(columns, str):
            columns = [columns]

        data_without_req_columns = []
        for x in self.instances:
            inst = {k: v for k, v in x.items() if k not in columns}
            data_without_req_columns.append(inst)
        return TransformersDataset(
            data_without_req_columns,
            task=self.task,
            text_column_name=self.text_column_name,
            label_column_name=self.label_column_name,
        )

    def rename_column(self, original_column_name: str, new_column_name: str):
        dataset = deepcopy(self)
        for i, inst in enumerate(dataset.instances):
            dataset.instances[i][new_column_name] = dataset.instances[i][
                original_column_name
            ]
            dataset.instances[i].pop(original_column_name)

        dataset.columns_dict[new_column_name] = dataset.columns_dict[
            original_column_name
        ]
        dataset.columns_dict.pop(original_column_name)
        return dataset

    def map(
        self,
        function,
        batched: bool = False,
        remove_columns: Union[List[str], str, None] = None,
        batch_size: int = 1000,
        **kwargs,
    ):
        instances_to_tokenize = []
        for x in self.instances:
            inst = {}
            for key, value in x.items():
                if isinstance(value, np.ndarray):
                    value = list(value)
                inst[key] = value
            instances_to_tokenize.append(inst)

        num_batches = ceil(len(instances_to_tokenize) / batch_size)
        # TODO: parallelize
        instances_tokenized = []
        for i_batch in tqdm(range(num_batches)):
            instances_batch = self._concatenate_instances(
                instances_to_tokenize[i_batch * batch_size : (i_batch + 1) * batch_size]
            )
            output = function(instances_batch)
            if isinstance(output, BatchEncoding):
                instances_tokenized.append(output.data)
            else:
                raise NotImplementedError

        instances = self._concatenate_batch_encodings_data(instances_tokenized)

        dataset = TransformersDataset(
            instances,
            text_column_name=self.text_column_name,
            label_column_name=self.label_column_name,
            tokenization_kwargs=self.tokenization_kwargs,
            task=self.task,
            id2label=self.id2label,
        )
        if remove_columns is not None:
            columns_to_remove = []
            for x in remove_columns:
                if x in dataset.columns:
                    columns_to_remove.append(x)
            dataset.remove_columns_(columns_to_remove)

        return dataset

    @staticmethod
    def _concatenate_instances(instances: List[dict]):
        keys = instances[0].keys()
        return {key: [inst[key] for inst in instances] for key in keys}

    @staticmethod
    def _concatenate_batch_encodings_data(data: List[Dict]):
        keys = data[0].keys()
        return {key: [obj for inst in data for obj in inst[key]] for key in keys}
