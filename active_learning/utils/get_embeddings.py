import torch
from torch.utils.data import DataLoader

from tqdm.notebook import tqdm
from transformers import DataCollatorWithPadding
from typing import Union
from datasets import Dataset as ArrowDataset
from .transformers_dataset import TransformersDataset
from ..modal_wrapper.transformers_api.modal_transformers import ModalTransformersWrapper
from .model_modifications import (
    get_model_without_cls_layer,
    _get_dim,
    ModelForFeaturesExtraction,
)


def get_embeddings(
    model,
    dataloader_or_data: Union[DataLoader, ArrowDataset, TransformersDataset],
    prepare_model: bool = True,
    use_activation: bool = False,
    use_spectralnorm: bool = False,
    to_eval_mode: bool = True,
    to_numpy: bool = False,
    data_is_tokenized: bool = False,
    batch_size: int = 100,
    use_automodel: bool = False,
    use_averaging: bool = False,
    **tokenization_kwargs,
):
    if prepare_model:
        model_without_cls_layer = get_model_without_cls_layer(
            model, use_activation, use_spectralnorm
        )
    else:
        model_without_cls_layer = model

    device = next(model.parameters()).device

    if not isinstance(dataloader_or_data, DataLoader):
        if not data_is_tokenized:
            dataloader_or_data = ModalTransformersWrapper.tokenize_data(
                data=dataloader_or_data, **tokenization_kwargs
            )
        dataloader_or_data = DataLoader(
            dataloader_or_data,
            shuffle=False,
            batch_size=batch_size,
            collate_fn=DataCollatorWithPadding(
                tokenizer=tokenization_kwargs["tokenizer"]
            ),
            pin_memory=(str(device).startswith("cuda")),
        )

    num_obs = len(dataloader_or_data.dataset)
    dim = _get_dim(model_without_cls_layer)
    if to_eval_mode:
        model_without_cls_layer.eval()

    embeddings = torch.empty((num_obs, dim), dtype=torch.float, device=device)
    if isinstance(model_without_cls_layer, ModelForFeaturesExtraction):
        possible_input_keys = model_without_cls_layer[
            0
        ].model.forward.__code__.co_varnames
    else:
        possible_input_keys = model_without_cls_layer.forward.__code__.co_varnames
    possible_input_keys = list(possible_input_keys) + ["input_ids", "attention_mask"]

    with torch.no_grad():
        start = 0
        for batch in tqdm(dataloader_or_data, desc="Embeddings created"):
            batch = {
                k: v.to(device) for k, v in batch.items() if k in possible_input_keys
            }
            predictions = model_without_cls_layer(**batch)
            if isinstance(model_without_cls_layer, ModelForFeaturesExtraction):
                batch_embeddings = predictions
            # TODO: make smarter
            elif use_automodel and not use_activation:
                if use_averaging:
                    batch_embeddings = predictions.last_hidden_state.mean(1)
                else:
                    batch_embeddings = predictions.last_hidden_state[:, 0]
            elif "pooler_output" in predictions.keys():
                batch_embeddings = predictions.pooler_output
            elif "last_hidden_state" in predictions.keys():
                batch_embeddings = predictions.last_hidden_state
            else:
                raise NotImplementedError

            end = start + len(batch["input_ids"])  # len(batch[list(batch.keys())[0]])
            embeddings[start:end].copy_(batch_embeddings, non_blocking=True)
            start = end

    if to_numpy:
        return embeddings.cpu().detach().numpy()
    return embeddings
