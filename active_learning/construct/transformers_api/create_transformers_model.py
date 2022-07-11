from pathlib import Path
import json
from torch import load
import time
from omegaconf.omegaconf import OmegaConf
from requests.models import HTTPError

from tokenizers import SentencePieceBPETokenizer, Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Lowercase
from tokenizers.processors import TemplateProcessing
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoModelForTokenClassification,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    set_seed,
)
from torch.nn.utils import spectral_norm

from ...models.fnet import FNetForSequenceClassification, FNetForTokenClassification
from ...models.resnet import resnet18
from ...models.text_cnn import TextClassificationCNN, load_embeddings
from ...utils.model_modifications import _get_pre_classifier_dropout_activation


def create_transformers_model_tokenizer(
    model_cfg,
    id2label: dict = None,
    seed: int = 42,
    cache_dir=None,
    embeddings=None,
    word2idx=None,
):

    set_seed(seed)
    pretrained_model_name = model_cfg.name
    num_labels = model_cfg.num_labels if id2label is None else len(id2label)
    label2id = {v: k for k, v in id2label.items()} if id2label is not None else None
    classifier_dropout = model_cfg.classifier_dropout

    model_cache_dir = Path(cache_dir) / "model" if cache_dir is not None else None
    tokenizer_cache_dir = (
        Path(cache_dir) / "tokenizer" if cache_dir is not None else None
    )

    if model_cfg.exists_in_repo:
        model_class = (
            AutoModelForSequenceClassification
            if model_cfg.type == "cls"
            else AutoModelForTokenClassification
            if model_cfg.type == "ner"
            else AutoModelForSeq2SeqLM
            if model_cfg.type == "abs-sum"
            else None
        )
        kwargs = get_classifier_dropout_kwargs(
            pretrained_model_name, classifier_dropout
        )
        if num_labels is not None:
            kwargs["num_labels"] = num_labels
        tokenizer_kwargs = get_tokenizer_kwargs(pretrained_model_name, model_cfg.type)
        # Kristofari sometimes returns an error with connection - need to handle it
        try:
            model = model_class.from_pretrained(
                pretrained_model_name,
                id2label=id2label,
                label2id=label2id,
                cache_dir=model_cache_dir,
                **kwargs
            )
        except HTTPError:
            model = model_class.from_pretrained(
                pretrained_model_name,
                id2label=id2label,
                label2id=label2id,
                cache_dir=model_cache_dir,
                local_files_only=True,
                **kwargs
            )
        if model_cfg.get("use_spectralnorm", False):
            pre_classifier, *_ = _get_pre_classifier_dropout_activation(model)
            spectral_norm(
                pre_classifier,
                n_power_iterations=getattr(model_cfg, "n_power_iterations", 1),
            )
        if "xlnet" in pretrained_model_name:
            model.config.use_mems_eval = False
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name, cache_dir=tokenizer_cache_dir, **tokenizer_kwargs
            )
        except HTTPError:
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name,
                cache_dir=tokenizer_cache_dir,
                local_files_only=True,
                **tokenizer_kwargs
            )
        assert tokenizer is not None, "Failed to load tokenizer"
        if model_cfg.tokenizer_max_length is not None:
            tokenizer.model_max_length = model_cfg.tokenizer_max_length
    else:
        # CV
        if model_cfg.type == "cv_cls":
            model = resnet18(mnist=True)
            tokenizer = None
            return model, tokenizer
        elif model_cfg.type == "cls" and model_cfg.name == "cnn":
            # build CNN for text classification
            if embeddings is None and model_cfg.embeddings_path is not None:
                embeddings, word2idx = load_embeddings(
                    model_cfg.embeddings_path, model_cfg.embeddings_cache_dir
                )
            model = TextClassificationCNN(
                pretrained_embedding=embeddings,
                freeze_embedding=model_cfg.freeze_embedding,
                vocab_size=model_cfg.vocab_size,
                embed_dim=model_cfg.embed_dim,
                filter_sizes=model_cfg.filter_sizes,
                num_filters=model_cfg.num_filters,
                num_classes=num_labels,
                dropout=model_cfg.classifier_dropout,
            )
            # create tokenizer
            tokenizer_model = WordLevel(word2idx, "[UNK]")
            tokenizer = Tokenizer(tokenizer_model)
            tokenizer.normalizer = Lowercase()
            tokenizer.pre_tokenizer = Whitespace()
            hf_tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]"
            )
            return model, hf_tokenizer

        # Implemented only for FNet
        assert model_cfg.name.startswith(
            "fnet"
        ), "Only FNet is supported among models out of HF repo!"
        assert model_cfg.type in [
            "cls",
            "ner",
        ], "Models not from HF repo are currently supported only for NER and classification tasks"

        path_to_pretrained = Path(model_cfg.path_to_pretrained)
        with open(path_to_pretrained / "config.json") as f:
            pretrained_model_cfg = json.load(f)

        model_class = (
            FNetForSequenceClassification
            if model_cfg.type == "cls"
            else FNetForTokenClassification
        )
        model = model_class(pretrained_model_cfg, num_labels)
        model.load_state_dict(
            load(path_to_pretrained / "fnet.statedict.pt"), strict=False
        )

        orig_tokenizer = SentencePieceBPETokenizer.from_file(
            str(path_to_pretrained / "vocab.json"),
            str(path_to_pretrained / "merges.txt"),
        )
        orig_tokenizer.post_processor = TemplateProcessing(
            single="<s> $A </s>",
            pair="<s> $A [SEP] $B:1 </s>:1",
            special_tokens=[("<s>", 1), ("</s>", 2), ("[MASK]", 6), ("[SEP]", 5)],
        )
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=orig_tokenizer,
            model_max_length=model_cfg.tokenizer_max_length,
            bos_token="<s>",
            eos_token="</s>",
            pad_token="<pad>",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]",
        )

    return model, tokenizer


# TODO: add all the other models
def get_classifier_dropout_kwargs(
    pretrained_model_name: str, classifier_dropout: float
):
    if "distilbert" in pretrained_model_name:
        key = "seq_classif_dropout"
    elif "deberta" in pretrained_model_name:
        key = "pooler_dropout"
    elif "xlnet" in pretrained_model_name:
        key = "summary_last_dropout"
    elif "distilrubert" in pretrained_model_name:
        key = "dropout"
    elif "rubert-base" in pretrained_model_name:
        key = "hidden_dropout_prob"
    else:
        key = "classifier_dropout"
    return {key: classifier_dropout}


def get_tokenizer_kwargs(pretrained_model_name: str, task: str = "cls"):
    if "roberta" in pretrained_model_name and task == "ner":
        return dict(add_prefix_space=True)
    return {}
