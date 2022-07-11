from torch import nn
from torch.nn.utils import spectral_norm, remove_spectral_norm
from torch.nn.utils.spectral_norm import SpectralNorm
from transformers import (
    BertForSequenceClassification,
    DistilBertForSequenceClassification,
    RobertaForSequenceClassification,
    ElectraForSequenceClassification,
    XLNetForSequenceClassification,
)


class Encoder(nn.Module):
    def __init__(self, model, model_name):
        super().__init__()
        self.name = model_name
        self.model = next(model.children())
        if model_name == "bert":
            self.model = nn.Sequential(*list(self.model.children())[:-1])

    def forward(self, *args, **kwargs):
        out = self.model(*args, **kwargs)
        return out.last_hidden_state[:, 0, :]


class ModelForFeaturesExtraction(nn.Sequential):
    def forward(self, *args, **kwargs):
        for module in self._modules.values():
            break
        inputs = module(*args, **kwargs)
        for i, module in enumerate(self._modules.values()):
            if i:
                inputs = module(inputs)
        return inputs


def get_model_without_cls_layer(model, use_activation=False, use_spectralnorm=False):
    model_name = get_name(model)
    model.eval()

    model_without_cls_layer = ModelForFeaturesExtraction(Encoder(model, model_name))
    _add_modules(
        model_without_cls_layer=model_without_cls_layer,
        original_model=model,
        use_activation=use_activation,
        use_spectralnorm=use_spectralnorm,
    )

    return model_without_cls_layer


def _add_modules(
    model_without_cls_layer, original_model, use_activation=False, use_spectralnorm=True
):
    pre_classifier, dropout, activation = _get_pre_classifier_dropout_activation(
        original_model
    )
    model_without_cls_layer.add_module("dropout", dropout)
    pre_classifier = _modify_layer_spectral_norm(
        pre_classifier, enable=use_spectralnorm
    )
    model_without_cls_layer.add_module("pre_classifier", pre_classifier)
    if use_activation:
        model_without_cls_layer.add_module("activation", activation)


def _modify_layer_spectral_norm(layer, enable=True):
    """
    Function to avoid overriding the layer data
    """
    # Check if spectral norm is already integrated
    name = "weight"
    sn_already_integrated = False
    for k, hook in layer._forward_pre_hooks.items():
        if isinstance(hook, SpectralNorm) and hook.name == name:
            sn_already_integrated = True
            break

    if (sn_already_integrated and not enable) or (not sn_already_integrated and enable):
        layer_copy = nn.Linear(layer.in_features, layer.out_features)
        layer_copy.weight.data = layer.weight.data.clone()
        layer_copy.bias.data = layer.bias.data.clone()
        if enable:
            return spectral_norm(layer_copy)
        return layer_copy
    return layer


def _get_dim(model_without_cls_layer):
    return list(model_without_cls_layer.state_dict().items())[-1][1].shape[0]


def _get_pre_classifier_dropout_activation(model):
    if isinstance(model, BertForSequenceClassification):
        return model.bert.pooler.dense, nn.Identity(), nn.Tanh()
    elif isinstance(model, DistilBertForSequenceClassification):
        return model.pre_classifier, nn.Identity(), nn.ReLU()
    elif isinstance(model, RobertaForSequenceClassification):
        return model.classifier.dense, model.classifier.dropout, nn.Tanh()
    elif isinstance(model, ElectraForSequenceClassification):
        return model.classifier.dense, model.classifier.dropout, nn.GELU()
    elif isinstance(model, XLNetForSequenceClassification):
        return (
            model.sequence_summary.summary,
            model.sequence_summary.first_dropout,
            nn.Tanh(),
        )
    else:
        raise NotImplementedError


def get_name(model):
    if isinstance(model, BertForSequenceClassification):
        return "bert"
    elif isinstance(model, DistilBertForSequenceClassification):
        return "distilbert"
    elif isinstance(model, RobertaForSequenceClassification):
        return "roberta"
    elif isinstance(model, ElectraForSequenceClassification):
        return "google/electra"
    elif isinstance(model, XLNetForSequenceClassification):
        return "xlnet"
    else:
        raise NotImplementedError


def _try_remove_spectral_norm(layer):
    try:
        # remove SN, if it used on train
        remove_spectral_norm(layer)
    except:
        pass
