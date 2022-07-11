"""
CNN model for text classification and utils
Adapted from https://github.com/chriskhanhtran/CNN-Sentence-Classification-PyTorch
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import wget
from collections import defaultdict
import shutil
import numpy as np
from tokenizers.pre_tokenizers import Whitespace
from gensim.models.keyedvectors import KeyedVectors
import logging

log = logging.getLogger()


def load_word_vectors(vectors_name, embeddings_path, model_cache_dir=None):
    """Loads embeddings by url and name."""
    if vectors_name == "fasttext":
        embeddings_name = embeddings_path.split("/")[-1]
        save_path = os.path.join(model_cache_dir, embeddings_name)
        # check if this file already loaded
        os.makedirs(model_cache_dir, exist_ok=True)
        if not (os.path.isfile(save_path)):
            save_path = wget.download(embeddings_path, out=save_path)
        # unzip it and extract data to arrays
        if not (os.path.isfile(os.path.join(model_cache_dir, "crawl-300d-2M.vec"))):
            shutil.unpack_archive(save_path, model_cache_dir)
        fname = "crawl-300d-2M.vec"
    elif vectors_name == "word2vec":
        embeddings_name = "GoogleNews-vectors-negative300.bin.gz"
        save_path = os.path.join(model_cache_dir, embeddings_name)
        # check if this file already loaded
        os.makedirs(model_cache_dir, exist_ok=True)
        if not (os.path.isfile(save_path)):
            save_path = wget.download(embeddings_path, out=save_path)
        if not (os.path.join(model_cache_dir, "GoogleNews-vectors-negative300.txt")):
            model = KeyedVectors.load_word2vec_format(save_path, binary=True)
            model.save_word2vec_format(
                os.path.join(model_cache_dir, "GoogleNews-vectors-negative300.txt"),
                binary=False,
            )
        fname = "GoogleNews-vectors-negative300.txt"
    embeddings = []
    word2idx = {}
    log.info(f"Loading embeddings...")
    with open(os.path.join(model_cache_dir, fname), "r") as f:
        for idx, line in enumerate(f.readlines()):
            if idx == 0:
                # 0 line contains number of words
                continue
            tokens = line.rstrip().split(" ")
            word = tokens[0]
            vector = np.array(tokens[1:], dtype=np.float32)
            # here each word is unique, because we load pretrained embeddings
            word2idx[word] = idx
            embeddings.append(vector)
    # add pad and unk tokens
    word2idx["[UNK]"] = 0
    embeddings.insert(0, np.mean(embeddings, axis=0))
    word2idx["[PAD]"] = idx + 1
    embeddings.append(np.zeros_like(embeddings[-1]))
    embeddings = torch.Tensor(np.asarray(embeddings))
    # After map words to ids and add pad and unk tokens
    # and also return np array of embeddings, not dict
    return embeddings, word2idx


def load_embeddings(embeddings_path, model_cache_dir=None):
    if embeddings_path == "fasttext":
        return load_word_vectors(
            "fasttext",
            "https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip",
            model_cache_dir,
        )
    elif embeddings_path == "word2vec":
        return load_word_vectors(
            "word2vec",
            "https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&confirm=no_antivirus",
            model_cache_dir,
        )
    else:
        raise NotImplementedError


def check_models(config):
    """Check if any of a models is cnn, and get params of the model"""
    models_type = ["model", "acquisition_model", "successor_model", "target_model"]
    for model in models_type:
        if config.get(model, False):
            if config[model]["name"] == "cnn":
                return (
                    config[model]["embeddings_path"],
                    config[model]["embeddings_cache_dir"],
                )
    return None, None


def load_embeddings_with_text(
    data,
    embeddings_path="https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip",
    model_cache_dir=None,
    text_name="text",
):
    """Loads embeddings by url, uses fasttext by default.
    Now works only for fasttext. Also only use words from data for embeddings
    """
    embeddings, word2idx = load_embeddings(embeddings_path, model_cache_dir)
    # now reduce embeddings size
    pre_tokenizer = Whitespace()
    new_word2idx = {}
    new_embeddings = []
    new_word2idx["[UNK]"] = 0
    idx = 1
    for sentence in data:
        # split sentence by words with same pretokenizer
        tokens = pre_tokenizer.pre_tokenize_str(sentence[text_name])
        tokens = [token[0].lower() for token in tokens]
        for token in tokens:
            if token not in new_word2idx and token in word2idx:
                new_word2idx[token] = idx
                new_embeddings.append(
                    np.array(embeddings[word2idx[token]], dtype=np.float32)
                )
                idx += 1
            elif token not in new_word2idx and token not in word2idx:
                # use random vector for embedding
                embedding = np.random.uniform(-0.1, 0.1, embeddings[0].shape)
                new_word2idx[token] = idx
                new_embeddings.append(np.array(embedding, dtype=np.float32))
                idx += 1
    new_embeddings.insert(0, np.mean(new_embeddings, axis=0))
    new_word2idx["[PAD]"] = idx
    new_embeddings.append(np.zeros_like(new_embeddings[-1]))
    new_embeddings = torch.Tensor(np.asarray(new_embeddings))
    log.info(f"Reduced embeddings size from {len(embeddings)} to {len(new_embeddings)}")
    return new_embeddings, new_word2idx


class TextClassificationCNN(nn.Module):
    """An 1D Convulational Neural Network for Sentence Classification."""

    def __init__(
        self,
        pretrained_embedding=None,
        freeze_embedding=False,
        vocab_size=None,
        embed_dim=300,
        filter_sizes=[3, 4, 5],
        num_filters=[100, 100, 100],
        num_classes=2,
        dropout=0.5,
    ):
        """
        The constructor for CNN_NLP class.

        Args:
            pretrained_embedding (torch.Tensor): Pretrained embeddings with
                shape (vocab_size, embed_dim)
            freeze_embedding (bool): Set to False to fine-tune pretraiend
                vectors. Default: False
            vocab_size (int): Need to be specified when not pretrained word
                embeddings are not used.
            embed_dim (int): Dimension of word vectors. Need to be specified
                when pretrained word embeddings are not used. Default: 300
            filter_sizes (List[int]): List of filter sizes. Default: [3, 4, 5]
            num_filters (List[int]): List of number of filters, has the same
                length as `filter_sizes`. Default: [100, 100, 100]
            n_classes (int): Number of classes. Default: 2
            dropout (float): Dropout rate. Default: 0.5
        """

        super(TextClassificationCNN, self).__init__()
        # Embedding layer
        if pretrained_embedding is not None:
            self.vocab_size, self.embed_dim = pretrained_embedding.shape
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embedding,
                padding_idx=self.vocab_size - 1,
                freeze=freeze_embedding,
            )
        else:
            self.embed_dim = embed_dim
            self.embedding = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=self.embed_dim,
                padding_idx=0,
                max_norm=5.0,
            )
        # Conv Network
        self.conv1d_list = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=self.embed_dim,
                    out_channels=num_filters[i],
                    kernel_size=filter_sizes[i],
                )
                for i in range(len(filter_sizes))
            ]
        )
        # Fully-connected layer and Dropout
        self.fc = nn.Linear(np.sum(num_filters), num_classes)
        self.dropout = nn.Dropout(p=dropout)
        self.loss_fn = nn.CrossEntropyLoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, input_ids, labels, **kwargs):
        """Perform a forward pass through the network.

        Args:
            input_ids (torch.Tensor): A tensor of token ids with shape
                (batch_size, max_sent_length)

        Returns:
            logits (torch.Tensor): Output logits with shape (batch_size,
                n_classes)
        """

        # Get embeddings from `input_ids`. Output shape: (b, max_len, embed_dim)
        x_embed = self.embedding(input_ids).float()

        # Permute `x_embed` to match input shape requirement of `nn.Conv1d`.
        # Output shape: (b, embed_dim, max_len)
        x_reshaped = x_embed.permute(0, 2, 1)

        # Apply CNN and ReLU. Output shape: (b, num_filters[i], L_out)
        x_conv_list = [F.relu(conv1d(x_reshaped)) for conv1d in self.conv1d_list]

        # Max pooling. Output shape: (b, num_filters[i], 1)
        x_pool_list = [
            F.max_pool1d(x_conv, kernel_size=x_conv.shape[2]) for x_conv in x_conv_list
        ]

        # Concatenate x_pool_list to feed the fully connected layer.
        # Output shape: (b, sum(num_filters))
        x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list], dim=1)

        # Compute logits. Output shape: (b, n_classes)
        logits = self.fc(self.dropout(x_fc))

        loss = self.loss_fn(logits, labels.view(-1))

        return {"loss": loss, "logits": logits}
