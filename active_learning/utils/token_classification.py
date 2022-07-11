from typing import List


def align_labels_with_tokens(
    labels: List[int], word_ids: List[int], tag_for_subtokens=None
):
    """
    Function to adjust the labels according to the tokenized input
    :param labels: original list of tags
    :param word_ids: list with ids of the original words in the tokenized list of tokens;
    can be extracted via `inputs.word_ids()`, where `inputs = tokenizer(...)`
    :param tag_for_subtokens: how to tokenize subtokens (BPEs) of a word. Have three options:
    imagine we have a word `loving` -> [`lov`, `###ing`].
    Option 1 (default) = None: `lov` -> B-TAG, `###ing` -> O
    Option 2 = 'i': `lov` -> B-TAG, `###ing` -> I-TAG
    Option 3 = 'b': `lov` -> B-TAG, `###ing` -> B-TAG
    """
    assert tag_for_subtokens in [
        None,
        "i",
        "b",
    ], "Param :tag_for_subtokens must be one of `None`, `'i'`, `'b'`!"
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id is None:
            # Special token
            new_labels.append(-100)
        elif word_id != current_word:
            # Start of a new word!
            current_word = word_id
            new_labels.append(labels[word_id])
        else:
            # Same word as previous token
            if tag_for_subtokens is not None:
                label = labels[word_id]
                # If the label is B-XXX we change it to I-XXX
                if tag_for_subtokens == "i" and label % 2 == 1:
                    label += 1
            else:
                label = -100
            new_labels.append(label)

    return new_labels
