"""Corpus processing utilities for word embeddings."""

import numpy as np
from dpl.datasets import Dataset


def preprocess(text: str):
    """
    Preprocess text corpus and create word-to-id and id-to-word mappings.

    Args:
        text: Input text string

    Returns:
        corpus: Array of word IDs
        word_to_id: Dictionary mapping words to IDs
        id_to_word: Dictionary mapping IDs to words
    """
    # Convert to lowercase and split into words
    text = text.lower()
    text = text.replace(".", " .")
    words = text.split()

    # Create word-to-id and id-to-word mappings
    word_to_id = {}
    id_to_word = {}

    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    # Create corpus (array of word IDs)
    corpus = np.array([word_to_id[word] for word in words], dtype=np.int32)

    return corpus, word_to_id, id_to_word


def create_contexts_target(corpus: np.ndarray, window_size: int = 1):
    """
    Create context-target pairs from corpus.

    Args:
        corpus: Array of word IDs
        window_size: Size of context window (default: 1)

    Returns:
        contexts: Array of context word IDs (num_samples, 2*window_size)
        target: Array of target word IDs (num_samples,)
    """
    target = corpus[window_size:-window_size]
    contexts = []

    for idx in range(window_size, len(corpus) - window_size):
        cs = []
        for t in range(-window_size, window_size + 1):
            if t == 0:
                continue
            cs.append(corpus[idx + t])
        contexts.append(cs)

    return np.array(contexts, dtype=np.int32), np.array(target, dtype=np.int32)


class CBOWDataset(Dataset):
    """
    Dataset for CBOW model that returns context words and target word.
    """

    def __init__(self, contexts: np.ndarray, target: np.ndarray):
        """
        Args:
            contexts: Array of context word IDs (num_samples, 2*window_size)
            target: Array of target word IDs (num_samples,)
        """
        self.contexts = contexts
        self.target = target

    def __getitem__(self, index):
        """
        Returns:
            contexts: Context word IDs for this sample (2,) for window_size=1
            target: Target word ID
        """
        return self.contexts[index], self.target[index]

    def __len__(self):
        return len(self.target)

    def prepare(self):
        pass
