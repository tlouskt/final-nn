# Imports
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike

def sample_seqs(seqs: List[str], labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample the given sequences to account for class imbalance. 
    Consider this a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """
    #initialize output
    sampled_seqs = []
    sampled_labels = []

    #get positive and negative sequences
    pos_seqs = seqs[labels == True]
    neg_seqs = seqs[labels == False]

    #if balanced
    if len(pos_seqs) == len(neg_seqs):
        sampled_seqs = list(seqs)
        sampled_labels = list(labels)
    #if pos < neg, sample positive more with replacement with length of negative seqs
    elif len(pos_seqs) < len(neg_seqs):
        over_pos = pos_seqs[np.random.choice(len(pos_seqs), len(neg_seqs), replace = True)]
        #new list of sequences and labels that correspond to oversampled dataset
        sampled_seqs = list(np.concatenate(neg_seqs, over_pos), axis=None)
        sampled_labels = list([True] * len(over_pos) + [False] * len(neg_seqs))
    #if neg < pos, sample negative more
    else:
        len(pos_seqs) > len(neg_seqs)
        over_neg = neg_seqs[np.random.choice(len(neg_seqs), len(pos_seqs), replace = True)]
        sampled_seqs = list(np.concatenate(pos_seqs, over_neg), axis=None)
        sampled_labels = list([True] * len(pos_seqs) + [False] * len(over_neg))

    return sampled_seqs, sampled_labels


def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one-hot encoding of a list of DNA sequences
    for use as input into a neural network.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence.
            For example, if we encode:
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0].
    """
    encoding = {
        "A": [1, 0, 0, 0],
        "T": [0, 1, 0, 0],
        "C": [0, 0, 1, 0],
        "G": [0, 0, 0, 1]
    }
    #initialize list to store one hot encoded sequence where dims are num sequences in arr and then length of seq*4 for one-hot
    one_hot_encodings = []
    #iterate through each sequence
    for seq in seq_arr:
        #list to store one hot bases in current sequenc
        base_one_hot = []
        #iterate through each base in sequence
        for base in seq:
            #add the one hot encoded base to list
            base_one_hot.append(encoding[base])
        #add flattened seq encoding to list of encodings
        base_one_hot = np.array(base_one_hot)
        base_one_hot = base_one_hot.flatten()
        one_hot_encodings.append(base_one_hot)
    return np.array(one_hot_encodings)
