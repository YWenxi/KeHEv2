import numpy as np
import tensorflow as tf
from tqdm import tqdm

def rank_entities(scores, true_tail_idx):
    """
    Given a score matrix and the index of the true tail, return the rank of the true tail.
    
    Args:
        scores: A 1D numpy array of scores for each entity.
        true_tail_idx: The index of the true tail entity.
    
    Returns:
        rank: The rank of the true tail entity (1-based index).
    """
    # sorted_indices = np.argsort(-scores)  # Sort scores in descending order
    sorted_indices = np.argsort(scores)
    rank = np.where(sorted_indices == true_tail_idx)[0][0] + 1  # Find the true tail rank (1-based index)
    return rank

def hits_at_k(ranks, k):
    """
    Compute Hits@K metric.
    
    Args:
        ranks: A list or numpy array of ranks.
        k: The value of K.
    
    Returns:
        hits_k: The proportion of ranks <= K.
    """
    return np.mean(ranks <= k)

def mrr(ranks):
    """
    Compute Mean Reciprocal Rank (MRR).
    
    Args:
        ranks: A list or numpy array of ranks.
    
    Returns:
        mrr: The mean of the reciprocal ranks.
    """
    return np.mean(1.0 / ranks)

def evaluate_link_prediction(model, test_dataset, num_entities):
    """
    Evaluate the model on link prediction using Hits@1, Hits@10, and MRR.
    
    Args:
        model: The trained KGE model.
        test_dataset: The test dataset (tf.data.Dataset).
        num_entities: Total number of entities in the dataset.
    
    Returns:
        metrics: A dictionary containing Hits@1, Hits@10, and MRR.
    """
    ranks = []

    for batch in tqdm(test_dataset, desc="Validation", unit="batch"):
        heads, relations, true_tails = batch[:, 0], batch[:, 1], batch[:, 2]
        
        for i in range(len(heads)):
            head, relation, true_tail = heads[i], relations[i], true_tails[i]

            # Compute scores for all possible tail entities
            scores = model((tf.fill([num_entities], head), tf.fill([num_entities], relation), np.arange(num_entities)))

            # Convert scores to numpy for further processing
            scores = scores.numpy()

            # Get the rank of the true tail entity
            true_tail_idx = true_tail.numpy()
            rank = rank_entities(scores, true_tail_idx)
            ranks.append(rank)

    ranks = np.array(ranks)
    
    metrics = {
        "Hits@1": hits_at_k(ranks, 1),
        "Hits@10": hits_at_k(ranks, 10),
        "MRR": mrr(ranks)
    }
    
    return metrics