import tensorflow as tf
import numpy as np
from tqdm import tqdm

def generate_negative_samples(positive_triples, num_entities):
    """
    Generate negative samples by corrupting the tail entity.
    
    Args:
        positive_triples: Tensor of shape (batch_size, 3) containing positive triples.
        num_entities: Total number of entities in the dataset.
    
    Returns:
        negative_triples: Tensor of shape (batch_size, 3) containing negative triples.
    """
    heads, relations, tails = positive_triples[:, 0], positive_triples[:, 1], positive_triples[:, 2]
    neg_tails = tf.random.uniform(shape=tails.shape, minval=0, maxval=num_entities, dtype=heads.dtype)
    negative_triples = tf.stack([heads, relations, neg_tails], axis=1)
    return negative_triples

from collections import defaultdict
import numpy as np
from sklearn.metrics import accuracy_score

def compute_thresholds(model, train_dataset, num_relations, num_entities):
    """
    Compute the threshold δr for each relation r using the training dataset.
    
    Args:
        model: The trained KGE model.
        train_dataset: The training dataset (tf.data.Dataset).
        num_relations: Total number of relations in the dataset.
        num_entities: Total number of entities in the dataset.
    
    Returns:
        thresholds: A numpy array of shape (num_relations,) containing thresholds for each relation.
    """
    # Initialize a dictionary to hold positive and negative scores for each relation
    relation_scores = defaultdict(lambda: {'pos_scores': [], 'neg_scores': []})

    for batch, _ in tqdm(train_dataset, desc="Computing Scores for all training Samples."):
        # Generate negative samples
        positive_triples = batch
        negative_triples = generate_negative_samples(positive_triples, num_entities)

        # Compute positive and negative scores
        pos_scores = model(positive_triples).numpy()
        neg_scores = model(negative_triples).numpy()

        # Store scores by relation
        for i in range(len(positive_triples)):
            r = positive_triples[i, 1].numpy()
            relation_scores[r]['pos_scores'].append(pos_scores[i])
            relation_scores[r]['neg_scores'].append(neg_scores[i])

    # Compute thresholds
    thresholds = np.zeros(num_relations)
    for r in tqdm(range(num_relations)):
        if len(relation_scores[r]['pos_scores']) == 0 or len(relation_scores[r]['neg_scores']) == 0:
            continue  # Skip if there are no samples for this relation

        r_pos_scores = np.array(relation_scores[r]['pos_scores'])
        r_neg_scores = np.array(relation_scores[r]['neg_scores'])

        # Combine positive and negative scores
        all_scores = np.concatenate([r_pos_scores, r_neg_scores])
        all_labels = np.concatenate([np.ones_like(r_pos_scores), np.zeros_like(r_neg_scores)])

        # Find the best threshold for the current relation
        best_threshold = find_best_threshold(all_scores, all_labels)
        thresholds[r] = best_threshold

    return thresholds

def find_best_threshold(scores, labels):
    """
    Find the best threshold to separate positive and negative samples.
    
    Args:
        scores: Array of scores for both positive and negative samples.
        labels: Array of labels (1 for positive, 0 for negative).
    
    Returns:
        best_threshold: The threshold that best separates the positive and negative samples.
    """
    best_threshold = 0.0
    best_metric = 0.0

    sorted_scores = np.sort(scores)
    start = np.min(scores)
    end = np.max(scores)
    for threshold in np.linspace(start, end, 101):
        predictions = scores < threshold
        metric = accuracy_score(labels, predictions)
        if metric > best_metric:
            best_metric = metric
            best_threshold = threshold

    return best_threshold

from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def evaluate_triple_classification(model, test_dataset, thresholds, num_entities):
    """
    Evaluate the triple classification task using the computed thresholds.
    
    Args:
        model: The trained KGE model.
        test_dataset: The test dataset (tf.data.Dataset).
        thresholds: The computed thresholds for each relation (numpy array).
        num_entities: Total number of entities in the dataset.
    
    Returns:
        metrics: A dictionary containing Accuracy, Precision, Recall, and F1-Score.
    """
    # all_labels = []
    all_pos_predictions = []
    all_neg_predictions = []

    for batch in tqdm(test_dataset):
        positive_triples = batch
        negative_triples = generate_negative_samples(positive_triples, num_entities)

        # Compute positive and negative scores
        pos_scores = model(positive_triples).numpy()
        neg_scores = model(negative_triples).numpy()
        
        thresholds_for_batch = tf.gather(thresholds, batch[:, 1])
        pos_prediction = pos_scores < thresholds_for_batch
        neg_prediction = neg_scores < thresholds_for_batch
        all_pos_predictions.append(pos_prediction)
        all_neg_predictions.append(neg_prediction)

    all_pos_predictions = np.concatenate(all_pos_predictions)
    all_neg_predictions = np.concatenate(all_neg_predictions)
    all_predictions = np.concatenate([all_pos_predictions, all_neg_predictions])
    all_labels = np.concatenate([np.ones_like(all_pos_predictions), np.zeros_like(all_neg_predictions)])

    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='binary')
    accuracy = accuracy_score(all_labels, all_predictions)

    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    }
    
    return metrics