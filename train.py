import os
import tensorflow as tf
import numpy as np
from preprocessing.dataloader import DataLoader
from models.kge import KGEModel
from models.loss import KGELoss, HierarchicalLoss
from evaluation.link_prediction import evaluate_link_prediction
from evaluation.triple_classification import evaluate_triple_classification, compute_thresholds
import argparse
import random
import logging


def setup_logger(exp_name):
    """Set up logging to file and console."""
    log_dir = f'results/{exp_name}'
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(f'{log_dir}/training.log', mode='w'),
                            logging.StreamHandler()
                        ])
    logger = logging.getLogger()
    return logger

# def self_adversarial_negative_sampling(pos_scores, neg_scores, alpha=0.5):
#     """
#     Apply self-adversarial negative sampling as described in the paper.
    
#     Args:
#         pos_scores: Tensor of positive triple scores.
#         neg_scores: Tensor of negative triple scores.
#         alpha: Temperature for sampling.
    
#     Returns:
#         sampled_neg_scores: Tensor of sampled negative triple scores.
#     """
#     # Compute the weights for sampling
#     neg_weights = tf.exp(neg_scores * alpha)
#     sampled_neg_scores = tf.reduce_sum(neg_weights * neg_scores) / tf.reduce_sum(neg_weights)
#     return sampled_neg_scores

# import tensorflow as tf

def self_adversarial_negative_sampling(pos_score, neg_scores, gamma=1.0, alpha=0.5):
    """
    Apply self-adversarial negative sampling as described in the paper.
    
    Args:
        pos_score: Tensor of the positive triple score.
        neg_scores: Tensor of negative triple scores.
        gamma: Margin used in the loss.
        alpha: Temperature for sampling.
    
    Returns:
        loss: The computed self-adversarial negative sampling loss.
    """
    # Compute the probability distribution p over the negative scores
    neg_weights = tf.nn.softmax(neg_scores * alpha)
    
    # Compute the weighted negative loss
    neg_loss = tf.reduce_sum(neg_weights * tf.math.log_sigmoid(neg_scores - gamma))
    
    # Compute the positive loss
    pos_loss = tf.math.log_sigmoid(gamma - pos_score)
    
    # Combine the positive and negative loss
    loss = -(pos_loss + neg_loss)
    
    return tf.reduce_mean(loss)


def train_step(model, optimizer, batch, labels, kge_loss_fn, 
               hierarchical_loss_fn: HierarchicalLoss, beta):
    """
    Perform a single training step.
    
    Args:
        model: The KGEModel instance (TransE, RotatE).
        optimizer: The optimizer instance.
        batch: The batch of triples (head, relation, tail).
        labels: Labels indicating whether each triple is an isA triple (1) or not (0).
        kge_loss_fn: The loss function for the usual embedding loss (TransE-style).
        hierarchical_loss_fn: The loss function for the hierarchical loss.
        beta: Weight for the hierarchical loss.
    
    Returns:
        total_loss: The total loss for this step.
    """
    with tf.GradientTape() as tape:
        # Positive and negative sampling
        heads, relations, tails = batch[:, 0], batch[:, 1], batch[:, 2]
        pos_scores = model((heads, relations, tails))

        # Negative sampling: replace the tail with a random entity
        neg_tails = tf.random.uniform(
            shape=tails.shape, minval=0, maxval=model.kge.num_entities, dtype=tf.int32
        )
        neg_scores = model((heads, relations, neg_tails))

        # Apply self-adversarial sampling
        # sampled_neg_scores = self_adversarial_negative_sampling(pos_scores, neg_scores, alpha=adversarial_sampling)

        # Compute the embedding loss (usual TransE-style loss)
        embedding_loss = kge_loss_fn(pos_scores, neg_scores)
        
        # Compute the hierarchical loss
        head_embedding = model.kge.entity_embedding(batch[:, 0])
        tail_embedding = model.kge.entity_embedding(batch[:, 2])
        hierarchical_loss = hierarchical_loss_fn(head_embedding, tail_embedding, labels)

        # Combine the losses
        total_loss = embedding_loss + beta * hierarchical_loss

    # Apply gradients
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return total_loss

def train(data_loader: DataLoader, model, optimizer, num_epochs, beta, validate_every=100, 
          exp_name="experiment", margin_hierarchical=6.0, margin_kge=2.0):
    """
    Training loop.
    
    Args:
        data_loader: The DataLoader instance.
        model: The KGEModel instance.
        optimizer: The optimizer instance.
        num_epochs: The number of training epochs.
        beta: Weight for the hierarchical loss.
    """
    kge_loss_fn = KGELoss(margin=margin_kge)
    hierarchical_loss_fn = HierarchicalLoss(margin_hierarchical=margin_hierarchical)
    test_batches = list(data_loader.get_test_dataset())
    
    logger = setup_logger(exp_name)
    logger.info(f"Starting training with experiment name: {exp_name}")
    
    # Log the configuration
    logger.info(f"Configurations: \n"
                f"num_epochs={num_epochs}, "
                f"beta={beta}, "
                f"model={model.model_name}, "
                f"optimizer={optimizer}, "
                f"margin_hierarchical={margin_hierarchical}")
    
    for epoch in range(num_epochs):
        # print(f"Epoch {epoch+1}/{num_epochs}")
        total_loss = 0.0
        for i, (batch, labels) in enumerate(data_loader.get_train_dataset()):
            batch_loss = train_step(model, optimizer, batch, labels, kge_loss_fn, hierarchical_loss_fn, beta)
            total_loss += batch_loss
            if (i + 1) % 100 == 0:
                logger.info(f"Epoch {epoch+1:3d}/{num_epochs}, Iters {i+1:4d}: Loss: {batch_loss.numpy():.6f}")
        
        if (epoch + 1) % validate_every == 0:
            # Validate on a random test batch
            random_test_batch = random.choice(test_batches)
            random_test_batch = tf.convert_to_tensor(random_test_batch)
            metrics = evaluate_link_prediction(model, [random_test_batch], num_entities=model.kge.num_entities)
            logger.info(f"Validation after Epoch {epoch+1}: Hits@1: {metrics['Hits@1']:.4f}, "
                        f"Hits@10: {metrics['Hits@10']:.4f}, MRR: {metrics['MRR']:.4f}")
            logger.info(f"Save Model at checkpoint/epoch {epoch+1}.")
            model.save_weights(os.path.join(f'results/{exp_name}', f'model_weights.h5'))
    
    metrics = evaluate_link_prediction(model, data_loader.get_test_dataset(), num_entities=model.kge.num_entities)
    logger.info(f"Evaluate Link Prediction after Training {epoch+1}: Hits@1: {metrics['Hits@1']:.4f}, "
                f"Hits@10: {metrics['Hits@10']:.4f}, MRR: {metrics['MRR']:.4f}")
    
    # Assuming model, data_loader, and num_relations are already defined
    # Compute thresholds using the training dataset
    thresholds = compute_thresholds(model, data_loader.get_train_dataset(), 
                                    num_relations=model.kge.num_relations, num_entities=model.kge.num_entities)

    # Evaluate on the test dataset
    classification_metrics = evaluate_triple_classification(
        model, data_loader.get_test_dataset(), thresholds, num_entities=model.kge.num_entities)

    # Print the evaluation results
    logger.info(f"Triple Classification Metrics: Accuracy: {classification_metrics['Accuracy']:.4f}, Precision: {classification_metrics['Precision']:.4f}, Recall: {classification_metrics['Recall']:.4f}, F1-Score: {classification_metrics['F1-Score']:.4f}")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a Knowledge Graph Embedding Model')
    parser.add_argument('--data_dir', type=str, default="data/yago", help='Path to the dataset directory')
    parser.add_argument('--batch_size', type=int, default=2000, help='Batch size for training')
    parser.add_argument('--embedding_dim', type=int, default=200, help='Dimension of the embeddings')
    parser.add_argument('--num_epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--beta', type=float, default=0.5, help='Weight for the hierarchical loss')
    parser.add_argument('--margin_kge', type=float, default=4.0, help='Margin for the kg embedding score')
    parser.add_argument('--margin_hier', type=float, default=4.0, help='Margin for the hierarchical loss')
    parser.add_argument('--model_name', type=str, default='TransE', choices=['TransE', 'RotatE'], help='The KGE model to use')
    
    args = parser.parse_args()
    
    # Generate experiment name based on configurations
    exp_name = f"{args.model_name}_dim{args.embedding_dim}_lr{args.learning_rate}_beta{args.beta}_bs{args.batch_size}_"\
                f"marginkge{args.margin_kge}_marginhier{args.margin_hier}_data{args.data_dir[5:]}"
    
    data_loader = DataLoader(data_dir=args.data_dir, batch_size=args.batch_size, method=2)
    model = KGEModel(model_name=args.model_name, num_entities=data_loader.num_entities, 
                     num_relations=data_loader.num_relations, 
                     embedding_dim=args.embedding_dim)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    
    train(data_loader, model, optimizer, num_epochs=args.num_epochs, beta=args.beta, exp_name=exp_name, 
          margin_hierarchical=args.margin_hier)