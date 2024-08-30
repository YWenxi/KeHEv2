import tensorflow as tf

class KGELoss(tf.keras.losses.Loss):
    def __init__(self, margin=6.0, reduction=tf.keras.losses.Reduction.AUTO):
        """Just Negative Sampling."""
        super(KGELoss, self).__init__(reduction=reduction)
        self.margin = margin

    def call(self, pos_scores, neg_scores):
        """
        Compute the margin-based ranking loss for knowledge graph embeddings.

        Args:
            pos_scores: Tensor of positive triple scores.
            neg_scores: Tensor of negative triple scores.

        Returns:
            loss: The computed margin-based loss.
        """
        return tf.reduce_mean(tf.nn.relu(self.margin + pos_scores - neg_scores))


class HierarchicalLoss(tf.keras.losses.Loss):
    def __init__(self, margin_hierarchical=12.0, reduction=tf.keras.losses.Reduction.AUTO):
        super(HierarchicalLoss, self).__init__(reduction=reduction)
        self.margin_hierarchical = margin_hierarchical

    def score_function(self, entity_embeddings, type_embeddings):
        """
        Compute the score function F_e.

        Args:
            entity_embeddings: Embeddings for entities (batch_size, embedding_dim).
            type_embeddings: Embeddings for types (batch_size, embedding_dim).

        Returns:
            score: The computed score for hierarchical loss.
        """
        diff_square = tf.square(type_embeddings) - tf.square(entity_embeddings)
        diff_square_clipped = tf.maximum(0.0, diff_square)
        score = tf.reduce_sum(diff_square_clipped, axis=1)
        return score

    def __call__(self, entity_embeddings, type_embeddings, labels):
        """
        Compute the hierarchical loss.

        Args:
            entity_embeddings: Embeddings for entities (batch_size, embedding_dim).
            type_embeddings: Embeddings for types (batch_size, embedding_dim).
            labels: Binary labels (1 for positive, 0 for negative).

        Returns:
            loss: The computed hierarchical loss.
        """
        scores = self.score_function(entity_embeddings, type_embeddings)
        probabilities = tf.sigmoid(self.margin_hierarchical - scores)

        positive_loss = -tf.math.log(tf.clip_by_value(1.0 - probabilities, 1e-10, 1.0))
        negative_loss = -tf.math.log(tf.clip_by_value(probabilities, 1e-10, 1.0))

        loss = tf.where(labels == 1, positive_loss, negative_loss)
        return tf.reduce_mean(loss)