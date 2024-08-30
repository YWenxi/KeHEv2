import tensorflow as tf
# from numpy import sqrt, savetxt, loadtxt
import numpy as np
from pathlib import Path


class BaseKGEModel(tf.keras.Model):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(BaseKGEModel, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        
        # Initialize embeddings
        uniform_range = 6 / np.sqrt(self.embedding_dim)
        self.entity_embedding = tf.keras.layers.Embedding(
            num_entities, embedding_dim, 
            embeddings_initializer=tf.keras.initializers.RandomUniform(minval=-uniform_range, maxval=uniform_range)
        )
        self.relation_embedding = tf.keras.layers.Embedding(
            num_relations, embedding_dim,
            embeddings_initializer=tf.keras.initializers.RandomUniform(minval=-uniform_range, maxval=uniform_range)
        ) 
    
    def compute_score(self, heads, relations, tails):
        """Compute the score for a batch of triples. This should be implemented by subclasses."""
        raise NotImplementedError
    
    def call(self, inputs):
        """Forward pass to compute scores for a batch of triples."""
        if isinstance(inputs, tuple):
            heads, relations, tails = inputs
        else:
            heads, relations, tails = inputs[:, 0], inputs[:, 1], inputs[:, 2]
        return self.compute_score(heads, relations, tails)
    
    def normalize_embeddings(self):
        """Normalize entity and relation embeddings if needed. Can be overridden by subclasses."""
        pass


class TransE(BaseKGEModel):
    def __init__(self, num_entities, num_relations, embedding_dim, margin=1.0, normalize="l2"):
        super(TransE, self).__init__(num_entities, num_relations, embedding_dim)
        self.margin = margin
        
        if normalize == "l2":
            self.normalize = tf.math.l2_normalize
        else:
            # TODO in the future
            raise NotImplementedError
    
    def compute_score(self, heads, relations, tails, ord=2):
        """Compute the TransE score for a batch of triples."""
        head_embeds = self.entity_embedding(heads)
        relation_embeds = self.relation_embedding(relations)
        tail_embeds = self.entity_embedding(tails)
        
        # score = tf.reduce_sum(tf.abs(head_embeds + relation_embeds - tail_embeds), axis=1)
        score = tf.norm(head_embeds + relation_embeds - tail_embeds, axis=1, ord=ord)
        return score
    
    def normalize_embeddings(self):
        """Normalize entity embeddings to unit length."""
        self.entity_embedding.embeddings.assign(tf.math.l2_normalize(self.entity_embedding.embeddings, axis=1))


class RotatE(BaseKGEModel):
    def __init__(self, num_entities, num_relations, embedding_dim, margin=1.0):
        super(RotatE, self).__init__(num_entities, num_relations, embedding_dim)
        self.margin = margin
        self.pi = 3.14159265358979323846
    
    def compute_score(self, heads, relations, tails):
        """Compute the RotatE score for a batch of triples."""
        head_embeds = self.entity_embedding(heads)
        relation_embeds = self.relation_embedding(relations)
        tail_embeds = self.entity_embedding(tails)

        # Normalize the relation embeddings into [0, 2pi]
        relation_embeds = relation_embeds / (self.embedding_dim / self.pi)
        
        # Convert to complex numbers
        re_relation_embeds = tf.cos(relation_embeds)
        im_relation_embeds = tf.sin(relation_embeds)

        re_head_embeds = head_embeds
        im_head_embeds = tf.zeros_like(head_embeds)
        
        re_tail_embeds = tail_embeds
        im_tail_embeds = tf.zeros_like(tail_embeds)
        
        re_pred = re_head_embeds * re_relation_embeds - im_head_embeds * im_relation_embeds
        im_pred = re_head_embeds * im_relation_embeds + im_head_embeds * re_relation_embeds
        
        score = tf.reduce_sum(tf.abs(re_pred - re_tail_embeds) + tf.abs(im_pred - im_tail_embeds), axis=1)
        return score
    
    def normalize_embeddings(self):
        """No normalization needed for RotatE, but method must be implemented."""
        pass


class KGEModel(tf.keras.Model):
    def __init__(self, model_name, num_entities, num_relations, embedding_dim, margin=1.0, pep=None):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        if model_name == 'TransE':
            self.kge = TransE(num_entities, num_relations, embedding_dim, margin)
        elif model_name == 'RotatE':
            self.kge = RotatE(num_entities, num_relations, embedding_dim, margin)
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
        
        # Learnable threshold parameters for pruning
        self.pep = pep
        if pep is None:
            print("No pruning.")
        elif pep == 'global':
            # Global-wise pruning: use a scalar threshold for all elements
            self.entity_threshold = tf.Variable(-15.0, trainable=True, dtype=tf.float32)
            self.relation_threshold = tf.Variable(-15.0, trainable=True, dtype=tf.float32)
        elif pep == 'dimension':
            # Dimension-wise pruning: use a vector threshold for each dimension
            self.entity_threshold = tf.Variable(tf.fill([self.kge.embedding_dim], -15.0), 
                                                trainable=True, dtype=tf.float32)
            self.relation_threshold = tf.Variable(tf.fill([self.kge.embedding_dim], -15.0), 
                                                  trainable=True, dtype=tf.float32)
        else:
            raise ValueError(f"Unsupported pruning strategy: {pep}")
    
    # ----------------Soft Threshold------------------------ 
    def soft_thresholding(self, embeddings, threshold):
        """
        Soft thresholding function to prune embeddings.
        """
        # if len(threshold.shape) == 0:  # Global threshold (scalar)
        #     threshold_value = tf.nn.sigmoid(threshold)  # g(s)
        #     pruned_embeddings = tf.sign(embeddings) * tf.nn.relu(tf.abs(embeddings) - threshold_value)
        # elif len(threshold.shape) == 1:  # Dimension-wise threshold (vector)
        #     threshold_values = tf.nn.sigmoid(threshold)  # g(s)
        #     pruned_embeddings = tf.sign(embeddings) * tf.nn.relu(tf.abs(embeddings) - threshold_values)
        # else:
        #     raise ValueError("Threshold shape is not supported.")
        threshold_values = tf.nn.sigmoid(threshold)  # g(s)
        pruned_embeddings = tf.sign(embeddings) * tf.nn.relu(tf.abs(embeddings) - threshold_values)
        
        return pruned_embeddings
    
    def pruned_entity_embedding(self, indices):
        """
        Retrieve pruned entity embeddings.
        """
        assert self.pep is not None
        embeddings = self.kge.entity_embedding(indices)  # Original embeddings
        pruned_embeddings = self.soft_thresholding(embeddings, self.entity_threshold)
        return pruned_embeddings

    def pruned_relation_embedding(self, indices):
        """
        Retrieve pruned relation embeddings.
        """
        assert self.pep is not None
        embeddings = self.kge.relation_embedding(indices)  # Original embeddings
        pruned_embeddings = self.soft_thresholding(embeddings, self.relation_threshold)
        return pruned_embeddings
    
    def pruned_score(self, inputs, ord=1):
        if isinstance(inputs, tuple):
            heads, relations, tails = inputs
        else:
            heads, relations, tails = inputs[:, 0], inputs[:, 1], inputs[:, 2]
        head_embeds = self.pruned_entity_embedding(heads)
        relation_embeds = self.pruned_relation_embedding(relations)
        tail_embeds = self.pruned_entity_embedding(tails)
        
        score = tf.norm(head_embeds + relation_embeds - tail_embeds, axis=1, ord=ord)
        return score
    
    # -----------------------------------------------------
    
    def call(self, inputs):
        """Forward pass to compute scores for a batch of triples."""
        if self.pep is None:
            return self.kge(inputs)
        else:
            return self.pruned_score(inputs)
    
    def normalize_embeddings(self):
        """Normalize embeddings if necessary."""
        self.model.normalize_embeddings()

    def compute_loss(self, pos_scores, neg_scores):
        """Compute the margin-based ranking loss."""
        return tf.reduce_mean(tf.maximum(0.0, self.kge.margin + pos_scores - neg_scores))
    
    def save_embedding(self, path):
        path = Path(path)
        assert isinstance(self.kge, tf.keras.Model)
        if path.is_dir():
            self.kge.save_weights(path / "kge_weights.h5")
        else:
            self.kge.save_weights(path)
        
    def load_embedding(self, path):
        path = Path(path)
        if not path.exists():
            raise FileExistsError
        if path.is_dir():
            path = path / "kge_weights.h5"
        self.kge.load_weights(path)
        
    def save_threshold(self, dir_path):
        dir_path = Path(dir_path)
        np.save(dir_path/"entity_threshold", self.entity_threshold.numpy())
        np.save(dir_path/"relation_threshold", self.relation_threshold.numpy())
        
    def load_threshold(self, dir_path):
        dir_path = Path(dir_path)
        entity_threshold = np.load(dir_path/"entity_threshold", dtype=np.float32)
        relation_threshold = np.load(dir_path/"relation_threshold", dtype=np.float32)
    
        if self.pep is not None:    
            self.entity_threshold = tf.Variable(entity_threshold, trainable=True, dtype=tf.float32)
            self.relation_threshold = tf.Variable(relation_threshold, trainable=True, dtype=tf.float32)