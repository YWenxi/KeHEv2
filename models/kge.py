import tensorflow as tf
from numpy import sqrt


class BaseKGEModel(tf.keras.Model):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(BaseKGEModel, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        
        # Initialize embeddings
        uniform_range = 6 / sqrt(self.embedding_dim)
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
    def __init__(self, model_name, num_entities, num_relations, embedding_dim, margin=1.0):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        if model_name == 'TransE':
            self.kge = TransE(num_entities, num_relations, embedding_dim, margin)
        elif model_name == 'RotatE':
            self.kge = RotatE(num_entities, num_relations, embedding_dim, margin)
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
    
    def call(self, inputs):
        """Forward pass to compute scores for a batch of triples."""
        return self.kge(inputs)
    
    def normalize_embeddings(self):
        """Normalize embeddings if necessary."""
        self.model.normalize_embeddings()

    def compute_loss(self, pos_scores, neg_scores):
        """Compute the margin-based ranking loss."""
        return tf.reduce_mean(tf.maximum(0.0, self.kge.margin + pos_scores - neg_scores))