import tensorflow as tf
import numpy as np

class DataLoader:
    def __init__(self, data_dir, batch_size=128, method=2):
        """
        DataLoader constructor.
        Args:
            data_dir (str): Path to the directory containing the dataset.
            batch_size (int): The size of the batches to be generated.
            method (int): The method to be used for loading the data (1 or 2).
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.method = method

        # Load the id mappings
        self.entity2id = self._load_id_mapping('id2entity.txt')
        self.relation2id = self._load_id_mapping('id2relation.txt')

        # Load the triples
        self.normal_triples = self._load_triples('converted_normal_train.txt')
        self.isa_triples = self._load_triples('converted_isA_train.txt')

        # Calculate the number of entities and relations
        self.num_entities = len(self.entity2id)
        self.num_relations = len(self.relation2id)

    def _load_id_mapping(self, filename):
        """Load the id mapping from a file."""
        mapping = {}
        with open(f'{self.data_dir}/{filename}', 'r') as f:
            for line in f:
                entity, id_ = line.strip().split('\t')
                mapping[entity] = int(id_)
        return mapping

    def _load_triples(self, filename):
        """Load triples from a file."""
        return np.loadtxt(f'{self.data_dir}/{filename}', dtype=int, delimiter='\t')

    def _create_tf_dataset(self, triples, labels=None):
        """Create a TensorFlow dataset from triples and optional labels."""
        if labels is None:
            dataset = tf.data.Dataset.from_tensor_slices(triples)
        else:
            dataset = tf.data.Dataset.from_tensor_slices((triples, labels))

        dataset = dataset.shuffle(buffer_size=len(triples))
        dataset = dataset.batch(self.batch_size)
        return dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    def get_train_dataset(self):
        """Create and return the training dataset based on the method."""
        if self.method in [1, "parallel"]:
            return self._create_tf_dataset_parallel()
        elif self.method in [2, "labelling"]:
            return self._create_tf_dataset_labelling()
        else:
            return self._create_tf_dataset_normal()

    def _create_tf_dataset_parallel(self):
        """Method 1: Create datasets for normal and isA triples, adjusted for batch size."""
        normal_dataset = self._create_tf_dataset(self.normal_triples)
        isa_dataset = self._create_tf_dataset(self.isa_triples)

        # Adjust batch sizes so we have the same number of iterations for each epoch
        normal_batch_size = max(1, self.batch_size * len(self.isa_triples) // len(self.normal_triples))
        isa_batch_size = max(1, self.batch_size * len(self.normal_triples) // len(self.isa_triples))

        normal_dataset = normal_dataset.unbatch().batch(normal_batch_size)
        isa_dataset = isa_dataset.unbatch().batch(isa_batch_size)

        return normal_dataset, isa_dataset

    def _create_tf_dataset_labelling(self):
        """Method 2: Create a single dataset with labels for normal and isA triples."""
        # Create labels: 1 for isA triples, 0 for other triples
        isA_relations = {self.relation2id['type'], self.relation2id['isa']}
        labels = np.array([1 if triple[1] in isA_relations else 0 for triple in self.normal_triples], dtype=int)

        # Create the dataset with triples and labels
        dataset = self._create_tf_dataset(self.normal_triples, labels)
        return dataset
    
    def _create_tf_dataset_normal(self):
        """Method 0: No information for isA: just triples."""
        return self._create_tf_dataset(self.normal_triples)

    def get_test_dataset(self):
        """Load and return the test dataset."""
        test_triples = self._load_triples('converted_normal_test.txt')
        return self._create_tf_dataset(test_triples)


# Usage Example:
# Assuming you have the `preprocessing/preprocessing.py` setup, you can use this as follows:
# data_loader = DataLoader(data_dir='./data/yago', batch_size=128, method=1)
# train_dataset = data_loader.get_train_dataset()
# test_dataset = data_loader.get_test_dataset()