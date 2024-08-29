import os
import numpy as np
import argparse

def load_triples(file_path):
    """Load triples from a text file."""
    return np.loadtxt(file_path, dtype=str, delimiter='\t')

def create_mappings(triples):
    """Create entity and relation mappings."""
    entities = np.unique(np.concatenate([triples[:, 0], triples[:, 2]]))
    relations = np.unique(triples[:, 1])

    entity2id = {entity: i for i, entity in enumerate(entities)}
    relation2id = {relation: i for i, relation in enumerate(relations)}

    return entity2id, relation2id

def convert_triples_to_id(triples, entity2id, relation2id):
    """Convert triples to ID format."""
    return np.array([
        [entity2id[h], relation2id[r], entity2id[t]]
        for h, r, t in triples
    ], dtype=int)

def save_mappings(mapping, file_path):
    """Save the entity or relation mappings to a file."""
    with open(file_path, 'w') as f:
        for key, value in mapping.items():
            f.write(f'{key}\t{value}\n')

def save_triples(triples, file_path):
    """Save the converted triples to a file."""
    np.savetxt(file_path, triples, fmt='%d', delimiter='\t')

def preprocess_id_mappings(mini_files, output_dir):
    """Preprocess to create id2entity.txt and id2relation.txt mappings."""
    all_triples = []
    for file in mini_files:
        triples = load_triples(file)
        all_triples.append(triples)
    
    all_triples = np.concatenate(all_triples, axis=0)
    entity2id, relation2id = create_mappings(all_triples)

    save_mappings(entity2id, os.path.join(output_dir, 'id2entity.txt'))
    save_mappings(relation2id, os.path.join(output_dir, 'id2relation.txt'))

    return entity2id, relation2id

def preprocess_combined_datasets(train_files, test_files, entity2id, relation2id, output_dir):
    """Combine and convert train/test datasets."""
    for split, files in zip(['train', 'test'], [train_files, test_files]):
        combined_triples = []
        for file in files:
            triples = load_triples(file)
            converted_triples = convert_triples_to_id(triples, entity2id, relation2id)
            combined_triples.append(converted_triples)
        
        combined_triples = np.concatenate(combined_triples, axis=0)
        save_triples(combined_triples, os.path.join(output_dir, f'converted_normal_{split}.txt'))

def preprocess_isA_datasets(train_files, test_files, entity2id, relation2id, output_dir, filters=["type", "isa"]):
    """Create and save isA (type/isa) datasets."""
    for split, files in zip(['train', 'test'], [train_files, test_files]):
        isA_triples = []
        for file in files:
            triples = load_triples(file)
            isA_triples.append(triples)
        
        isA_triples = np.concatenate(isA_triples, axis=0)
        if filters is not None:
            isA_triples = isA_triples[np.isin(isA_triples[:, 1], filters)]
        isA_triples = convert_triples_to_id(isA_triples, entity2id, relation2id)
        save_triples(isA_triples, os.path.join(output_dir, f'converted_isA_{split}.txt'))

def main(dataset_dir):
    # Define file paths based on dataset directory
    mini_files = [
        os.path.join(dataset_dir, 'insnet_mini.txt'),
        os.path.join(dataset_dir, 'InsType_mini.txt'),
        os.path.join(dataset_dir, 'ontonet.txt')
    ]
    
    train_files = [
        os.path.join(dataset_dir, 'insnet_train.txt'),
        os.path.join(dataset_dir, 'InsType_train.txt'),
        os.path.join(dataset_dir, 'ontonet_train.txt')
    ]
    
    test_files = [
        os.path.join(dataset_dir, 'insnet_test.txt'),
        os.path.join(dataset_dir, 'InsType_test.txt'),
        os.path.join(dataset_dir, 'ontonet_test.txt')
    ]
    
    output_dir = dataset_dir
    
    # Step 1: Create id2entity.txt and id2relation.txt
    entity2id, relation2id = preprocess_id_mappings(mini_files, output_dir)

    # Step 2: Combine and convert train/test datasets
    preprocess_combined_datasets(train_files, test_files, entity2id, relation2id, output_dir)

    # Step 3: Create and save isA (type/isa) datasets
    preprocess_isA_datasets(
        train_files=[os.path.join(dataset_dir, 'InsType_train.txt'), os.path.join(dataset_dir, 'ontonet_train.txt')],
        test_files=[os.path.join(dataset_dir, 'InsType_test.txt'), os.path.join(dataset_dir, 'ontonet_test.txt')],
        entity2id=entity2id,
        relation2id=relation2id,
        output_dir=output_dir
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess YAGO or DBpedia datasets.')
    parser.add_argument('--dataset', type=str, default="data/yago",
                        help='Path to the dataset directory (e.g., data/yago or data/dp)')
    args = parser.parse_args()

    main(args.dataset)