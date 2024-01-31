import os
import argparse


def build_vocabulary(smiles_list):
    """
    Builds a vocabulary mapping from characters to integers.
    Args:
        smiles_list (list): A list of SMILES strings.

    Returns:
        dict: A dictionary mapping characters to integers.
    """
    unique_chars = set(char for smiles in smiles_list for char in smiles)
    return {char: idx for idx, char in enumerate(sorted(unique_chars))}


def custom_smiles_tokenizer(smiles, vocab):
    """
    Tokenizes a SMILES string and converts it to a list of integer indices based on the vocabulary.
    Args:
        smiles (str): A SMILES string.
        vocab (dict): A dictionary mapping characters to integers.

    Returns:
        list: A list of integer indices representing the tokens in the SMILES string.
    """
    return [vocab[char] for char in smiles if char in vocab]

def tokenize_and_convert_smiles(input_file, output_file):
    # read in SMILES data from the input file
    with open(input_file, 'r') as f:
        smiles_data = [line.strip() for line in f]

    # build vocabulary
    vocab = build_vocabulary(smiles_data)

    # tokenize each SMILES string and convert to indices
    tokenized_and_converted = [custom_smiles_tokenizer(smiles, vocab) for smiles in smiles_data]
    
    # write tokenized and converted SMILES to the output file
    with open(output_file, 'w') as f:
        for tokenized in tokenized_and_converted:
            f.write(' '.join(map(str, tokenized)) + '\n')

    return vocab

def main(input_file, output_file):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    vocab = tokenize_and_convert_smiles(input_file, output_file)
    # Save vocabulary for later use
    vocab_file = os.path.join(os.path.dirname(input_file), 'vocab.txt')
    with open(current_dir + '/tests/data/vocab.txt', 'w') as f:
        for char, idx in vocab.items():
            f.write(f'{char} {idx}\n')
    print('vocab.txt saved to UTILS/preprocess/tests/data/vocab.txt')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tokenize and convert SMILES strings.')
    parser.add_argument('--input', '-i', default='tests/data/tgt-test.txt',
                        help='Input file path for SMILES strings')
    parser.add_argument('--output', '-o', default='tests/data/tokenized_tgt-test.txt',
                        help='Output file path for tokenized SMILES strings')

    args = parser.parse_args()
    current_dir = os.path.dirname(os.path.realpath(__file__))
    input_path = os.path.join(current_dir, args.input)
    output_path = os.path.join(current_dir, args.output)
    main(input_path, output_path)




