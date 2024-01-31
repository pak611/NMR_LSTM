import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import re
import torch.nn.functional as F
import argparse
import os

def build_peak_splitting_vocab(file_path):
    vocab = {}
    vocab_counter = 1  # Start counter for assigning unique integers
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split('|')
            for part in parts[1:]:
                tokens = part.split()
                if len(tokens) >= 3:
                    pattern = tokens[2]
                    if pattern not in vocab:
                        vocab[pattern] = vocab_counter
                        vocab_counter +=  1
    return vocab




def convert_to_ints(hnmr_tokens, vocab):
    converted_tokens = []
    for group in hnmr_tokens:
        converted_group = []
        for token in group:
            if token in vocab:
                # Convert the token to its corresponding integer value
                new_token = vocab[token]
                converted_group.append(new_token)
            else:
                # Assign a new integer to an unknown token
                converted_group.append(float(token))
        converted_tokens.append(converted_group)
    return converted_tokens


def parse_sequence(sequence):
    # Splitting the sequence into chemical formula and HNMR data
    formula, hnmr_data = sequence.split('1HNMR')
    formula = formula.strip()
    hnmr_data = hnmr_data.strip().split('|')

        # Parsing the chemical formula
    formula_tokens = re.findall(r'([A-Z][a-z]*)(\d*)', formula)
    # Standardize the formula tokens: (Element, Count)
    formula_tokens = [(el, int(count) if count else 1) for el, count in formula_tokens]

    # Parsing the HNMR data
    hnmr_tokens = []
    for data in hnmr_data:
        # Example Pattern: "10.7 11.2 d 5H"
        tokens = re.findall(r'(\d+\.\d+)\s+(\d+\.\d+)\s+([a-z]+)\s+(\d+)H', data.strip())
        hnmr_tokens.extend(tokens)

    return formula_tokens, hnmr_tokens



def pad_sequence(sequences, max_length, padding_token=0, sublist_length=4):
    # Assuming each sequence is a list of sublists
    padded_sequences = sequences.copy()
    current_length = len(padded_sequences)
    
    if current_length < max_length:
        # Calculate the number of padding sublists needed
        padding_sublists_needed = max_length - current_length
        # Create padding sublists each consisting of 'sublist_length' number of padding tokens
        padding_sublists = [[padding_token] * sublist_length for _ in range(padding_sublists_needed)]
        # Add these padding sublists to the original sequences
        padded_sequences.extend(padding_sublists)

    # If the sequences are longer than max_length, they are truncated
    elif current_length > max_length:
        padded_sequences = padded_sequences[:max_length]

    return padded_sequences




def preprocess_sequence(sequence, vocab):
    # Combined preprocessing function
    _ , hnmr_tokens = parse_sequence(sequence)
    converted_sequence = convert_to_ints(hnmr_tokens, vocab)
    return converted_sequence


def compound_tensor(sequences, max_length):

    # Create a compound tensor from a list of hnmr sequences
    padded_tensor_list = []
    for seq in sequences:
        padded_tensor = torch.tensor(seq)
        padded_tensor_list.append(padded_tensor)
    compound_tensor = torch.stack(padded_tensor_list)
    return compound_tensor



def main(input_path, output_path):
    # Read in the data into a list
    # build peak splitting vocabulary
    vocab = build_peak_splitting_vocab(input_path)
    with open(input_path, 'r') as f:
        data = [line.strip() for line in f]
        padded_sequences = []
        for sequence in data:
            converted_sequence = preprocess_sequence(sequence, vocab)
            padded_sequences.append(pad_sequence(converted_sequence, max_length=20))
            
        # Renamed the variable to avoid conflict with the function name
        compound_tensor_result = compound_tensor(padded_sequences, max_length=20)
        # save the tensor to a file
        torch.save(compound_tensor_result, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert HNMR data to tensors.')
    parser.add_argument('--input', '-i', default='tests/data/src-test.txt',
                        help='Input file path for HNMR data')
    parser.add_argument('--output', '-o', default='tests/data/testing_tensors.pt',
                        help='Output file path for tensors')
    args = parser.parse_args()
    current_dir = os.path.dirname(os.path.realpath(__file__))
    input_path = os.path.join(current_dir, args.input)
    output_path = os.path.join(current_dir, args.output)
    main(input_path = input_path, output_path = output_path)
