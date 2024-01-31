
#%%
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit import Chem
from rdkit.Chem import rdmolops
import torch



def smiles2graphs(file_path: str
                  ) -> (list, list):

    # Specify the file path
    file_path = 'C://Users//patri//OneDrive//Desktop//DL_NMR//data//external//smiles.csv'

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)


    smiles_strings = df['Smiles'].tolist()

    mol_graphs = []
    for smile in smiles_strings:
        mol = Chem.MolFromSmiles(smile)
        mol_graph = Chem.MolToMolBlock(mol)
        mol_graphs.append(mol_graph)




    # Initialize lists to store adjacency matrices and node feature matrices
    adjacency_matrices = []
    node_features = []

    for mol_graph in mol_graphs:
        # Convert string to RDKit molecule
        mol = Chem.MolFromMolBlock(mol_graph)

        # Get adjacency matrix from RDKit molecule
        adjacency_matrix = rdmolops.GetAdjacencyMatrix(mol)

        # Convert adjacency matrix to tensor
        adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.float32)

        # Append to lists
        adjacency_matrices.append(adjacency_matrix)

        # Create node feature matrix
        node_feature_matrix = []
        for atom in mol.GetAtoms():
            # Use atomic number as feature to represent atom type
            atom_type = atom.GetAtomicNum()
            node_feature_matrix.append([atom_type])

        # Convert node feature matrix to tensor
        node_feature_matrix = torch.tensor(node_feature_matrix, dtype=torch.float32)

        # Append to lists
        node_features.append(node_feature_matrix)

    return(adjacency_matrices, node_features)


def tokenized_smiles_to_tensor(input_file, output_file, sequence_length):
    
    # Read in the tokenized SMILES from the input file
    with open(input_file, 'r') as f:
        sequences = [list(map(int, line.split())) for line in f]

    # Pad sequences to a fixed length
    sequences = [seq + [0]*(sequence_length - len(seq)) for seq in sequences]
    

    # Convert to tensor
    sequences_tensor = torch.tensor(sequences, dtype=torch.long)

    print('sequences_tensor.shape:', sequences_tensor.shape)


    # Save the tensor to a file
    torch.save(sequences_tensor, output_file)
