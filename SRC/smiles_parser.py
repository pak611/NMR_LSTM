import pandas as pd

def parse_smiles(file_path):

    # Read the file
    with open(file_path, 'r') as file:
        lines = file.readlines()


    # Parse each line into (smiles, CHEMBLID, cut_bonds)
        data = []
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts)>= 3:
                smiles = parts[0]
                chembl_id = parts[1]
                cut_bonds = [int(bond) for bond in parts[2].replace(' ', ',').split(',')]
                data.append((smiles, chembl_id, cut_bonds))

    df = pd.DataFrame(data, columns=['smiles', 'chembl_id', 'cut_bonds'])
    return df