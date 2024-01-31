import argparse
import random
import rdkit
import sys
import time

from rdkit import Chem
from rdkit.Chem import Descriptors

def RobustSmilesMolSupplier(filename):
    with open(filename) as f:
        for line in f:
            smile, name = line.strip().split("\t")  # enforce TAB-separated
            try:
                mol = Chem.MolFromSmiles(smile)
                mol.SetProp("name", name)
                yield (name, mol)
            except Exception:
                print("ERROR: cannot parse: %s" % line, file=sys.stderr, end='')

def find_cuttable_bonds(mol, debug=False):
    res = []
    for b in mol.GetBonds():
        if not b.IsInRing():
            res.append(b)
    return res

def get_cut_bonds(frag_weight, mol):
    cuttable_bonds = find_cuttable_bonds(mol)
    cut_bonds_indexes = [b.GetIdx() for b in cuttable_bonds]
    total_weight = Descriptors.MolWt(mol)
    nb_frags = round(total_weight / frag_weight)
    max_cuts = min(len(cut_bonds_indexes), nb_frags - 1)
    random.shuffle(cut_bonds_indexes)
    return cut_bonds_indexes[0:max_cuts]

def tag_cut_bonds(frag_weight, randomize, mol):
    name = mol.GetProp("name")
    to_cut = get_cut_bonds(frag_weight, mol)
    smi = Chem.MolToSmiles(mol)
    return (smi, name, to_cut)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="fragment molecules (tag cut bonds)")
    parser.add_argument("-i", metavar="input.smi", dest="input_fn", help="molecules input file")
    parser.add_argument("-o", metavar="output.smi", dest="output_fn", help="fragments output file")
    parser.add_argument("--seed", dest="seed", default=-1, type=int, help="RNG seed")
    parser.add_argument("-n", dest="nb_passes", default=1, type=int, help="number of fragmentation passes")
    parser.add_argument("-w", dest="frag_weight", default=150.0, type=float, help="fragment weight (default=150Da)")
    
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()

    if args.seed != -1:
        random.seed(args.seed)

    with open(args.output_fn, 'w') as output:
        mol_supplier = RobustSmilesMolSupplier(args.input_fn)
        for name, mol in mol_supplier:
            for i in range(args.nb_passes):
                smi, parent_name, bond_indices = tag_cut_bonds(args.frag_weight, args.seed != -1, mol)
                output_line = f"{smi}\t{parent_name}\t{' '.join(map(str, bond_indices))}\n"
                output.write(output_line)
