"""Build/reuse feature caches with the project's original feature encoders."""

import argparse
from pathlib import Path


def prepare(dataset):
    import joblib
    from utils.protein_init import protein_init
    from utils.ligand_init import ligand_init

    data_path = Path('DataSets') / f'{dataset}.txt'
    if not data_path.is_file():
        raise FileNotFoundError(f'{data_path}: see DataSets/README.md')
    rows = data_path.read_text(encoding='utf-8').strip().split('\n')
    proteins = list(set(row.split(' ')[-2] for row in rows))
    smiles = list(set(row.split(' ')[-3] for row in rows))
    cache = Path('DataSets/Preprocessed')
    cache.mkdir(parents=True, exist_ok=True)
    protein_path = cache / f'{dataset}-protein-new.pkl'
    ligand_path = cache / f'{dataset}-ligand-hi-new.pkl'
    print(f'{dataset}: {len(rows)} pairs, {len(smiles)} SMILES, {len(proteins)} sequences')

    # Only load caches you created or obtained from a trusted source.
    protein_dict = joblib.load(protein_path) if protein_path.exists() else {}
    missing = [seq for seq in proteins if seq not in protein_dict]
    if missing:
        protein_dict.update(protein_init(missing))
        joblib.dump(protein_dict, protein_path)
    print(f'Protein cache: {protein_path}')

    ligand_dict = joblib.load(ligand_path) if ligand_path.exists() else {}
    missing = [smi for smi in smiles if smi not in ligand_dict]
    if missing:
        ligand_dict.update(ligand_init(missing, mode='BRICS'))
        joblib.dump(ligand_dict, ligand_path)
    invalid = [smi for smi in smiles if ligand_dict.get(smi) is None]
    if invalid:
        raise ValueError(f'{len(invalid)} SMILES could not be encoded. Do not silently drop benchmark rows.')
    print(f'Ligand cache: {ligand_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=['DrugBank', 'BIOSNAP', 'Davis', 'BD2D'])
    args = parser.parse_args()
    prepare(args.dataset)
