import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, rdMolDescriptors
from rdkit.DataStructs import ExplicitBitVect
from pycaret.regression import *

# Load Harvard OPV dataset and filter based on LUMO_calc
data = pd.read_csv('https://raw.githubusercontent.com/AjStephan/havard-smile-opv/main/Non-fullerene%20small-molecules%20acceptors.csv')
opv_df = data.drop(columns=[
    'index', 'inchikey', 'LUMO_calib_stds', 'HOMO_calib_stds',
    'molW', 'PCE_calc', 'Voc_calc', 'Jsc_calc',
    'FF_calc', 'EQE_calc', 'PCE_calib', 'Voc_calib', 'Jsc_calib', 'FF_calib',
    'EQE_calib', 'PCE_cdiff', 'PCE_calib_plus'], axis=1)
opv_df['mol'] = opv_df['smiles'].apply(Chem.MolFromSmiles)

# Filter dataset for molecules with LUMO_calc <= -3.5
opv_df_filtered = opv_df[opv_df['LUMO_calc'] <= -3.5].reset_index(drop=True)

# Fingerprint generation functions
def generate_morgan_fingerprint(mol, radius=2, nBits=2048):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)

def generate_maccs166_fingerprint(mol):
    return MACCSkeys.GenMACCSKeys(mol)

def generate_atom_pair_fingerprint(mol, nBits=2048):
    fp = rdMolDescriptors.GetAtomPairFingerprint(mol)
    return convert_to_bit_vector(fp, nBits)

def generate_fcfp_fingerprint(mol, radius=2, nBits=2048):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits, useFeatures=True)

# Helper to convert fingerprints to bit vectors
def convert_to_bit_vector(fp, nBits=2048):
    bit_vector = ExplicitBitVect(nBits)
    for bit in fp.GetNonzeroElements().keys():
        bit_vector.SetBit(bit % nBits)
    return bit_vector

# Add fingerprint column
def add_fingerprint_to_df(df, fingerprint_func, fp_name):
    df[fp_name] = df['mol'].apply(fingerprint_func)
    return df

# Split fingerprint bits into individual columns
def split_fingerprint_bits(df, fp_column, prefix):
    bit_array = np.array([list(fp) for fp in df[fp_column].values])
    bit_columns = pd.DataFrame(bit_array, columns=[f'{prefix}_{i}' for i in range(bit_array.shape[1])])
    df = pd.concat([df.drop(columns=[fp_column]), bit_columns], axis=1)
    return df

# Fingerprint functions and their identifiers
fingerprint_functions = [
    (generate_morgan_fingerprint, 'morgan_fp', 'morgan'),
    (generate_maccs166_fingerprint, 'maccs_fp', 'maccs'),
    (generate_atom_pair_fingerprint, 'atom_pair_fp', 'atom_pair'),
    (generate_fcfp_fingerprint, 'fcfp_fp', 'fcfp')
]

# Process and analyze fingerprints using PyCaret
for fp_func, fp_name, prefix in fingerprint_functions:
    df = opv_df_filtered.copy()
    df = add_fingerprint_to_df(df, fp_func, fp_name)
    df = split_fingerprint_bits(df, fp_name, prefix)

    print(f'Processing {prefix} fingerprint...')

    # Prepare data for PyCaret
    X = df.iloc[:, 8:]  # Feature columns
    y = df['GAP_calc']  # Target column
    pycaret_data = pd.concat([X, y], axis=1)

    # PyCaret setup and model comparison
    exp_reg = setup(pycaret_data, target='GAP_calc', session_id=123, silent=True)
    best_model = compare_models(exclude=['catboost'], sort='MAE')
    print(f'Best model for {prefix} fingerprint: {best_model}')