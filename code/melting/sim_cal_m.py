import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from multiprocessing import Pool
from tqdm import tqdm
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

SHAP_PATH = '/home/ychen3338/project_2/code/melting/shap_feature_m.csv'
TRAIN_PATH = '/home/ychen3338/project_2/data/m_train.csv'
IL_SMILES_PATH = '/home/ychen3338/project_2/data/IL_smiles.csv'
OUTPUT_PATH = '/home/ychen3338/project_2/code/melting/smiles_m.csv'
N_JOBS = 24

shap_df = pd.read_csv(SHAP_PATH)
shap_values = shap_df['Mean_SHAP_Value'].values
nonzero_bits = ((~np.isnan(shap_values)) & (shap_values != 0)).astype(int)
negative_bits = ((~np.isnan(shap_values)) & (shap_values < 0)).astype(int)
n_bits = len(nonzero_bits)

nonzero_fp = DataStructs.ExplicitBitVect(n_bits)
negative_fp = DataStructs.ExplicitBitVect(n_bits)
for i in range(n_bits):
    if nonzero_bits[i]:
        nonzero_fp.SetBit(i)
    if negative_bits[i]:
        negative_fp.SetBit(i)

train_df = pd.read_csv(TRAIN_PATH)
train_smiles = train_df['smiles'].tolist()
train_tms = train_df['Tm'].tolist()

dice_scores = []
for smi in train_smiles:
    mol = Chem.MolFromSmiles(smi)
    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=4, nBits=n_bits)
        dice = DataStructs.DiceSimilarity(fp, nonzero_fp)
        dice_scores.append(dice)

p25_dice = np.percentile(dice_scores, 25)
print(f"Step 1: 25th percentile of Dice similarity = {p25_dice:.3f}")

tanimoto_scores_tm_low = []
for smi, tm in zip(train_smiles, train_tms):
    if tm >= 298.15:
        continue
    mol = Chem.MolFromSmiles(smi)
    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=4, nBits=n_bits)
        tanimoto = DataStructs.TanimotoSimilarity(fp, negative_fp)
        tanimoto_scores_tm_low.append(tanimoto)

pXX = np.percentile(tanimoto_scores_tm_low, 25)
print(f"Step 2: Tanimoto similarity of samples = {pXX:.3f}")

def filter_smile(smile):
    try:
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=4, nBits=n_bits)
        dice = DataStructs.DiceSimilarity(fp, nonzero_fp)
        tanimoto = DataStructs.TanimotoSimilarity(fp, negative_fp)
        if dice > p25_dice and tanimoto > pXX:
            return (smile, dice, tanimoto)
    except:
        return None

il_df = pd.read_csv(IL_SMILES_PATH)
smiles_list = il_df['IL_smile'].tolist()
print(f"Filtering {len(smiles_list)} ILs using {N_JOBS} cores...")

with Pool(N_JOBS) as pool:
    results = list(tqdm(pool.imap(filter_smile, smiles_list), total=len(smiles_list)))

results = [r for r in results if r is not None]
filtered_df = pd.DataFrame(results, columns=['IL_smile', 'Dice_Coefficient', 'Tanimoto_Coefficient'])
filtered_df.to_csv(OUTPUT_PATH, index=False)
