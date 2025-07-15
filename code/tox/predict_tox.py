import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import joblib
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

INPUT_FILE = '/home/ychen3338/project_2/code/tox/simles_tox.csv'
MODEL_FILE = '/home/ychen3338/project_2/code/tox/best_lgbm_model.pkl'
OUTPUT_FILE = '/home/ychen3338/project_2/data/after_m_tox.csv'
FP_RADIUS = 2
FP_LENGTH = 1097
THRESHOLD_K = 1
N_JOBS = 24
BATCH_SIZE = 100_000  

print("Loading model...")
best_model = joblib.load(MODEL_FILE)

print(f"Reading IL data from {INPUT_FILE}...")
df = pd.read_csv(INPUT_FILE)
df[['Cation', 'Anion']] = df['IL_smile'].str.split('.', n=1, expand=True)

def get_fingerprint(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(FP_LENGTH, dtype='float32')
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=FP_RADIUS, nBits=FP_LENGTH)
        return np.array(list(fp.ToBitString()), dtype='float32')
    except:
        return np.zeros(FP_LENGTH, dtype='float32')

def parallel_fingerprint(smiles_list):
    with Pool(processes=N_JOBS) as pool:
        fps = list(tqdm(pool.imap(get_fingerprint, smiles_list), total=len(smiles_list), desc="Generating fingerprints"))
    return np.array(fps)

all_results = []

num_batches = int(np.ceil(len(df) / BATCH_SIZE))

for i in range(num_batches):
    print(f"\n--- Processing batch {i+1}/{num_batches} ---")
    start = i * BATCH_SIZE
    end = min((i + 1) * BATCH_SIZE, len(df))
    batch_df = df.iloc[start:end].copy()

    cation_list = batch_df['Cation'].tolist()
    anion_list = batch_df['Anion'].tolist()

    x_c = parallel_fingerprint(cation_list)
    x_a = parallel_fingerprint(anion_list)
    X_pred = np.concatenate([x_c, x_a], axis=1)

    batch_df['tox_pre'] = best_model.predict(X_pred)
    batch_filtered = batch_df[batch_df['tox_pre'] > THRESHOLD_K][['IL_smile', 'm_pre', 'tox_pre']]
    all_results.append(batch_filtered)

df_filtered = pd.concat(all_results, ignore_index=True)
df_filtered.to_csv(OUTPUT_FILE, index=False)
