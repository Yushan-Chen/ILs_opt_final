import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import joblib
from multiprocessing import Pool, cpu_count
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

class morgan_fp:
    def __init__(self, radius, length):
        self.radius = radius
        self.length = length
    def __call__(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(self.length, dtype='float32') 
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, self.length)
        npfp = np.array(list(fp.ToBitString())).astype('float32')
        return npfp

def parallel_fp(smiles_list, fp_fn, n_jobs=None):
    if n_jobs is None:
        n_jobs = max(cpu_count() - 1, 1)
    with Pool(n_jobs) as pool:
        fps = pool.map(fp_fn, smiles_list)
    return np.array(fps)

def conv_data_parallel(data, fp_fn, n_jobs=None):
    c_fp = parallel_fp(data['Cation'].tolist(), fp_fn, n_jobs=n_jobs)
    a_fp = parallel_fp(data['Anion'].tolist(), fp_fn, n_jobs=n_jobs)
    return np.concatenate([c_fp, a_fp], axis=1)

params = {'fp_radius': 4, 'fp_length': 1199}  
fp_fn = morgan_fp(params['fp_radius'], params['fp_length'])

best_model = joblib.load('/home/ychen3338/project_2/code/vis/best_lgbm_model.pkl')

df = pd.read_csv('/home/ychen3338/project_2/code/vis/simles_vis.csv')
df[['Cation', 'Anion']] = df['IL_smile'].str.split('.', n=1, expand=True)

X_pred = conv_data_parallel(df, fp_fn, n_jobs=24)

T = np.full((X_pred.shape[0], 1), 298.15, dtype='float32')
P = np.full((X_pred.shape[0], 1), 101.325, dtype='float32')

X_pred = np.concatenate([X_pred, T, P], axis=1)

df['vis_pre'] = best_model.predict(X_pred)

df_filtered = df[df['vis_pre'] < np.log(10)]
output_cols = ['IL_smile', 'Cation', 'Anion', 'm_pre', 'tox_pre','vis_pre']
df_filtered[output_cols].to_csv('/home/ychen3338/project_2/data/after_m_tox_vis.csv', index=False)