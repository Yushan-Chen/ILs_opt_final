import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from hyperopt import hp, tpe, Trials, STATUS_OK, fmin
from sklearn.metrics import mean_squared_error
import csv
import warnings
from hyperopt.early_stop import no_progress_loss

warnings.filterwarnings("ignore")

train=pd.read_csv('/home/ychen3338/project_2/data/vis_train.csv')
test=pd.read_csv('/home/ychen3338/project_2/data/vis_test.csv')

class morgan_fp:
    def __init__(self, radius, length):
        self.radius = radius
        self.length = length
    def __call__(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, self.length)
        npfp = np.array(list(fp.ToBitString())).astype('float32')
        return npfp

def conv_data(data, fp):
    data['c-fp'] = data['Cation'].apply(fp)
    x_c=np.array(list(data['c-fp']))
    data['a-fp'] = data['Anion'].apply(fp)
    x_a=np.array(list(data['a-fp']))
    x_con = data[['T', 'P']].values
    xx = np.concatenate([x_c, x_a, x_con], axis =1)
    y = data['vis'].values
    return xx, y

space = {'depth': hp.quniform('depth', 1,6,1),
         'l2_leaf_reg': hp.uniform('l2_leaf_reg', 3, 100.0),
          'learning_rate':hp.loguniform('learning_rate', np.log(0.0001), np.log(0.025)),
          'iterations':hp.quniform('iterations', 1, 1000, 1),
         'bagging_temperature':hp.uniform('bagging_temperature', 1, 200),
         'random_strength':hp.uniform('random_strength', 1, 200),
        'fp_radius':hp.randint('fp_radius', 5),
        'fp_length': hp.quniform('fp_length', 10, 4000, 1)}
def fit(params):
    fp = morgan_fp(params['fp_radius'], params['fp_length'])
    model = CatBoostRegressor(depth = params['depth'], l2_leaf_reg= params['l2_leaf_reg'], learning_rate = params['learning_rate'],
                         iterations=params['iterations'], bagging_temperature=params['bagging_temperature'],
                         random_strength=params['random_strength'],random_state=10, verbose=False)
    X, y = conv_data(train, fp)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    val_loss = []
    train_loss = []
    
    for train_index, val_index in kf.split(X):
        x_train, x_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        model.fit(x_train, y_train)
        
        y_val_pred = model.predict(x_val)
        y_train_pred = model.predict(x_train)
        
        val_loss.append(np.sqrt(mean_squared_error(y_val, y_val_pred)))
        train_loss.append(np.sqrt(mean_squared_error(y_train, y_train_pred)))
    
    return np.mean(val_loss), np.mean(train_loss)

def objective(params):
    global ITERATION
    ITERATION +=1
    for name in ['depth', 'iterations','fp_radius', 'fp_length']:
        params[name] = int(params[name])
    loss, train_loss = fit(params)
    loss =loss
    off_connection = open(out_file, 'a')
    writer = csv.writer(off_connection)
    writer.writerow([loss,train_loss, params, ITERATION])
    #pickle.dump(bayes_trial, open(dir_data + "h2_cat.p", "wb"))
    return {'loss':loss,'train_loss':train_loss, 'params': params, 'iteration':ITERATION, 'status':STATUS_OK}

import csv
out_file ='vis_MF.csv'
off_connection =open( out_file, 'w')
writer = csv.writer(off_connection)
writer.writerow(['loss','train_loss', 'params', 'iteration'])
off_connection.close()

tpe_algo = tpe.suggest
bayes_trial = Trials()

from hyperopt.early_stop import no_progress_loss
global ITERATION
ITERATION =0
best = fmin(fn = objective, space =space, algo = tpe_algo, trials = bayes_trial, 
            early_stop_fn=no_progress_loss(500),max_evals=3000) #, rstate= np.random.RandomState(50)