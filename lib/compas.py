import numpy as np
import pandas as pd

def load_data(path = 'data/compas_prepped.json',
              binarize = True, # Deprecated and ignored
             ):

    data = pd.read_csv('../data/compas-scores-two-years.csv')
    
    # Data cleaning as performed by propublica
    data = data[data['days_b_screening_arrest'] <= 30]
    data = data[data['days_b_screening_arrest'] >= -30]
    data = data[data['is_recid'] != -1]
    data = data[data['c_charge_degree'] <= "O"]
    data = data[data['score_text'] != 'N/A']
    
    n = data.shape[0]
    n_train = int(n * .8)
    n_test = n - n_train
    data = np.array(pd.concat([
        data[[
            'priors_count',
            'age',
            'juv_fel_count',
            'juv_misd_count',
            'juv_other_count',
        ]],
        pd.DataFrame({
            'white': data['race'].isin(['Caucasian']),
            'black': data['race'].isin(['African-American']),
            'other': ~data['race'].isin(['Caucasian', 'African-American']),
            'male': data['sex'].isin(['Male']),
            'female': data['sex'].isin(['Female']),
        }).reindex(['white', 'black', 'other', 'male', 'female'], axis=1),
        data[['two_year_recid']],
    ], axis = 1).values)+0
    
    
    # Shuffle for train/test splitting
    np.random.seed(seed=78712)
    data = np.random.shuffle(data)
    
    data_train = data[:n_train, :]
    data_test = data[n_train:, :]
    
    output = {
        'data_train': data_train,
        'data_test': data_test,
        'covariate_labels': [
            'age',
            'priors_count',
            'juv_fel_count',
            'juv_misd_count',
            'juv_other_count',
        ],
        'feature_labels': ['white', 'black', 'other', 'male', 'female'],
        'feature_groups': [0] * 3 + [1] * 2,
        'X_cols': list(range(0, 5)),
        'A_cols': list(range(5, 10)),
        'Y_col': 10,
        'x0s': np.linspace(18, 50, 33),
        'x1s': np.linspace(0, 20, 21),
    }
    
    # Remove an outlier
    output['data_train'] = output['data_train'][output['data_train'][:, 0] > 0.] 
    
    return output


def load_data_black_white(
    path = '../data/compas_prepped.json',
    binarize = True, # Deprecated and ignored
):

    data = pd.read_csv('../data/compas-scores-two-years.csv')
    
    # Data cleaning as performed by propublica
    data = data[data['days_b_screening_arrest'] <= 30]
    data = data[data['days_b_screening_arrest'] >= -30]
    data = data[data['is_recid'] != -1]
    data = data[data['c_charge_degree'] <= "O"]
    data = data[data['score_text'] != 'N/A']
    data = pd.concat([
        data[[
            'priors_count',
            'age',
            'juv_fel_count',
            'juv_misd_count',
            'juv_other_count',
        ]],
        pd.DataFrame({
            'white': data['race'].isin(['Caucasian']),
            'black': data['race'].isin(['African-American']),
        }).reindex(['white', 'black'], axis=1),
        data[['two_year_recid']],
    ], axis = 1)
    
    data = data[data['white'] | data['black']]
    
    data = data.values + 0.
    n = data.shape[0]
    n_train = int(n * .8)
    n_test = n - n_train
    
    # Shuffle for train/test splitting
    data = np.random.shuffle(data)
    data_train = data[:n_train, :]
    data_test = data[n_train:, :]
    
    output = {
        'data_train': data_train,
        'data_test': data_test,
        'covariate_labels': [
            'age',
            'priors_count',
            'juv_fel_count',
            'juv_misd_count',
            'juv_other_count',
        ],
        'feature_labels': ['white', 'black'],
        'feature_groups': [0] * 3 + [1] * 2,
        'X_cols': list(range(0, 5)),
        'A_cols': list(range(5, 7)),
        'Y_col': 7,
        'x0s': np.linspace(18, 50, 33),
        'x1s': np.linspace(0, 20, 21),
    }
    
    # Remove an outlier
    output['data_train'] = output['data_train'][output['data_train'][:, 0] > 0.] 
    
    return output
