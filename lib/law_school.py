import numpy as np


def load_data(
    train_path = 'data/law_school_cf_train.csv',
    test_path = 'data/law_school_cf_test.csv',
    binarize = False,
    ):
    
    # UGPA	LSAT	ZFYA	amerind	asian	black	hisp	mexican	other	puerto	white	female	male
    data_train = np.genfromtxt(train_path, delimiter=',', skip_header=True)
    data_test = np.genfromtxt(test_path, delimiter=',', skip_header=True)
    
    # Flip majority/minority for consistency
    data_train = np.concatenate([data_train[:, :11], np.flip(data_train[:, -2:], axis=1)], axis=1)
    data_test  = np.concatenate([ data_test[:, :11], np.flip( data_test[:, -2:], axis=1)], axis=1)
    
    if binarize:
        threshold = np.percentile(data_train[:, 2], 50)
        data_train[:, 2] = data_train[:, 2] >= threshold
        data_test[:, 2] = data_test[:, 2] >= threshold
    
    output = {
        'data_train': data_train,
        'data_test': data_test,
        'covariate_labels': ['UGPA', 'LSAT'],
        'feature_labels': ['amerind', 'asian', 'black', 'hisp', 'mexican', 'other', 'puerto', 'white', 'male', 'female'],
        'feature_groups': [0] * 8 + [1] * 2,
        'X_cols': list(range(0, 2)),
        'Y_col': 2,
        'A_cols': list(range(3, 13)),
        'x0s': np.linspace(1.5, 4.2, 28),
        'x1s': np.linspace(12, 48, 37),
    }
    
    # Remove an outlier
    output['data_train'] = output['data_train'][output['data_train'][:, 0] > 0.] 
    
    return output
