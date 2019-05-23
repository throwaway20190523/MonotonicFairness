import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from aif360.datasets import german_dataset as Dataset

def load_data():
    df = Dataset.GermanDataset().convert_to_dataframe()[0]

    # Original data is 1 = Good, 2 = Bad, change that to 0 = Bad, 1 = Good
    df['credit'] = 2 - df['credit']

    # Transform some one-hot variables into ordinal variables
    df['status'] = (
        df['status=A11'] * 1 +   
        df['status=A12'] * 2 +   
        df['status=A13'] * 3 +   
        df['status=A14'] * 4 +
        0
    )
    df['credit_history'] = (
        df['credit_history=A30'] * 1 +   
        df['credit_history=A31'] * 2 +   
        df['credit_history=A32'] * 3 +   
        df['credit_history=A33'] * 4 +   
        df['credit_history=A34'] * 5 +   
        0
    )
    df['employment'] = (
        df['employment=A71'] * 1 +   
        df['employment=A72'] * 2 +   
        df['employment=A73'] * 3 +   
        df['employment=A74'] * 4 +   
        df['employment=A75'] * 5 +   
        0
    )
    df['savings'] = (
        df['savings=A61'] * 1 +   
        df['savings=A62'] * 2 +   
        df['savings=A63'] * 3 +   
        df['savings=A64'] * 4 +   
        df['savings=A65'] * 5 +   
        0
    )
    df['other_debtors'] = (
        df['other_debtors=A101'] * 1 +   
        df['other_debtors=A102'] * 2 +   
        df['other_debtors=A103'] * 3 +   
        0
    )
    df['skill_level'] = (
        df['skill_level=A171'] * 1 +   
        df['skill_level=A172'] * 2 +   
        df['skill_level=A173'] * 3 +   
        df['skill_level=A174'] * 4 +   
        0
    )


    # Outcome variable
    out_df = df[['credit']]

    # Protected variable
    out_df['young'] = 1 - df['age']
    out_df['aged'] = df['age']

    # Variables we reduced
    out_df['status'] = df['status']                 # 1
    out_df['credit_history'] = df['credit_history'] # 1
    out_df['employment'] = df['employment']         # 1
    out_df['savings'] = df['savings']               # 1
    out_df['other_debtors'] = df['other_debtors']   # 0
    out_df['skill_level'] = df['skill_level']       # 0

    # Existing monotonic variables
    out_df['investment_as_income_percentage'] = df['investment_as_income_percentage'] # -1
    out_df['month'] = df['month'] # -1
    out_df['credit_amount'] = df['credit_amount'] # -1

    # Existing numeric variables
    out_df['residence_since'] = df['residence_since']
    out_df['number_of_credits'] = df['number_of_credits']
    out_df['people_liable_for'] = df['people_liable_for']
    out_df['sex'] = df['sex']

    # Existing dummy variables
    out_df['purpose=A40'] = df['purpose=A40']
    out_df['purpose=A41'] = df['purpose=A41']
    out_df['purpose=A410'] = df['purpose=A410']
    out_df['purpose=A42'] = df['purpose=A42']
    out_df['purpose=A43'] = df['purpose=A43']
    out_df['purpose=A44'] = df['purpose=A44']
    out_df['purpose=A45'] = df['purpose=A45']
    out_df['purpose=A46'] = df['purpose=A46']
    out_df['purpose=A48'] = df['purpose=A48']
    out_df['purpose=A49'] = df['purpose=A49']
    out_df['property=A121'] = df['property=A121']
    out_df['property=A122'] = df['property=A122']
    out_df['property=A123'] = df['property=A123']
    out_df['property=A124'] = df['property=A124']
    out_df['installment_plans=A141'] = df['installment_plans=A141']
    out_df['installment_plans=A142'] = df['installment_plans=A142']
    out_df['installment_plans=A143'] = df['installment_plans=A143']
    out_df['housing=A151'] = df['housing=A151']
    out_df['housing=A152'] = df['housing=A152']
    out_df['housing=A153'] = df['housing=A153']
    out_df['telephone=A191'] = df['telephone=A191']
    out_df['telephone=A192'] = df['telephone=A192']
    out_df['foreign_worker=A201'] = df['foreign_worker=A201']
    out_df['foreign_worker=A202'] = df['foreign_worker=A202']


    out_df.sample(frac=1., random_state=78712).reset_index(drop=True)

    n_train = int(out_df.shape[0] * 0.8)
    output = {
        'data_train': out_df[:n_train].values,
        'data_test':  out_df[n_train:].values,
        'covariate_labels': [c for c in out_df.columns][3:],
        'feature_labels': ['young', 'aged'],
        'feature_groups': [1] * 2,
        'X_cols': list(range(3, out_df.shape[1])),
        'Y_col': 0,
        'A_cols': [1, 2],
        'x0s': None, #np.linspace(1.5, 4.2, 28),
        'x1s': None, #np.linspace(12, 48, 37),
        'monotonicity': [1, 1, 1, 1, 0, 0, -1, -1, -1] + [0] * 28,
    }
    
    return output
