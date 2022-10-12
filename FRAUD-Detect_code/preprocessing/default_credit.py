import numpy as np
import pandas as pd
from core import save

def get_default_credit(save_df=False):
    # output files
    dataset = 'default_credit'
    decision = 'DEFAULT_PAYEMENT'

    df = pd.read_csv('./raw_datasets/default_credit/default_credit.csv')
    df = df.dropna()
    missing_fractions = df.isnull().mean().sort_values(ascending=False) # Fraction of data missing for each variable
    print(missing_fractions[missing_fractions > 0]) # Print variables that are missing data


    if save_df:
        for rseed in range(10):
            save(df, dataset, decision, rseed)
    #return df