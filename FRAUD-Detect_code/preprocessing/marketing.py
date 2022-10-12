import numpy as np
import pandas as pd
from core import save


def get_marketing(save_df=False):
    # output files
    dataset = 'marketing'
    decision = 'subscribed'

    df = pd.read_csv('./raw_datasets/marketing/marketing.csv')
    
    df['age'] = df['age'].apply(lambda x: 'age:30-60' if ((x >= 30) & (x <=60))  else 'age:not30-60')
    
    if save_df:
        for rseed in range(10):
            save(df, dataset, decision, rseed)
    #return df
