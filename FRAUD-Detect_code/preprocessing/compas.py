import numpy as np
import pandas as pd
from core import save
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def get_compas(save_df=False):

    # output files
    dataset = 'compas'
    decision = 'two_year_recid'

    

    """Loads COMPAS dataset from https://raw.githubusercontent.com/algofairness/fairness-comparison/master/fairness/data/raw/propublica-recidivism.csv
    :param: save_intermediate: save the transformed dataset. Do not save by default.
    """
    url = "https://raw.githubusercontent.com/algofairness/fairness-comparison/master/fairness/data/raw/propublica-recidivism.csv"

    df = pd.read_csv(url)

    df = df[['sex', 'age', 'race', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count', 'c_charge_degree', 'two_year_recid']]

    df.to_csv('./raw_datasets/compas/compas.csv', index=False)

    print(len(df))

    df = df[(df['race']=='African-American') | (df['race']=='Caucasian')]

    print(len(df))

    df = df.replace({'c_charge_degree': {'M': 'Misdemeanor', 'F': 'Felony'}})

    if save_df:
       for rseed in range(10):
            save(df, dataset, decision, rseed)
    #return df