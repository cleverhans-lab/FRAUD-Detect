from __future__ import print_function

from functools import partial

# utils
import pickle
import argparse
import os
import pandas as pd 
import numpy as np 
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import load_model

from collections import Counter




data_dict = {
    'adult_income'      : ('adult_income', 'income'),
    'compas'            : ('compas', 'two_year_recid'),
    'default_credit'    : ('default_credit', 'DEFAULT_PAYEMENT'),
    'marketing'         : ('marketing', 'subscribed')      
}


def get_labels(data, rseed):

    dataset, decision = data_dict[data]
    datadir = '../preprocessing/preprocessed/{}/'.format(dataset)  

    labeldir = './true_labels/{}'.format(dataset)
    
    if not os.path.exists(labeldir):
        os.mkdir(labeldir)  

    #filenames
    suffix = 'OneHot'
    test_file       = '{}{}_test{}_{}.csv'.format(datadir, dataset, suffix, rseed)
    sg_file         = '{}{}_attack{}_{}.csv'.format(datadir, dataset, suffix, rseed)

    # load dataframe
    df_test     = pd.read_csv(test_file)
    df_sg       = pd.read_csv(sg_file)

    ## test set
    y_test = df_test[decision].astype('int32')

    ## sg set
    y_sg = df_sg[decision].astype('int32')


    label_name_sg   = '{}/label_sg_{}.csv'.format(labeldir, rseed)
    label_name_test = '{}/label_test_{}.csv'.format(labeldir, rseed)

    df_sg = pd.DataFrame()
    df_sg['prediction'] = y_sg
    df_sg.to_csv(label_name_sg, index=False)

    df_test = pd.DataFrame()
    df_test['prediction'] = y_test
    df_test.to_csv(label_name_test, index=False)
    

if __name__ == '__main__':
    # parser initialization
    parser = argparse.ArgumentParser(description='Script pretraining bbox models')
    parser.add_argument('--dataset', type=str, default='adult_income', help='adult_income, compas, default_credit, marketing')
    parser.add_argument('--rseed', type=int, default=0, help='random seed: choose between 0 - 9')
    
    args = parser.parse_args()
    dataset = args.dataset
    rseed = args.rseed
    
    get_labels(dataset, rseed)
        


