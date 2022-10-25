import numpy as np
import pandas as pd
import argparse


# parser initialization
parser = argparse.ArgumentParser(description='Script for datasets summary')

parser.add_argument('--dataset', type=str, default='adult_income', help='Dataset: adult_income, compas, default_credit, marketing')
args = parser.parse_args()


dataset = args.dataset 


suffixes = ['full', 'fullOneHot', 'fullRules']

print('Summary for dataset: {}'.format(dataset))
for s in suffixes:
    filename_s = './{}/{}_{}.csv'.format(dataset, dataset, s)
    df_s = pd.read_csv(filename_s)
    if s == 'full':
        print('Size of the dataset: {}'.format(len(df_s)))
    
    print('Numbers of elmts for {}: {}'.format(s, len(list(df_s)) - 1))
    

