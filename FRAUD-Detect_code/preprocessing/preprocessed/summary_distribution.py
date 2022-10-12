import numpy as np
import pandas as pd
import argparse


# parser initialization
parser = argparse.ArgumentParser(description='Script for datasets summary')

parser.add_argument('--dataset', type=str, default='adult_income', help='Dataset: adult_income, compas, default_credit, marketing')
args = parser.parse_args()

dataset = args.dataset 

data_map = {
    'adult_income'      : ('gender_Female', 'gender_Male'),
    'compas'            : ('race_African-American', 'race_Caucasian'),
    'default_credit'    : ('SEX_Female', 'SEX_Male'),
    'marketing'         : ('age_age:not30-60', 'age_age:30-60')
}


suffixes = ['full', 'fullOneHot', 'fullRules']


filename_all = './{}/{}_fullOneHot.csv'.format(dataset, dataset)
df_all = pd.read_csv(filename_all)
minority_all = df_all[data_map['{}'.format(dataset)][0]]
majority_all = df_all[data_map['{}'.format(dataset)][1]]


# suing group
sg_results_min = []
sg_results_maj = []

for rseed in range(10):
    filename_sg = './{}/{}_attackOneHot_{}.csv'.format(dataset, dataset, rseed)
    df_sg = pd.read_csv(filename_sg)
    minority_sg = df_sg[data_map['{}'.format(dataset)][0]]
    sg_results_min.append(np.mean(minority_sg))
    majority_sg = df_sg[data_map['{}'.format(dataset)][1]]
    sg_results_maj.append(np.mean(majority_sg))

# test set
test_results_min = []
test_results_maj = []

for rseed in range(10):
    filename_sg = './{}/{}_testOneHot_{}.csv'.format(dataset, dataset, rseed)
    df_sg = pd.read_csv(filename_sg)
    minority_test = df_sg[data_map['{}'.format(dataset)][0]]
    test_results_min.append(np.mean(minority_test))
    majority_test = df_sg[data_map['{}'.format(dataset)][1]]
    test_results_maj.append(np.mean(majority_test))


print("----"*10)

print("dataset-wide distribution")
print('minority  : {} and majority: {}'.format(np.round(np.mean(minority_all), 2), np.round(np.mean(majority_all), 2)))

print("----"*10)

print("suing-group distribution")
print('minority : {} and majority: {}'.format(np.round(np.mean(sg_results_min), 2), np.round(np.mean(sg_results_maj), 2)))

print("----"*10)

print("test set distribution")
print('minority : {} and majority: {}'.format(np.round(np.mean(test_results_min), 2), np.round(np.mean(test_results_maj), 2)))

print("----"*10)
