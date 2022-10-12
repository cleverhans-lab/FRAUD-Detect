import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from collections import Counter
from mlxtend.frequent_patterns import fpgrowth
from config import bin_dict, bin_dict_5, num_cols_dict, min_support_dict, sensitive_attr_dict, dropList_dict


import os
import random

def discretized(df, num_cols):

    df_num = df[num_cols]
    df_other = df[list(set(df.columns) - set(num_cols))]

    binner = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    
    
    df_num_transformed = binner.fit_transform(df_num)

    
    df_num_transformed = pd.DataFrame(df_num_transformed, columns=num_cols)

    

    for col in num_cols:
        df_num_transformed = df_num_transformed.replace({col:bin_dict_5})
    

    df_num_transformed = pd.get_dummies(df_num_transformed)

    df_num_transformed.reset_index(drop=True, inplace=True)
    df_other.reset_index(drop=True, inplace=True)

    df_final = pd.concat([df_num_transformed, df_other], axis=1)
    
    return df_final


def create_rules(df, dataset_name):

    df_sens = df[sensitive_attr_dict[dataset_name]]

    df_pos = df.drop(labels=dropList_dict[dataset_name], axis=1)

    #add neg cols
    cols = list(df_pos)
    df_neg = pd.DataFrame()
    

    for col in cols:
        df_neg['not_{}'.format(col)] = 1 - df_pos[col]

    
    print('ones rules -->>>>>>>>', len(list(df_pos)) + len(list(df_neg)))


    ll = fpgrowth(df_pos, min_support=min_support_dict[dataset_name], max_len=2, use_colnames=True)


    rules = [list(x) for x in ll['itemsets']]
    

    df_rules = pd.DataFrame()


    print('mined rules -->>>>>>>>', len(rules))
    

    for rule in rules:
        if (len(rule)==1):
            #key = rule[0]
            #df_rules[key] = dataset[key]
            pass

        else:
            key1 = rule[0]
            key2 = rule[1]

            key = key1 + '__AND__' + key2
            df_rules[key] = np.logical_and(df_pos[key1], df_pos[key2]).astype(int)
        

    df_all = pd.concat([df_sens, df_pos, df_neg, df_rules], axis=1)


    print('all rules -->>>>>>>>', len(list(df_all)))

    return df_all


def save(df, dataset, decision, rseed):

    df = shuffle(df, random_state=99)

    outdir = './preprocessed/{}/'.format(dataset)

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    #plain
    full_name    = '{}{}_full.csv'.format(outdir, dataset)

    train_name   = '{}{}_train_{}.csv'.format(outdir, dataset, rseed)
    test_name    = '{}{}_test_{}.csv'.format(outdir, dataset, rseed)
    attack_name  = '{}{}_attack_{}.csv'.format(outdir, dataset, rseed)
    

    #onehot
    oneHot_full_name    = '{}{}_fullOneHot.csv'.format(outdir, dataset)

    oneHot_train_name   = '{}{}_trainOneHot_{}.csv'.format(outdir, dataset, rseed)
    oneHot_test_name    = '{}{}_testOneHot_{}.csv'.format(outdir, dataset, rseed)
    oneHot_attack_name  = '{}{}_attackOneHot_{}.csv'.format(outdir, dataset, rseed)

    #discretized
    discretized_full_name    = '{}{}_fullDiscretized.csv'.format(outdir, dataset)

    discretized_train_name   = '{}{}_trainDiscretized_{}.csv'.format(outdir, dataset, rseed)
    discretized_test_name    = '{}{}_testDiscretized_{}.csv'.format(outdir, dataset, rseed)
    discretized_attack_name  = '{}{}_attackDiscretized_{}.csv'.format(outdir, dataset, rseed)

    #rule
    rules_full_name    = '{}{}_fullRules.csv'.format(outdir, dataset)

    rules_train_name   = '{}{}_trainRules_{}.csv'.format(outdir, dataset, rseed)
    rules_test_name    = '{}{}_testRules_{}.csv'.format(outdir, dataset, rseed)
    rules_attack_name  = '{}{}_attackRules_{}.csv'.format(outdir, dataset, rseed)


    # one_hot_df
    df_onehot = pd.get_dummies(df)

    # discretized_df
    df_discretized = discretized(df_onehot, num_cols_dict[dataset])

    # rules_df
    df_rules = create_rules(df_discretized, dataset)

    # plain data
    df_train, df_holdout, indices_train, indices_holdout = train_test_split(df, range(len(df)), test_size=0.33, random_state=rseed, stratify=df[decision])
    df_test, df_attack, indices_test, indices_attack = train_test_split(df_holdout, range(len(df_holdout)), test_size=0.5, random_state=rseed, stratify=df_holdout[decision])

    # onehot data
    df_onehot_train = df_onehot.iloc[indices_train,:]
    df_onehot_holdout  = df_onehot.iloc[indices_holdout,:]
    df_onehot_test, df_onehot_attack = df_onehot_holdout.iloc[indices_test,:], df_onehot_holdout.iloc[indices_attack,:]

    # discretized data
    df_discretized_train = df_discretized.iloc[indices_train,:]
    df_discretized_holdout  = df_discretized.iloc[indices_holdout,:]
    df_discretized_test, df_discretized_attack = df_discretized_holdout.iloc[indices_test,:], df_discretized_holdout.iloc[indices_attack,:]

    # rules data
    df_rules_train = df_rules.iloc[indices_train,:]
    df_rules_holdout  = df_rules.iloc[indices_holdout,:]
    df_rules_test, df_rules_attack = df_rules_holdout.iloc[indices_test,:], df_rules_holdout.iloc[indices_attack,:]

    #save the full dataset
    df.to_csv(full_name, index=False)
    df_onehot.to_csv(oneHot_full_name, index=False)
    df_discretized.to_csv(discretized_full_name, index=False)
    df_rules.to_csv(rules_full_name, index=False)


    #save train set 
    df_train.to_csv(train_name, index=False)
    df_onehot_train.to_csv(oneHot_train_name, index=False)
    df_discretized_train.to_csv(discretized_train_name, index=False)
    df_rules_train.to_csv(rules_train_name, index=False)


    #save test set
    df_test.to_csv(test_name, index=False)
    df_onehot_test.to_csv(oneHot_test_name, index=False)
    df_discretized_test.to_csv(discretized_test_name, index=False)
    df_rules_test.to_csv(rules_test_name, index=False)

    #save attack set
    df_attack.to_csv(attack_name, index=False)
    df_onehot_attack.to_csv(oneHot_attack_name, index=False)
    df_discretized_attack.to_csv(discretized_attack_name, index=False)
    df_rules_attack.to_csv(rules_attack_name, index=False)

    
    
    