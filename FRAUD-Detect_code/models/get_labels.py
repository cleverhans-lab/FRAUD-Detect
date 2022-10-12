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


    
def prepare_data(data, rseed):

    dataset, decision = data_dict[data]
    datadir = '../preprocessing/preprocessed/{}/'.format(dataset)    

    #filenames
    suffix = 'OneHot'
    train_file      = '{}{}_train{}_{}.csv'.format(datadir, dataset, suffix, rseed)
    test_file       = '{}{}_test{}_{}.csv'.format(datadir, dataset, suffix, rseed)
    sg_file         = '{}{}_attack{}_{}.csv'.format(datadir, dataset, suffix, rseed)

    # load dataframe
    df_train    = pd.read_csv(train_file)
    df_test     = pd.read_csv(test_file)
    df_sg       = pd.read_csv(sg_file)

    # prepare the data
    scaler = StandardScaler()
    ## training set
    y_train = df_train[decision]
    X_train = df_train.drop(labels=[decision], axis = 1)
    X_train = scaler.fit_transform(X_train)
    ### cast
    X_train = np.asarray(X_train).astype(np.float32)
    y_train = np.asarray(y_train).astype(np.float32)

    ## test set
    y_test = df_test[decision]
    X_test = df_test.drop(labels=[decision], axis = 1)
    X_test = scaler.fit_transform(X_test)
    ### cast
    X_test = np.asarray(X_test).astype(np.float32)
    y_test = np.asarray(y_test).astype(np.float32)

     ## sg set
    y_sg = df_sg[decision]
    X_sg = df_sg.drop(labels=[decision], axis = 1)
    X_sg = scaler.fit_transform(X_sg)
    ### cast
    X_sg = np.asarray(X_sg).astype(np.float32)
    y_sg = np.asarray(y_sg).astype(np.float32)

    return X_train, y_train, X_test, y_test, X_sg, y_sg





def get_labels(dataset, model_class, rseed):

    # load data as np array
    X_train, y_train, X_test, y_test, X_sg, y_sg = prepare_data(dataset, rseed)
    
    # model path
    outdir = '../models/pretrained/{}/'.format(dataset)
    model_path = '{}{}_{}.h5'.format(outdir, model_class, rseed)

    labeldir = './labels/{}/'.format(dataset)
    
    if not os.path.exists(labeldir):
        os.mkdir(labeldir)
   

    def get_predictions(model_class, X_train, y_train, X_test, y_test, X_sg, y_sg):
        predictions_train, predictions_test, predictions_sg = None, None, None

        if model_class == 'DNN':
            # load model
            mdl = load_model(model_path)

            # get prediction
            #---train
            predictions_train = (mdl.predict(X_train) > 0.5).astype('int32')
            predictions_train = [x[0] for x in predictions_train]
            #---test
            predictions_test = (mdl.predict(X_test) > 0.5).astype('int32')
            predictions_test = [x[0] for x in predictions_test]
            #---sg
            predictions_sg = (mdl.predict(X_sg) > 0.5).astype('int32')
            predictions_sg = [x[0] for x in predictions_sg]

            
        
        if model_class in ['RF', 'SVM', 'AdaBoost', 'XgBoost']:
            # load model
            mdl = pickle.load(open(model_path,"rb"))

            # get prediction
            #---train
            predictions_train = mdl.predict(X_train)
            predictions_train = [int(x) for x in predictions_train]

            #---test
            predictions_test = mdl.predict(X_test)
            predictions_test = [int(x) for x in predictions_test]

            #---sg
            predictions_sg = mdl.predict(X_sg) 
            predictions_sg = [int(x) for x in predictions_sg]


        return predictions_train, predictions_test, predictions_sg

    _, predictions_test, predictions_sg = get_predictions(model_class, X_train, y_train, X_test, y_test, X_sg, y_sg)

    label_name_sg = '{}{}_sg_{}.csv'.format(labeldir, model_class, rseed)
    label_name_test = '{}{}_test_{}.csv'.format(labeldir, model_class, rseed)

    df_sg = pd.DataFrame()
    df_sg['prediction'] = predictions_sg
    df_sg.to_csv(label_name_sg, index=False)

    df_test = pd.DataFrame()
    df_test['prediction'] = predictions_test
    df_test.to_csv(label_name_test, index=False)

    

    


if __name__ == '__main__':
    # parser initialization
    parser = argparse.ArgumentParser(description='Script pretraining bbox models')
    parser.add_argument('--dataset', type=str, default='adult_income', help='adult_income, compas, default_credit, marketing')
    parser.add_argument('--rseed', type=int, default=0, help='random seed: choose between 0 - 9')
    parser.add_argument('--model_class', type=str, default='DNN', help='DNN, RF, AdaBoost, XgBoost')

    # get input
    args = parser.parse_args()
    dataset = args.dataset
    rseed = args.rseed
    model_class = args.model_class


    get_labels(dataset, model_class, rseed)


