import numpy as np
import pandas as pd
from core import save


def get_adult_income(save_df=False):

    # output files
    dataset = 'adult_income'
    decision = 'income'

    raw_data_1 = np.genfromtxt('./raw_datasets/adult_income/adult.data', delimiter=', ', dtype=str)

    raw_data_2 = np.genfromtxt('./raw_datasets/adult_income/adult.test', delimiter=', ', dtype=str, skip_header=1)

    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'educational-num', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

    df_1 = pd.DataFrame(raw_data_1, columns=column_names)
    df_2 = pd.DataFrame(raw_data_2, columns=column_names)

    df = pd.concat([df_1, df_2], axis=0)


    # For more details on how the below transformations 
    df = df.astype({"age": np.int64, "educational-num": np.int64, "hours-per-week": np.int64, "capital-gain": np.int64, "capital-loss": np.int64 })

    df = df.replace({'workclass': {'Without-pay': 'Other/Unknown', 'Never-worked': 'Other/Unknown'}})
    df = df.replace({'workclass': {'Federal-gov': 'Government', 'State-gov': 'Government', 'Local-gov':'Government'}})
    df = df.replace({'workclass': {'Self-emp-not-inc': 'Self-Employed', 'Self-emp-inc': 'Self-Employed'}})
    df = df.replace({'workclass': {'Never-worked': 'Self-Employed', 'Without-pay': 'Self-Employed'}})
    df = df.replace({'workclass': {'?': 'Other/Unknown'}})

    df = df.replace({'occupation': {'Adm-clerical': 'White-Collar', 'Craft-repair': 'Blue-Collar',
                                           'Exec-managerial':'White-Collar','Farming-fishing':'Blue-Collar',
                                            'Handlers-cleaners':'Blue-Collar',
                                            'Machine-op-inspct':'Blue-Collar','Other-service':'Service',
                                            'Priv-house-serv':'Service',
                                           'Prof-specialty':'Professional','Protective-serv':'Service',
                                            'Tech-support':'Service',
                                           'Transport-moving':'Blue-Collar','Unknown':'Other/Unknown',
                                            'Armed-Forces':'Other/Unknown','?':'Other/Unknown'}})

    df = df.replace({'marital-status': {'Married-civ-spouse': 'Married', 'Married-AF-spouse': 'Married', 'Married-spouse-absent':'Married','Never-married':'Single'}})


    df = df[['age','workclass','education', 'marital-status', 'relationship', 'occupation', 'race', 'gender', 'capital-gain', 'capital-loss',  'hours-per-week', 'income']]

    #df = df[['age','workclass','education', 'marital-status', 'relationship', 'occupation', 'race', 'gender',  'hours-per-week', 'income']]

    df = df.replace({'income': {'<=50K': 0, '<=50K.': 0,  '>50K': 1, '>50K.': 1}})

    df = df.replace({'education': {'Assoc-voc': 'Assoc', 'Assoc-acdm': 'Assoc',
                                           '11th':'School', '10th':'School', '7th-8th':'School', '9th':'School',
                                          '12th':'School', '5th-6th':'School', '1st-4th':'School', 'Preschool':'School'}})


    df = df.rename(columns={'marital-status': 'marital_status', 'hours-per-week': 'hours_per_week', 'capital-gain': 'capital_gain', 'capital-loss': 'capital_loss'})
    #df = df.rename(columns={'marital-status': 'marital_status', 'hours-per-week': 'hours_per_week'})

    if save_df:
        for rseed in range(10):
            save(df, dataset, decision, rseed)
    #return df