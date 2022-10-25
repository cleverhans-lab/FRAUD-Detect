import pandas as pd 



def process_compas(df):
    df_new = df.rename(columns = {
                                    'race_African-American': 'race_African_American'
                                })
    return df_new


def process_marketing(df):
    df_new = df.rename(columns = {
                                    'age_age:30-60': 'age_age_30_60',
                                    'age_age:not30-60': 'age_age_not_30_60'
                                })
    return df_new





def prepare(dataset, suffix):
    for rseed in range(10):
        filename = "./preprocessed/{}/{}_{}OneHot_{}.csv".format(dataset, dataset, suffix, rseed)
        df = pd.read_csv(filename)
        if dataset == "compas":
            df = process_compas(df)
            df.to_csv(filename, index=False)
        if dataset == "marketing":
            df = process_marketing(df)
            df.to_csv(filename, index=False)
        
        

datasets = ["compas", "marketing"]
suffixes = ["attack", "test"]

for dataset in datasets:
    for suffix in suffixes:
        prepare(dataset, suffix)
