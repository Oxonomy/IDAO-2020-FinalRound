import pandas as pd
from joblib import dump, load


def preprocess(df: pd.DataFrame, is_train=True):
    if is_train:
        columns_dict = {}
        cat_cols = [el for el in list(df.select_dtypes(include=['object']).columns) if el != 'card_id']
        for c in cat_cols:
            columns_dict[c] = df[c].unique()
        df = pd.get_dummies(df, columns=cat_cols)
        dump(columns_dict, 'columns_dict.joblib')
    else:
        columns_dict = load('columns_dict.joblib')
        for feature in columns_dict.keys():
            for v in columns_dict[feature]:
                df[feature+'_'+v] = df[feature]==v
            df=df.drop([feature], axis=1)

    df = df.fillna(0)
    return df
