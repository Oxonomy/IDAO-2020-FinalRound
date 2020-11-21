import pandas as pd


def preprocess(df):
    cols = [el for el in list(df.select_dtypes(include=['object']).columns) if el != 'card_id']
    df = pd.get_dummies(df, columns=cols)

    na_cols = df.loc[:, df.isna().any()].columns
    for col in na_cols:
        df[col + '_na'] = df[col].isna().astype(int)

    df = df.fillna(0)

    return df
