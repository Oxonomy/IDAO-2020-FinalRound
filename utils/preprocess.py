import pandas as pd
from joblib import dump, load


def preprocess(df: pd.DataFrame, is_train=True):
    if is_train:
        columns_dict = {}
        cat_cols = [el for el in list(df.select_dtypes(include=['object']).columns) if el != 'card_id']
        for c in cat_cols:
            columns_dict[c] = df[c].unique()
        df = pd.get_dummies(df, columns=cat_cols)

        columns_order = list(df.columns)
        columns_order.remove('target')
        dump((columns_dict, columns_order), 'columns_dict.joblib')
    else:
        columns_dict, columns_order = load('columns_dict.joblib')
        for feature in columns_dict.keys():
            for v in columns_dict[feature]:
                df[feature + '_' + v] = df[feature] == v
            df = df.drop([feature], axis=1)
        df = df[columns_order]
    df = df.fillna(0)
    return df


def add_columns(df: pd.DataFrame):
    df['addr_region_fact_encoding2'] = (df['addr_region_fact_encoding2'] * 11).round(0).astype(int)
    df['addr_region_fact_encoding1'] = (df['addr_region_fact_encoding1'] * 83).round(0).astype(int)
    df['addr_region_reg_encoding1'] = (df['addr_region_reg_encoding1'] * 83).round(0).astype(int)
    df['addr_region_reg_encoding2'] = (df['addr_region_reg_encoding2'] * 11).round(0).astype(int)
    df['app_addr_region_reg_encoding2'] = (df['app_addr_region_reg_encoding2'] * 11).round(0).astype(int)
    df['app_addr_region_reg_encoding1'] = (df['app_addr_region_reg_encoding1'] * 83).round(0).astype(int)
    df['app_addr_region_fact_encoding1'] = (df['app_addr_region_fact_encoding1'] * 83).round(0).astype(int)
    df['app_addr_region_fact_encoding2'] = (df['app_addr_region_fact_encoding2'] * 11).round(0).astype(int)
    df['app_addr_region_sale_encoding1'] = (df['app_addr_region_sale_encoding1'] * 39).round(0).astype(int)
    df['app_addr_region_sale_encoding2'] = (df['app_addr_region_sale_encoding2'] * 7).round(0).astype(int)
    df['t_0'] = (df['addr_region_fact_encoding1'] + df['addr_region_fact_encoding2'])
    df['t_1'] = (df['addr_region_reg_encoding1'] + df['addr_region_reg_encoding2'])
    df['t_2'] = (df['app_addr_region_reg_encoding2'] + df['app_addr_region_reg_encoding1'])
    df['t_3'] = (df['app_addr_region_fact_encoding2'] + df['app_addr_region_fact_encoding1'])
    df['t_4'] = (df['app_addr_region_sale_encoding2'] + df['app_addr_region_sale_encoding1'])
    df['notnull'] = pd.np.sum(df.notnull().to_numpy(), axis=1)
    return df


def reset_averages(predict):
    order = pd.np.argsort(-predict)
    top_k = int(0.05 * len(predict))
    predict[(predict < predict[order][top_k]) & (predict > predict[order][-top_k])] = -1
    return predict
