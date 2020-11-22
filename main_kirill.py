import pandas as pd
from joblib import dump, load
from catboost import CatBoostRegressor, CatBoostClassifier
from utils.preprocess import preprocess, reset_averages
import numpy as np

print("LL")
df = pd.read_csv('test.csv', sep=',')
prediction = df[["card_id"]].copy(deep=True)


df = preprocess(df, False)
model = load('best_catboost.joblib')


X = df[df.columns[1:]].to_numpy()
prediction["target"] = reset_averages(model.predict(X))
prediction.to_csv("prediction.csv", index=False)
print(len(prediction))
