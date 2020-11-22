import pandas as pd
from models.linear_regression.linear_regression import LinearRegression as LR
from utils.preprocess import preprocess
import numpy as np

print("LL")
df = pd.read_csv('test.csv', sep=',')
prediction = df[["card_id"]].copy(deep=True)


df = preprocess(df, False)
lr = LR()
lr.load_model()

X = df[df.columns[-171:-133]].to_numpy()
prediction["target"] = lr.predict(X).reshape(-1)
prediction.to_csv("prediction.csv", index=False)
print(len(prediction))

test = pd.read_csv("test.csv")
