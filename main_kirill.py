import pandas as pd
from utils.preprocess import preprocess, reset_averages, add_columns
from models.catboost_regression.catboost_regression import CatboostRegressor

df = pd.read_csv('test.csv', sep=',')
prediction = df[["card_id"]].copy(deep=True)

df = add_columns(df)
df = preprocess(df, False)
model = CatboostRegressor()
model.load_model()

X = df[df.columns[1:]].to_numpy()
prediction["target"] = reset_averages(model.predict(X))
prediction.to_csv("prediction.csv", index=False)
print(len(prediction))
