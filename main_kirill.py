import pandas as pd
import numpy as np
from joblib import dump, load
from utils.preprocess import preprocess, reset_averages, add_columns

df = pd.read_csv('test.csv', sep=',')
prediction = df[["card_id"]].copy(deep=True)

df = add_columns(df)
df = preprocess(df, False)
X = df[df.columns[1:]].to_numpy()

#predict = np.zeros(len(df))
#for i in range(10):
#model = load("models_joblib/models_" + str(0) + "_.joblib")
model = load("model.joblib")
predict = model.predict(X)

#predict -= predict.min()

prediction["target"] = reset_averages(predict)
prediction.to_csv("prediction.csv", index=False)
print(len(prediction))
