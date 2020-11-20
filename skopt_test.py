import os
from time import sleep

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_blobs
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot
from os import makedirs
import numpy as np

# fit model on dataset
from sklearn.model_selection import KFold


def fit_model(trainX, trainy):
    # define model
    model = Sequential()
    model.add(Dense(25, input_dim=2, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit model
    model.fit(trainX, trainy, epochs=100, verbose=0, batch_size=512)
    return model


# generate 2d classification dataset
X, y = make_blobs(n_samples=100000, centers=3, n_features=2, cluster_std=2, random_state=2)
y = to_categorical(y)
X, X_true, y, y_true = train_test_split(X, y, test_size=0.9)

X, X_val, y, y_val = train_test_split(X, y, test_size=0.2)

n_members = 10

models = []
score = 0

rkf = RepeatedKFold(n_splits=n_members, n_repeats=3, random_state=12)
for train, test in rkf.split(X):
    model = fit_model(X[train], y[train])
    models.append(model)

    predict = model.predict(X[test])
    score += accuracy_score(y[test], (predict > 0.5))
print("Test:", score / len(models))

y_predict = y_val * 0
for model in models:
    predict = model.predict(X_val)

    y_predict += predict / len(models)

print('--/--' * 10)
print("Val:", accuracy_score(y_val, (y_predict > 0.5)))


y_predict = y_true * 0
for model in models:
    predict = model.predict(X_true, batch_size=1024)

    y_predict += predict / len(models)

print('--/--' * 10)
print("True:", accuracy_score(y_true, (y_predict > 0.5)))
