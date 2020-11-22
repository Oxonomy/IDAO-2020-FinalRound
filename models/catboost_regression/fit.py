import random
import plotly
import warnings
import keras as k
import numpy as np
import pandas as pd
import seaborn as sns
from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_objective
from skopt.utils import use_named_args
import matplotlib.pyplot as plt
from joblib import dump, load
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import roc_curve, roc_auc_score
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from models.linear_regression.linear_regression import LinearRegression as LR

import config as c
from sklearn.model_selection import train_test_split
from pipeline.Layer import Layer
from utils.cuda import turn_off_gpu
from models.svm.svm import SVM
from utils.metrics import roc_auc_score_at_K
from models.catboost_regression.catboost_regression import CatboostRegressor
from utils.preprocess import preprocess, add_columns, reset_averages
from models.keras_dense_classifier.keras_dense_classifier import KerasDenseClassifier as KDC
from visualization.utils import plot_correlation_matrix, plot_scatterplot_matrix

warnings.filterwarnings('ignore')

df = pd.read_csv('train.csv', sep=',')
pd.set_option('display.max_columns', 500)
#df = add_columns(df)
df = preprocess(df)

X = df[df.columns[2:]].to_numpy()
y = df['target'].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

cr=CatboostRegressor()
cr.create_model(CatboostRegressor.default_model_constructor_parameters)
print(cr.fit_model(X, y))
