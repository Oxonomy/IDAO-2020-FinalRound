import random
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
from sklearn.datasets import fetch_california_housing
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly
import plotly.graph_objs as go

import config as c
from pipeline.Layer import Layer
from utils.cuda import turn_off_gpu
from models.svm.svm import SVM
from utils.metrics import roc_auc_score_at_K
from utils.preprocess import preprocess 
from models.keras_dense_classifier.keras_dense_classifier import KerasDenseClassifier as KDC
from visualization.utils import plot_correlation_matrix, plot_scatterplot_matrix

warnings.filterwarnings('ignore')

df = pd.read_csv('test.csv', sep=',')
prediction = df[["card_id"]].copy(deep=True)

df['addr_region_fact_encoding2'] = (df['addr_region_fact_encoding2']*11).round(0).astype(int)
df['addr_region_fact_encoding1'] = (df['addr_region_fact_encoding1']*83).round(0).astype(int)
df['addr_region_reg_encoding1'] = (df['addr_region_reg_encoding1']*83).round(0).astype(int)
df['addr_region_reg_encoding2'] = (df['addr_region_reg_encoding2']*11).round(0).astype(int)
df['app_addr_region_reg_encoding2'] = (df['app_addr_region_reg_encoding2']*11).round(0).astype(int)
df['app_addr_region_reg_encoding1'] = (df['app_addr_region_reg_encoding1']*83).round(0).astype(int)
df['app_addr_region_fact_encoding1'] = (df['app_addr_region_fact_encoding1']*83).round(0).astype(int)
df['app_addr_region_fact_encoding2'] = (df['app_addr_region_fact_encoding2']*11).round(0).astype(int)
df['app_addr_region_sale_encoding1'] = (df['app_addr_region_sale_encoding1']*39).round(0).astype(int)
df['app_addr_region_sale_encoding2'] = (df['app_addr_region_sale_encoding2']*7).round(0).astype(int)
df = preprocess(df)

df = df[['inquiry_1_week', 'channel_name_2_cat_5', 'channel_name_modified_2018_cat_2']]
X = df.to_numpy()
kds = KDC()
kds.load_ensemble()
y = kds.predict(X)

prediction["target"] = y
prediction.to_csv("prediction.csv", index=False)
