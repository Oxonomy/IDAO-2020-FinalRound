import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import IncrementalPCA

sns.set_theme(style="darkgrid")

from mlxtend.preprocessing import standardize
from mlxtend.feature_extraction import PrincipalComponentAnalysis
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression


df = pd.read_csv('train.csv', sep=',')
pd.set_option('display.max_columns', 500)
df['addr_region_fact_encoding2'] = (df['addr_region_fact_encoding2']*11).round(0)#.astype(int)
df['addr_region_fact_encoding1'] = (df['addr_region_fact_encoding1']*83).round(0)#.astype(int)
df['addr_region_reg_encoding1'] = (df['addr_region_reg_encoding1']*83).round(0)#.astype(int)
df['addr_region_reg_encoding2'] = (df['addr_region_reg_encoding2']*11).round(0)#.astype(int)
df['app_addr_region_reg_encoding2'] = (df['app_addr_region_reg_encoding2']*11).round(0)#.astype(int)
df['app_addr_region_reg_encoding1'] = (df['app_addr_region_reg_encoding1']*83).round(0)#.astype(int)
df['app_addr_region_fact_encoding1'] = (df['app_addr_region_fact_encoding1']*83).round(0)#.astype(int)
df['app_addr_region_fact_encoding2'] = (df['app_addr_region_fact_encoding2']*11).round(0)#.astype(int)
df['app_addr_region_sale_encoding1'] = (df['app_addr_region_sale_encoding1']*39).round(0)#.astype(int)
df['app_addr_region_sale_encoding2'] = (df['app_addr_region_sale_encoding2']*7).round(0)#.astype(int)

df = df[df.columns[26:36]]
X = df.to_numpy()


pca = IncrementalPCA(n_components=6, batch_size=100)
pca.fit(X)
X_pca = pca.transform(X)

reg = LinearRegression().fit(X_pca, X)
print(reg.score(X_pca, X))
