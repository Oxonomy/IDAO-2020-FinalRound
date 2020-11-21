import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

warnings.filterwarnings('ignore')
sns.set()

df = pd.read_csv('train.csv', sep=',')
df = df[df.columns[-10:]].sample(5000)
numerical = [c for c in df.columns if df[c].dtype.name != 'object']

g = sns.pairplot(df, kind="scatter", diag_kind="kde", plot_kws=dict(marker=".", linewidth=1, color="b"), height=5)
g.map_lower(sns.histplot, **{'bins': 50, 'pthresh': .1, 'color': "b"})
g.map_lower(sns.kdeplot, **{'levels': 4, 'color': "gray", 'linewidth': 1})
plt.show()
