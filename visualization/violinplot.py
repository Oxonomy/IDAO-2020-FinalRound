import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

warnings.filterwarnings('ignore')
sns.set()

df = pd.read_csv('train.csv', sep=',')
pd.set_option('display.max_columns', 500)
df.head()
df['is_region_0'] = df['region'] == 61
numerical = [c for c in df.columns if df[c].dtype.name != 'object']

fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(20, 20))
for idx, feat in enumerate(numerical[-16:]):
    ax = axes[int(idx / 4), idx % 4]
    sns.violinplot(x='is_region_0', y=feat, data=df, ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel(feat)
fig.tight_layout()
plt.show()
