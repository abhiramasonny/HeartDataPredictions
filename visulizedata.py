import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import *

data = pd.read_csv('Heart Attack.csv')
needsnorms = ['age', 'gender', 'impluse', 'pressurehight', 'pressurelow', 'glucose', 'kcm', 'troponin']
data[needsnorms] = data[needsnorms].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

stddev = 2
for col in needsnorms:
    data[col] = gaussian_filter1d(data[col], stddev) ** 2
fig, ax = plt.subplots(figsize=(12, 8))
colors = {'positive':'red', 'negative':'green'}

for col in needsnorms:
    ax.scatter(data.index, data[col], label=col, color=data['class'].apply(lambda x: colors[x]))

ax.set_xlabel('Index')
ax.set_ylabel('Normalized Value')
ax.set_title('Gaussian Smoothed Data')
ax.legend(loc='upper left', bbox_to_anchor=(1,1))

plt.tight_layout()
plt.show()
