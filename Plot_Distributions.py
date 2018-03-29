import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df_noise = pd.read_pickle("results/2_state_noise.pickle")
df_true  = pd.read_pickle("results/2_state_true.pickle")

bins = np.arange(0, 1, 0.05)

fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (10,5))
ax = ax.ravel()

ax[0].hist(df_true["fret"], bins = bins, color = "lightgrey", normed = True, zorder = 1, label = "True distribution")
ax[1].hist(df_noise["fret"], bins = bins, color = "orange", normed = True, zorder = 2, label = "Observed distribution")

for a in ax:
    a.set_xlim(0,1)
    a.legend(loc = "upper right")

plt.tight_layout()
plt.show()