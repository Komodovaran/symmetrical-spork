import lib as lib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_noise = pd.read_pickle("results/2_state_noise.pickle")
df_true  = pd.read_pickle("results/2_state_true.pickle")

tmax = df_noise["time"].max()
# Pick 6 random trace IDs where one must be full length
t = 0
while t != tmax:
    random_ids = np.random.choice(len(df_noise["id"].unique()), size = 6)
    random_traces = df_noise[df_noise["id"].isin(random_ids)]
    t = df_noise["time"].max()

# Plot selected random traces
fig, ax = plt.subplots(nrows = 2, ncols = 3, figsize = (8,3))
ax = ax.ravel()
n = 0
for id, grp in random_traces.groupby("id"):
    ax[n].plot(grp["time"], grp["fret"], label = id, color = "black")
    ax[n].set_xlim(0,200)
    ax[n].set_ylim(0,1)
    ax[n].set_xlabel("time")
    ax[n].set_ylabel("FRET")
    ax[n].legend(loc = "upper right")
    n += 1

plt.tight_layout()
lib.save_current_fig("example_traces")


# Plot distribution
bins = np.arange(0, 1, 0.03)
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (10,5))
ax = ax.ravel()

ax[0].hist(df_true["fret"], bins = bins, color = "lightgrey", normed = True, zorder = 1, label = "True distribution")
ax[1].hist(df_noise["fret"], bins = bins, color = "orange", normed = True, zorder = 2, label = "Observed distribution")

for a in ax:
    a.set_xlim(0,1)
    a.legend(loc = "upper right")
plt.tight_layout()
lib.save_current_fig("distributions")