import lib as lib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats

df_noise = pd.read_pickle("results/2_state_noise.pickle")
df_true  = pd.read_pickle("results/2_state_true.pickle")

PLOT_RANDOM_TRACES = True
PLOT_DISTRIBUTIONS = True
PLOT_LIFETIMES     = False
TEST_HMM_FIT       = False


# plt.plot(df_noise[df_noise["id"] == 1]["fret"])
# plt.plot(df_true[df_true["id"] == 1]["fret"])
# plt.show()

if PLOT_RANDOM_TRACES:

    # Pick 6 random trace IDs where one must be full length
    rand_traces_noise, random_ids = lib.pick_random_traces(trace_df = df_noise, n_traces = 6)
    rand_traces_true = df_true[df_true["id"].isin(random_ids)]

    # Plot selected random traces
    fig, ax = plt.subplots(nrows = 6, ncols = 1, figsize = (5,10))
    ax = ax.ravel()
    n = 0

    for id, grp_noise in rand_traces_noise.groupby("id"):
        grp_true = rand_traces_true[rand_traces_true["id"] == id]

        ax[n].plot(grp_noise["time"], grp_noise["fret"], label = "Noisy trace", color = "royalblue", lw = 1)
        ax[n].plot(grp_true["time"], grp_true["fret"], label = "True trace", color = "firebrick", lw = 0.8)

        ax[n].set_xlim(0,200)
        ax[n].set_ylim(0,1)
        ax[n].set_xlabel("time")
        ax[n].set_ylabel("FRET")
        ax[n].legend(loc = "upper right")
        n += 1

    plt.tight_layout()
    lib.save_current_fig("example_traces")

if PLOT_DISTRIBUTIONS:
    bins = np.arange(0, 1, 0.03)
    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (9,4))
    ax = ax.ravel()

    ax[0].hist(df_true["fret"], bins = bins, color = "firebrick", normed = True, zorder = 1, label = "True distribution")
    ax[1].hist(df_noise["fret"], bins = bins, color = "royalblue", normed = True, zorder = 2, label = "Observed distribution")

    for a in ax:
        a.set_xlim(0,1)
        a.set_xlabel("FRET")
        a.set_ylabel("Probability density")
        a.legend(loc = "upper right")
    plt.tight_layout()
    lib.save_current_fig("distributions")

if PLOT_LIFETIMES:
    # Plot lifetimes of true distribution
    lifetimes = []
    for id, grp_noise in tqdm(df_true.groupby("id")):
        lif = lib.hmm_lifetimes(grp_noise["fret"], grp_noise["time"])
        lifetimes.append(lif["lifetime"])

    lifetimes = lib.flatten_list(lifetimes, as_array = True)

    def _single_exp(x, scale):
        return stats.expon.pdf(x, loc = 0, scale = scale)

    tau, *_ = lib.lh_fit(f = _single_exp,
                         data = lifetimes,
                         binned_likelihood = True,
                         scale = 8,
                         limit_scale = (1, 15))

    xpts = np.linspace(0, 20, 200)
    bins = np.arange(0,20, 1)

    plt.hist(lifetimes, color = "lightgrey", bins = bins, normed = True, label = r"$\tau$ = {:.1f}".format(*tau))
    plt.plot(xpts, _single_exp(xpts, *tau), color = "firebrick")
    plt.xlim(1,15)

    plt.legend()
    plt.show()


if TEST_HMM_FIT:
    pass