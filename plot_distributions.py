import lib as lib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats
import uncertainties as un

# Read pickles
df_S_noise         = pd.read_pickle("data/2S_state_noise.pickle")
df_S_true          = pd.read_pickle("data/2S_state_true.pickle")
df_S_true_nobleach = pd.read_pickle("data/2S_state_true_nobleach.pickle")

df_F_noise         = pd.read_pickle("data/2F_state_noise.pickle")
df_F_true          = pd.read_pickle("data/2F_state_true.pickle")
df_F_true_nobleach = pd.read_pickle("data/2F_state_true_nobleach.pickle")

# Make it so that all steps function independently, to reduce re-computing time
PLOT_SINGLE_TRACE  = True
PLOT_RANDOM_TRACES = False
PLOT_DISTRIBUTIONS = False
PLOT_LIFETIMES     = False
TEST_HMM_FIT       = False

# Iterate over this
dfs = [("S_", df_S_noise, df_S_true, df_S_true_nobleach),
       ("F_", df_F_noise, df_F_true, df_F_true_nobleach)]


if PLOT_SINGLE_TRACE:
    for dftitle, df_noise, df_true, df_true_nobleach in dfs:
        # Pick 6 random trace IDs where one must be full length
        rand_trace, _ = lib.pick_random_traces(trace_df = df_true, n_traces = 1)

        # Plot selected random traces
        fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (5,2))

        lifetimes = lib.hmm_lifetimes(rand_trace["fret"], rand_trace["time"])
        n_datapoints = len(lifetimes)

        for id, grp in rand_trace.groupby("id"):
            ax.plot(grp["time"], grp["fret"], label = "Trace fit", color = "royalblue", lw = 1)

            grp = grp.loc[grp["fret"].shift(-1) != grp["fret"]]
            ax.plot(grp["time"], grp["fret"], "o", markersize = 5, color = "royalblue", label = "Observed {} transitions".format(n_datapoints))

            ax.set_xlim(0,200)
            ax.set_ylim(0,1)
            ax.set_xlabel("time")
            ax.set_ylabel("FRET")
            ax.legend(loc = "upper right")

        plt.tight_layout()
        lib.save_current_fig(dftitle + "single_trace")

if PLOT_RANDOM_TRACES:
    for dftitle, df_noise, df_true, df_true_nobleach in dfs:
        # Pick 6 random trace IDs where one must be full length
        rand_traces_noise, random_ids = lib.pick_random_traces(trace_df = df_noise, n_traces = 6)
        rand_traces_true = df_true[df_true["id"].isin(random_ids)]

        # Plot selected random traces
        fig, ax = plt.subplots(nrows = 6, ncols = 1, figsize = (5,10))
        ax = ax.ravel()
        n = 0

        for id, grp in rand_traces_noise.groupby("id"):
            grp_true = rand_traces_true[rand_traces_true["id"] == id]

            ax[n].plot(grp["time"], grp["fret"], label = "Observed trace", color = "royalblue", lw = 1)
            ax[n].plot(grp_true["time"], grp_true["fret"], label = "Underlying true trace", color = "firebrick", lw = 0.8)

            ax[n].set_xlim(0,200)
            ax[n].set_ylim(0,1)
            ax[n].set_xlabel("time")
            ax[n].set_ylabel("FRET")
            ax[n].legend(loc = "upper right")
            n += 1

        plt.tight_layout()
        lib.save_current_fig(dftitle + "example_traces")

if PLOT_DISTRIBUTIONS:
    bins = np.arange(0, 1, 0.03)
    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (9,4))
    ax = ax.ravel()

    ax[0].hist(df_F_true["fret"], bins = bins, color = "firebrick", normed = True, zorder = 1, label = "True distribution")
    ax[1].hist(df_F_noise["fret"], bins = bins, color = "royalblue", normed = True, zorder = 2, label = "Observed distribution")

    for a in ax:
        a.set_xlim(0,1)
        a.set_xlabel("FRET")
        a.set_ylabel("Probability density")
        a.legend(loc = "upper right")
    plt.tight_layout()
    lib.save_current_fig("distributions")

if PLOT_LIFETIMES:

    for dftitle, df_noise, df_true, df_true_nobleach in dfs:

        def _single_exp(x, scale):
            return stats.expon.pdf(x, loc = 0, scale = scale)

        # Plot lifetimes of true distribution, and see how observed trace lengths affect this
        min_lengths = [0, 0.1, 0.25, 0.50, 0.75, 1] # percentages of max trace length.

        # Maximum number of traces
        n_orig = len(df_true_nobleach["id"].unique())

        taus = []
        lifetimes = []
        n_traces = []
        tmax = []

        for trace_len in min_lengths:
            tmax_i = int(df_true_nobleach["time"].max() * trace_len)
            df = df_F_true.groupby("id").filter(lambda x: len(x) >= tmax_i)

            lifetimes_i = []
            for id, grp in tqdm(df.groupby("id")):
                lif = lib.hmm_lifetimes(grp["fret"], grp["time"])
                lifetimes_i.append(lif["lifetime"])

            lifetimes_i = lib.flatten_list(lifetimes_i, as_array = True)

            tau_i, err_i, _ = lib.lh_fit(f = _single_exp,
                                  data = lifetimes_i,
                                  binned_likelihood = True,
                                  scale = 8,
                                  limit_scale = (1, 15))

            tau_i = un.ufloat(*tau_i, *err_i)

            n_traces.append(len(df["id"].unique()))
            taus.append(tau_i)
            lifetimes.append(lifetimes_i)
            tmax.append(tmax_i)

        xpts = np.linspace(0, 20, 200)
        bins = np.arange(0,20, 1)

        fig, ax = plt.subplots(nrows = 2, ncols = 3, figsize = (12,6))
        ax = ax.ravel()

        for i in range(len(ax)):
            ax[i].hist(lifetimes[i],
                       color = "lightgrey",
                       bins = bins,
                       normed = True,
                       label = "trace length $\geq$ {}% of longest\n{}/{} traces\n{} datapoints".format(int(min_lengths[i]*100),
                                                                                                        n_traces[i],
                                                                                                        n_orig,
                                                                                                        n_datapoints))

            ax[i].plot(xpts, _single_exp(xpts, taus[i].n), color = "firebrick", label = r"$\tau$ = {}".format(taus[i]))
            ax[i].legend(loc = "upper right")
            ax[i].set_xlim(1)
            ax[i].set_xlabel("Dwell time")
            ax[i].set_ylabel("Probability density")

        plt.tight_layout()
        lib.save_current_fig(dftitle + "lifetimes")
        plt.show()

        # Quite remarkably, the lifetime can be accurately estimated from very few traces, and so this certainly
        # tells us that more data isn't necessarily better, if the quality is poor
        # TODO: do the same for slow enzyme

if TEST_HMM_FIT:
    # do we need this, as we already have the fit of the trace?
    pass