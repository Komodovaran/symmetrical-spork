import lib as lib
import funcs as funcs

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import uncertainties as un
from tqdm import tqdm

# Read pickles
df_S_noise         = pd.read_pickle("data/2S_state_noise.pickle")
df_S_true          = pd.read_pickle("data/2S_state_true.pickle")
df_S_true_nobleach = pd.read_pickle("data/2S_state_true_nobleach.pickle")

df_F_noise         = pd.read_pickle("data/2F_state_noise.pickle")
df_F_true          = pd.read_pickle("data/2F_state_true.pickle")
df_F_true_nobleach = pd.read_pickle("data/2F_state_true_nobleach.pickle")

# Make it so that all steps function independently, to reduce re-computing time
SINGLE_TRACE  = False
RANDOM_TRACES = False
DISTRIBUTIONS = False
LIFETIMES     = True
STICH_TRACES  = False

# Iterate over this (title, color, noise, true, true_nobleach)
dfs = [("S_", "#8DA0CB", df_S_noise, df_S_true, df_S_true_nobleach),
       ("F_", "#66C2A5", df_F_noise, df_F_true, df_F_true_nobleach)]

if SINGLE_TRACE:
    for dftitle, dfcol, df_noise, df_true, df_true_nobleach in dfs:

        # Pick 6 random trace IDs where one must be full length
        rand_trace, _ = lib.pick_random_traces(trace_df = df_true, n_traces = 1, min_frames = 50)

        # Plot selected random traces
        fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (6,2))

        lifetimes = lib.hmm_lifetimes(rand_trace["fret"], rand_trace["time"])
        n_datapoints = len(lifetimes)

        ax.plot(rand_trace["time"], rand_trace["fret"], label = "Trace fit", color = dfcol, lw = 1.3)

        # remove consecutive duplicates and endpoints to show how lifetimes are calculated
        rand_trace = rand_trace.loc[rand_trace["fret"].shift(-1) != rand_trace["fret"]][1:-1]
        ax.plot(rand_trace["time"], rand_trace["fret"], "o", markersize = 5, color = dfcol, label = "{} transitions".format(n_datapoints))

        ax.set_xlim(0,100)
        ax.set_ylim(0,1)
        ax.set_xlabel("time")
        ax.set_ylabel("FRET")
        ax.legend(loc = "upper right")

        plt.tight_layout()
        lib.save_current_fig(dftitle + "single_trace")

if RANDOM_TRACES:
    for dftitle, dfcol, df_noise, df_true, df_true_nobleach in dfs:
        # Pick 6 random trace IDs where one must be full length
        rand_traces_noise, random_ids = lib.pick_random_traces(trace_df = df_noise, n_traces = 5)
        rand_traces_true = df_true[df_true["id"].isin(random_ids)]

        # Plot selected random traces
        fig, axes = plt.subplots(nrows = len(set(random_ids)), ncols = 1, figsize = (8,4))

        tmax = int(df_true_nobleach["time"].max())

        for ax, (id, grp) in zip(axes, rand_traces_noise.groupby("id")):
            grp_true = rand_traces_true[rand_traces_true["id"] == id]
            t_i = int(grp["time"].max())

            ax.plot(grp["time"], grp["fret"], label = "Observed trace", color = dfcol, lw = 1.3)
            ax.plot(grp_true["time"], grp_true["fret"], color = "firebrick", lw = 0.7)

            if t_i < tmax:
                ax.axvspan(xmin = t_i, xmax = tmax, color = "black", fill = True, alpha = 0.1)

            ax.set_xlim(0,tmax)
            ax.set_ylim(-0.15, 1.15)
            plt.setp(ax.get_xticklabels(), visible = False)

        fig.text(0.08, 0.5, 'FRET', va = 'center', rotation = 'vertical')
        plt.setp(ax.get_xticklabels(), visible = True)
        plt.subplots_adjust(hspace = .0)
        plt.xlabel("time")
        lib.save_current_fig(dftitle + "example_traces")

if DISTRIBUTIONS:
    for dftitle, dfcol, df_noise, df_true, df_true_nobleach in dfs:
        bins = np.arange(0, 1, 0.03)
        fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (9,4))
        ax = ax.ravel()

        ax[0].hist(df_F_true["fret"], bins = bins, color = "firebrick", normed = True, zorder = 1, label = "True distribution")
        ax[1].hist(df_F_noise["fret"], bins = bins, color = dfcol, normed = True, zorder = 2, label = "Observed distribution")

        for a in ax:
            a.set_xlim(0,1)
            a.set_xlabel("FRET")
            a.set_ylabel("Probability density")
            a.legend(loc = "upper right")
        plt.tight_layout()
        lib.save_current_fig(dftitle + "distributions")

if LIFETIMES:
    for dftitle, dfcol, df_noise, df_true, df_true_nobleach in dfs:

        # Plot lifetimes of true distribution, and see how observed trace lengths affect this
        min_lengths = [0, 0.1, 0.25, 0.50, 0.75, 1] # percentages of max trace length.

        # Maximum number of traces
        n_orig = len(df_true_nobleach["id"].unique())

        taus = []
        lifetimes = []
        n_traces = []
        tmax = []

        for trace_len in min_lengths:
            if trace_len is 0:
                tmax_i = int(df_true_nobleach["time"].max())
                df = df_true_nobleach
            else:
                tmax_i = int(df_true_nobleach["time"].max() * trace_len)
                df = df_true.groupby("id").filter(lambda x: len(x) >= tmax_i)

            lifetimes_i = []
            for id, grp in tqdm(df.groupby("id")):
                lif = lib.hmm_lifetimes(grp["fret"], grp["time"])
                lifetimes_i.append(lif["lifetime"])

            lifetimes_i = lib.flatten_list(lifetimes_i, as_array = True)

            tau_i, err_i, _ = lib.lh_fit(f = funcs.single_exp,
                                  data = lifetimes_i,
                                  binned_likelihood = True,
                                  scale = 4,
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
            n_datapoints = len(lifetimes[i])

            ax[i].hist(lifetimes[i],
                       color = dfcol,
                       bins = bins,
                       normed = True,
                       label = "trace length $\geq$ {}% of longest\n{}/{} traces\n{} datapoints".format(int(min_lengths[i]*100),
                                                                                                        n_traces[i],
                                                                                                        n_orig,
                                                                                                        n_datapoints))

            ax[i].plot(xpts, funcs.single_exp(xpts, taus[i].n), color = "firebrick", label = r"$\tau$ = {}".format(taus[i]))
            ax[i].legend(loc = "upper right")
            ax[i].set_xlim(1)
            ax[i].set_xlabel("Dwell time")
            ax[i].set_ylabel("Probability density")

        plt.tight_layout()
        lib.save_current_fig(dftitle + "lifetimes")

        # Quite remarkably, the lifetime can be accurately estimated from very few traces, and so this certainly
        # tells us that more data isn't necessarily better, if the quality is poor


if STICH_TRACES:

    n_reshuffles = 100
    min_lengths = [0, 0.1, 0.25, 0.50, 0.75, 1]  # percentages of max trace length.

    # Plot lifetimes of true distribution, and see how observed trace lengths affect this

    # 1.   : remove traces shorter than 100% of observation time (remove any traces with bleaching)
    # .5   : remove traces shorter than 50 % of observation time
    # 0.   : remove traces shorter than 0 %  of observation time (keep all traces)

    for dftitle, dfcol, df_noise, df_true, df_true_nobleach in dfs:

        reshuf_tau = []

        # Maximum number of traces
        n_orig = len(df_true_nobleach["id"].unique())

        taus = []
        n_traces = []
        tmax = []

        for trace_len in min_lengths:
            if trace_len is 0:
                df = df_true_nobleach
            else:
                tmax_i = int(df_true_nobleach["time"].max() * trace_len)
                df = df_true.groupby("id").filter(lambda x: len(x) >= tmax_i)

            tau_reshuf = []
            for i in tqdm(range(n_reshuffles)):
                df_shuf = lib.shuffle_df_groups(df_true, group = "id")
                df_shuf["time"] = range(len(df_shuf))
                df_shuf["time"] += 1

                lifetimes = lib.hmm_lifetimes(df_shuf["fret"], df_shuf["time"], drop_extra_cols = True)

                tau_i, err_i, _ = lib.lh_fit(f = funcs.single_exp,
                                             data = lifetimes,
                                             binned_likelihood = True,
                                             scale = 4,
                                             limit_scale = (1, 15))
                tau_reshuf.append(*tau_i)
                n_traces.append(len(df_shuf["id"].unique()))

            # manually calc std err of mean for plt.errorbar
            # tr_mu   = np.mean(tau_reshuf)
            # tau_sem = np.std(tau_reshuf)/np.sqrt(n_reshuffles)
            # tau = un.ufloat(tr_mu, tau_sem)

            taus.append(tau_reshuf)

        sns.boxplot(x = min_lengths,
                    y = taus,
                    color = dfcol,
                    notch = True,
                    bootstrap = 1000,   # bootstrapping with resampling to get CI because we have no idea how the distribution looks like
                    width = 0.3)

        plt.xlabel("Traces shorter than fraction removed")
        plt.ylabel(r"$\tau$")

        lib.save_current_fig(dftitle + "stitched_mc")