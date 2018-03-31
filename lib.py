import pomegranate as pg
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import os
import probfit
import iminuit
from sklearn import cluster

pd.set_option("display.width", 1000)
pd.set_option('display.max_rows', 1000)
pd.options.mode.chained_assignment = None  # default='warn'


def flatten_list(input_list, as_array = False):
    """
    Parameters
    ----------
    input_list:
        flattens python list of lists to single list
    as_array:
        True returns numpy array, False returns iterable python list
    Returns
    -------
    flattened list in the chosen format
    """

    flat_lst = list(itertools.chain.from_iterable(input_list))
    if as_array:
        flat_lst = np.array(flat_lst)
    return flat_lst


def generate_traces(output_title,
                    transitions,
                    starts,
                    means,
                    noise,
                    n_traces = 100,
                    trace_max_len = 200,
                    trace_min_len = 10,
                    bleach_time = 50):
    """
    Generates random traces for k number of states and saves to pickle

    Parameters
    ----------
    output_title:
        title
    transitions:
        k * k transition matrix
    starts:
        k start points
    means:
        k means
    noise:
        singular value for all state noises, assuming they're the same
    n_traces:
        number of traces to generate
    trace_max_len
        longest observation time possible
    trace_min_len
        shortest observation time possible
    bleach_time
        rate of single exponential bleaching
    """

    # Initialize distributions from means and noise
    dists = [pg.NormalDistribution(m, noise) for m in means]

    # Generate HMM model
    model = pg.HiddenMarkovModel.from_matrix(transitions, distributions = dists, starts = starts)
    model.bake()

    fret_lst, time_lst, id_lst = [], [], []
    n = 0
    while n < n_traces:
        if bleach_time:
            # Bleach time from single exp dist
            bleach = np.int(np.random.exponential(bleach_time, 1))

            if bleach < trace_min_len:
                continue

            # Generate samples, and remove datapoints after bleaching
            fret = np.array(model.sample(trace_max_len))[:bleach]
            time = np.array(range(len(fret)))[:bleach]
            time +=1
        else:
            fret = np.array(model.sample(trace_max_len))
            time = np.array(range(len(fret)))
            time +=1

        # Append results
        fret_lst.append(fret)
        time_lst.append(time)
        id_lst.append([n]*len(fret))

        n += 1

    fret_lst = flatten_list(fret_lst, as_array = True)
    time_lst = flatten_list(time_lst, as_array = True)
    id_lst   = flatten_list(id_lst, as_array = True)

    # noinspection PyTypeChecker
    df = pd.DataFrame(dict(fret = fret_lst,
                           time = time_lst,
                           id = id_lst))

    # Save results to pickle
    os.makedirs("data/", exist_ok = True)

    output = "data/" + output_title + ".pickle"
    df.to_pickle(output, compression = None)



def fit_hmm(obs,
            time,
            model_selection = "likelihood",
            trans_prob = 0.25,
            self_trans = 0.90,
            state_noise = 0.1,
            min_trans = 0.01,
            init_means = False):
    """
    Fit traces with a Hidden Markov model.

    ** Might need optimization **

    Parameters
    ----------
    obs:
        values to fit
    time:
        timepoints for values to fit
    model_selection:
        determine best fit from "bic" or "likelihood", or a fixed number of states
    trans_prob:
        transition probabilities
    self_trans:
        self-transition probabilities
    state_noise:
        gaussian noise of each state
    min_trans:
        smallest transition to be considered
    init_means:
        list of initial means for each state. If none given, means are initialized by k-means

    Returns
    -------
    a tuple containing an array of idealized fit values the same length as input and an array of hmm_lifetimes
    """

    def _bic(data, k_states, log_likelihood):
        """
        Bayesian information criterion
        - parameters for k_states gaussians (2*k)
        - k_states * k_states size transition matrix (k**2)
        - start and end probabilities for each state (2*k)
        """
        n_params = k_states*4 + k_states**2
        return np.log(len(data)) * n_params - 2 * log_likelihood

    def _means_type(data, init_means, k_states):

        if init_means and len(init_means) is k_states:
            dist_means = init_means
        else:
            dist_means = cluster.KMeans(k_states).fit(data).cluster_centers_
        return dist_means

    def _start_end(k_states):
        """
        uniform starts and end probabilities as initial guesses.
        Will be normalized internally by Pomegranate to sum up to 1
        """
        starts = np.array([1]*k_states)
        ends = starts
        return starts, ends

    # Read in the data and trim outliers
    df = pd.DataFrame(dict(time = time, obs = obs))

    X = df["obs"].values.reshape(-1, 1)

    MLE = []  # Maximum likelihood estimation
    BIC = []  # Bayesian information criterion selection
    allmodels = []  # Store all models here

    if len(X) < 5:
        states = [1]
    elif init_means:
        states = [len(init_means)]
    else:
        states = [1, 2, 3, 4]

    for k in states:
        if k == 1:
            dists = [pg.NormalDistribution(means[0], state_noise)]

            trans_mat = np.array([[trans_prob]])

            starts, ends = _start_end(k)

            model = pg.HiddenMarkovModel.from_matrix(trans_mat, dists, starts, ends)

            loglik = model.log_probability(X)
            bic = _bic(X, k, loglik)

            MLE.append(loglik)
            allmodels.append(model)
            BIC.append(bic)

        if k == 2:
            means = _means_type(X, init_means, k_states = k)

            dists = [pg.NormalDistribution(means[0], state_noise),
                     pg.NormalDistribution(means[1], state_noise)]

            trans_mat = np.array([[self_trans, trans_prob],
                                  [trans_prob, self_trans]])

            starts, ends = _start_end(k)

            model = pg.HiddenMarkovModel.from_matrix(trans_mat, dists, starts, ends)

            loglik = model.log_probability(X)
            bic = _bic(X, k, loglik)

            MLE.append(loglik)
            allmodels.append(model)
            BIC.append(bic)

        if k == 3:
            means = _means_type(X, init_means, k_states = k)


            dists = [pg.NormalDistribution(means[0], state_noise),
                     pg.NormalDistribution(means[1], state_noise),
                     pg.NormalDistribution(means[2], state_noise)]

            trans_mat = np.array([[self_trans, trans_prob, trans_prob],
                                  [trans_prob, self_trans, trans_prob],
                                  [trans_prob, trans_prob, self_trans]])

            starts, ends = _start_end(k)

            model = pg.HiddenMarkovModel.from_matrix(trans_mat, dists, starts, ends)

            loglik = model.log_probability(X)
            bic = _bic(X, k, loglik)

            MLE.append(loglik)
            allmodels.append(model)
            BIC.append(bic)

        elif k == 4:
            means = _means_type(X, init_means, k_states = k)

            dists = [pg.NormalDistribution(means[0], state_noise),
                     pg.NormalDistribution(means[1], state_noise),
                     pg.NormalDistribution(means[2], state_noise),
                     pg.NormalDistribution(means[3], state_noise)]

            trans_mat = np.array([[self_trans, trans_prob, trans_prob, trans_prob],
                                  [trans_prob, self_trans, trans_prob, trans_prob],
                                  [trans_prob, trans_prob, self_trans, trans_prob],
                                  [trans_prob, trans_prob, trans_prob, self_trans]])

            starts, ends = _start_end(k)

            model = pg.HiddenMarkovModel.from_matrix(trans_mat, dists, starts, ends)

            loglik = model.log_probability(X)
            bic = _bic(X, k, loglik)

            MLE.append(loglik)
            allmodels.append(model)
            BIC.append(bic)

    if len(X) < 5:
        model = allmodels[0]
    elif init_means:
        model = allmodels[len(init_means) - 1]
    elif model_selection is "likelihood":
        model = allmodels[np.argmin(BIC)]
    elif model_selection is "bic":
        model = allmodels[np.argmax(MLE)]
    elif int(model_selection) in states:
        model = allmodels[model_selection + 1]
    else:
        raise ValueError("Model selection must be either 'likelihood', 'bic' or an integer with the number of states")

    hidden_states = model.predict(X)

    df["hmm_state"] = hidden_states
    df["unique_state_ID"] = df["hmm_state"].transform(lambda group: (group.diff() != 0).cumsum())

    # Calculate the mean ("idealized") obs for every state. The transform method puts the data back in the original tdp_df
    df["idealized"] = df.groupby(["hmm_state"], as_index = False)["obs"].transform("mean")

    idealized = df["idealized"]

    lifetimes = hmm_lifetimes(idealized_trace = df["idealized"],
                              time = df["time"],
                              min_trans = min_trans)

    return idealized, lifetimes


def hmm_lifetimes(idealized_trace, time, min_trans = 0.01, drop_extra_cols = False):
    """
    Get lifetimes from trace fits

    Parameters
    ----------
    idealized_trace:
        hmm fitted trace where each state has only one value
    min_trans:
        smallest transition to consider

    Returns
    -------
    Pandas dataframe with hmm_lifetimes and the associated transition points
    """

    # noinspection PyTypeChecker
    df = pd.DataFrame(dict(idealized = idealized_trace,
                           time = time))

    # Find E_after from E_before
    df["E_after"] = np.roll(df["idealized"], -1)

    # Find out when there's a change in state, depending on the minimum transition size set
    df["state_jump"] = df["idealized"].transform(lambda group: (abs(group.diff()) >= min_trans).cumsum())

    # Drop duplicates
    df = df.drop_duplicates(subset = "state_jump", keep = "last")

    # Find the difference for every time
    timedif = np.diff(df["time"])
    timedif = np.append(np.nan, timedif)  # Append nan to beginning

    df["lifetime"] = timedif

    df = df[1:]   # drop first observed lifetime
    df = df[:-1]  # drop last observed lifetime

    df.drop(["obs", "time", "hmm_state", "unique_state_ID", "state_jump"], axis = 1, inplace = True, errors = "ignore")
    df.rename(columns = {'idealized': 'E_before'}, inplace = True)

    lifetimes = df

    return lifetimes


def lh_fit(data, f, binned_likelihood, **kwargs):
    """
    Parameters
    ----------
    data: array
        unbinned data to fit
    f: function
        function to fit which returns the likelihood
    binned_likelihood: bool
        binned or unbinned likelihood
    kwargs:
        write parameters to fit like a (scalar), a_limit (range), fix_a (bool)

    Returns
    -------
    params: array
        array of estimated fit parameters
    errs: array
        array of estimated fit parameter errors
    loglh: scalar
        the minimized log likelihood
    """
    # Create an unbinned likelihood object with function and data.
    if binned_likelihood:
        minimize = probfit.BinnedLH(f, data)
    else:
        minimize = probfit.UnbinnedLH(f, data)

    # Minimizes the unbinned likelihood for the given function
    m = iminuit.Minuit(minimize,
                       **kwargs,
                       print_level = 0,
                       pedantic = False)
    m.migrad()

    params = np.array([val for val in m.values.values()])
    errs   = np.array([val for val in m.errors.values()])
    log_lh = np.sum(np.log(f(data, *params)))

    return params, errs, log_lh


def save_current_fig(output_name):
    os.makedirs("results/", exist_ok = True)
    output = "results/" + output_name + ".pdf"
    plt.savefig(output)
    plt.close()


def pick_random_traces(trace_df, n_traces, min_frames = "full"):
    """
    Parameters
    ----------
    trace_df:
        pandas dataframe containing all traces
    n_traces:
        number of traces to randomly select
    min_frames:
        at least one trace will fulfil this minimum length criterion.

    Returns
    -------
    dataframe with randomly selected traces
    """

    if min_frames is "full":
        min_frames = trace_df["time"].max()

    t = 0
    while t < min_frames:
        random_ids = np.random.choice(len(trace_df["id"].unique()), size = n_traces)
        random_traces = trace_df[trace_df["id"].isin(random_ids)]
        t = random_traces["time"].max()

    return random_traces, random_ids