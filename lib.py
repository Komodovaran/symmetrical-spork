import pomegranate as pg
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import os

pd.set_option("display.width", 1000)
pd.set_option('display.max_rows', 1000)
pd.options.mode.chained_assignment = None  # default='warn'


def flatten_list(input_list, as_array = False):
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

    # Initialize distributions from means and noise
    dists = [pg.NormalDistribution(m, noise) for m in means]

    # Generate HMM model
    model = pg.HiddenMarkovModel.from_matrix(transitions, distributions = dists, starts = starts)
    model.bake()

    fret_lst, time_lst, id_lst = [], [], []
    n = 0
    while n < n_traces:
        # Bleach time from single exp dist
        bleach = np.int(np.random.exponential(bleach_time, 1))

        if bleach < trace_min_len:
            continue

        # Generate samples, and remove datapoints after bleaching
        fret = np.array(model.sample(trace_max_len))[:bleach]
        time = np.array(range(0, len(fret)))[:bleach]

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
    os.makedirs("results/", exist_ok = True)

    output = "results/" + output_title + ".pickle"
    df.to_pickle(output, compression = None)


def save_current_fig(output_name):
    os.makedirs("results/", exist_ok = True)
    output = "results/" + output_name + ".pdf"
    plt.savefig(output)
    plt.close()