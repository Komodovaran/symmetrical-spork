import pomegranate as pg
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import os
from tqdm import tqdm

def flatten_list(input_list, as_array = False):
    flat_lst = list(itertools.chain.from_iterable(input_list))
    if as_array:
        flat_lst = np.array(flat_lst)
    return flat_lst

def generate_traces(output_title, transitions, starts, means, noise = 0.04, n_traces = 100, trace_max_length = 200, bleach_time = 40):
    # Initialize distributions from means and noise
    dists = []
    for m in means:
        dists.append(pg.NormalDistribution(m, noise))

    fret_lst, time_lst, id_lst = [], [], []
    for i in tqdm(range(n_traces)):
        # Bleach time from single exp dist
        bleach = np.int(np.random.exponential(bleach_time, 1))

        # Generate HMM model
        model = pg.HiddenMarkovModel.from_matrix(transitions, distributions = dists, starts = starts)
        model.bake()

        # Generate samples, and remove datapoints after bleaching
        fret = np.array(model.sample(trace_max_length))
        time = np.array(range(0, len(fret)))

        fret = fret[:bleach]
        time = time[:bleach]

        # Append results
        fret_lst.append(fret)
        time_lst.append(time)
        id_lst.append([i]*len(fret))

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


case1 = dict(output_title     = "case 1",
             n_traces         = 1000,
             bleach_time      = 50,
             trace_max_length = 200,
             means            = (0.3, 0.8),
             starts           = np.array([0.50, 0.50]),
             transitions      = np.array([[0.9, 0.1],
                                          [0.1, 0.9]]))

generate_traces(**case1)
