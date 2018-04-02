import lib as lib
import numpy as np
np.random.seed(1)

# Note that initializing with the exact same parameters (except noise) will result in the
# exact same HMM traces (for one particular set of transition parameters) and ordering,
# so they can be plotted directly on top of one another if the same ID is used everywhere

lib.generate_traces(SMALL_SAMPLE_TEST = True,
                    n_traces_full     = 300,
                    n_traces_test     = 10,
                    noise_level       = 0.10,
                    bleach_time       = 40,
                    trace_min_len     = 10,
                    trace_max_len     = 200,
                    state_means       = (0.25, 0.75),
                    slow_trans_mtrx   = np.array([[0.90, 0.10],
                                                  [0.10, 0.90]]),
                    fast_trans_mtrx   = np.array([[0.75, 0.25],
                                                  [0.25, 0.75]]),
                    labels            = ["2S_state_noise",
                                         "2S_state_true",
                                         "2S_state_true_nobleach",
                                         "2F_state_noise",
                                         "2F_state_true",
                                         "2F_state_true_nobleach"])