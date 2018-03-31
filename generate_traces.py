import lib as lib
import numpy as np

# Note that initializing with the exact same parameters (except noise) will result in the
# exact same HMM traces and ordering, so they can be plotted directly on top of one another, per ID
SMALL_SAMPLE_TEST = True

np.random.seed(1)

if not SMALL_SAMPLE_TEST:
    print("Generating long traces.")
    lib.generate_traces(output_title  = "2_state_noise",
                        n_traces      = 1000,
                        noise         = 0.10,
                        bleach_time   = 70,
                        trace_min_len = 10,
                        trace_max_len = 200,
                        means         = (0.25, 0.75),
                        starts        = np.array([0.50, 0.50]),
                        transitions   = np.array([[0.75, 0.25],
                                                  [0.25, 0.75]]))

    lib.generate_traces(output_title  = "2_state_true",
                        n_traces      = 1000,
                        noise         = 0,
                        bleach_time   = 70,
                        trace_min_len = 10,
                        trace_max_len = 200,
                        means         = (0.25, 0.75),
                        starts        = np.array([0.50, 0.50]),
                        transitions   = np.array([[0.75, 0.25],
                                                  [0.25, 0.75]]))

    lib.generate_traces(output_title  = "2_state_true_nobleach",
                        n_traces      = 1000,
                        noise         = 0,
                        bleach_time   = False,
                        trace_min_len = 10,
                        trace_max_len = 200,
                        means         = (0.25, 0.75),
                        starts        = np.array([0.50, 0.50]),
                        transitions   = np.array([[0.75, 0.25],
                                                  [0.25, 0.75]]))

else:
    print("Generating short traces for testing purposes.")
    lib.generate_traces(output_title  = "2_state_noise",
                        n_traces      = 50,
                        noise         = 0.10,
                        bleach_time   = 70,
                        trace_min_len = 10,
                        trace_max_len = 200,
                        means         = (0.25, 0.75),
                        starts        = np.array([0.50, 0.50]),
                        transitions   = np.array([[0.75, 0.25],
                                                  [0.25, 0.75]]))

    lib.generate_traces(output_title  = "2_state_true",
                        n_traces      = 50,
                        noise         = 0,
                        bleach_time   = 70,
                        trace_min_len = 10,
                        trace_max_len = 200,
                        means         = (0.25, 0.75),
                        starts        = np.array([0.50, 0.50]),
                        transitions   = np.array([[0.75, 0.25],
                                                  [0.25, 0.75]]))


    lib.generate_traces(output_title  = "2_state_true_nobleach",
                        n_traces      = 50,
                        noise         = 0,
                        bleach_time   = False,
                        trace_min_len = 10,
                        trace_max_len = 200,
                        means         = (0.25, 0.75),
                        starts        = np.array([0.50, 0.50]),
                        transitions   = np.array([[0.75, 0.25],
                                                  [0.25, 0.75]]))


print("Traces have been generated. Remember to re-run plots and fits.")