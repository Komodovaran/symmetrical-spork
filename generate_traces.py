import lib as lib
import numpy as np

lib.generate_traces(output_title  = "2_state_noise",
                    n_traces      = 1000,
                    noise         = 0.11,
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