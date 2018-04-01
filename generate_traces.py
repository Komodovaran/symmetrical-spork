import lib as lib
import numpy as np
np.random.seed(1)

# Note that initializing with the exact same parameters (except noise) will result in the
# exact same HMM traces and ordering, so they can be plotted directly on top of one another, per ID

SMALL_SAMPLE_TEST = True

n_traces_full = 300
n_traces_test = 15
noise_level   = 0.10
bleach_time   = 70
trace_min_len = 10
trace_max_len = 200

means         = (0.25, 0.75)
starts        = np.array([0.50, 0.50])

slow_transitions = np.array([[0.97, 0.03],
                             [0.03, 0.97]])

fast_transitions = np.array([[0.75, 0.25],
                             [0.25, 0.75]])

# Please follow this format, or stuff will break
outputs = ["2S_state_noise",
           "2S_state_true",
           "2S_state_true_nobleach",
           "2F_state_noise",
           "2F_state_true",
           "2F_state_true_nobleach"]


if SMALL_SAMPLE_TEST:
    print("Generating short traces.")
    n_traces = n_traces_test
else:
    print("Generating long traces.")
    n_traces = n_traces_full

for n, title in enumerate(outputs):
    trans_freq = title.split("_")[0][1]
    if trans_freq == "S":
        transitions = slow_transitions
    elif trans_freq == "F":
        transitions = fast_transitions
    else:
        raise ValueError("Invalid name encountered")

    title_end = title.split("_")[-1]
    if title_end == "noise":
        noise = noise_level
    else:
        noise = 0

    lib.generate_traces(output_title  = title,
                        n_traces      = n_traces,
                        noise         = noise,
                        bleach_time   = bleach_time,
                        trace_min_len = trace_min_len,
                        trace_max_len = trace_max_len,
                        means         = means,
                        starts        = starts,
                        transitions   = transitions)