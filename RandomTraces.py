import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


plot_traces = True
plot_stats  = True
add_noise   = True


np.random.seed(1)

#####################################
# Set parameters
#####################################

n_traces         = 50

true_states      = [0.3, 0.5, 0.7, 0.9]  # True underlying states
true_occupancies = [0.2, 0.3, 0.4, 0.1]  # True occupancies (must sum up to one)

bleach_time      = 20                    # Average bleaching time of donor/acceptor
recording_time   = 60                    # Total recording time
energy_barrier   = 5                    # Scales the dwell time in each state

noise_level      = 0.05                 # Noise level

#####################################
# Generate traces
#####################################

df_traces = pd.DataFrame()
bleaching_time_lst = []

for i in tqdm(range(n_traces)):

    # Reset frame and lists
    frame     = 0
    frame_lst = []
    fret_lst  = []
    id_lst = []

    # Decide bleaching time
    bleaching_time = float(np.random.exponential(scale=bleach_time, size=1))

    if bleaching_time > recording_time:
        bleaching_time = recording_time

    while frame < bleaching_time:

        # Decide state
        fret = np.random.choice(a=true_states, p=true_occupancies)    # NB all transitions are are allowed (including self-transitions)

        # Decide dwell time
        occupancy = true_occupancies[true_states.index(fret)]
        dwell_time = energy_barrier * float(np.random.exponential(scale=occupancy, size=1))

        for n in range(int(dwell_time)):
            if add_noise:
                # Generate random noise
                noise = np.random.normal(loc=0, scale=noise_level)
            else:
                noise = 0

            frame_lst.append(frame)
            fret_lst.append(fret + noise)
            id_lst.append(i)

            frame +=1

    while frame < recording_time:
        if add_noise:
            # Generate random noise
            noise = np.random.normal(loc=0, scale=noise_level)
        else:
            noise = 0

        frame_lst.append(frame)
        fret_lst.append(noise)
        id_lst.append(i)

        frame += 1

    # Append fret values to data frame
    data_tuples = list(zip(id_lst, frame_lst, fret_lst))
    df_temp = pd.DataFrame(data_tuples, columns=["ID", "frame", "FRET value"])

    df_traces = pd.concat([df_traces, df_temp], ignore_index=True)

    # Append bleaching time to list
    bleaching_time_lst.append(bleaching_time)

    # Plot trace
    if plot_traces:
        plt.figure(figsize = (6, 3))
        plt.plot(frame_lst, fret_lst)
        plt.xlabel("Frame")
        plt.ylabel("FRET")
        plt.xlim(0, recording_time)
        plt.ylim(-0.1, 1.1)
        plt.tight_layout()
        plt.savefig("traces/random_trace_{:d}.pdf".format(i))
        plt.close()

# Export traces to csv file
df_traces.to_csv("traces/traces.csv")


#####################################
# Plot other stuff
#####################################

if plot_stats:
    plt.figure()
    plt.hist(bleaching_time_lst, bins=15, label="N_traces = {:d}".format(len(bleaching_time_lst)))
    plt.xlabel("Bleaching time (frames)")
    plt.ylabel("#")
    plt.legend()
    plt.savefig("stats/trace_duration.pdf")

