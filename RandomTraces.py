import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm
import os

np.random.seed(1)

def fret_trace_generator(n_traces              = 50,
                        fret_states            = [0.2, 0.4, 0.6, 0.8],  # True fret states
                        transition_probability = [0.2, 0.4, 0.3, 0.1],  # Probabilities must sum up to one
                        energy_barrier         = 20,                    # Scales the dwell times
                        bleach_time            = 20,                    # Average bleaching time of donor/acceptor
                        recording_time         = 100,                   # Total recording time
                        noise_level            = 0.05,                  # Noise level
                        include_bleaching      = False,                 # Generate random noise after bleaching
                        plot_traces            = False,
                        plot_other             = True):

    df_traces = pd.DataFrame()
    bleaching_time_lst = []

    for trace in tqdm(range(n_traces)):

        # Reset frame and lists
        frame           = 0
        fret            = 0     # initial fret value
        id_lst          = []
        frame_lst       = []
        fret_lst        = []
        true_fret_lst   = []

        # Decide bleaching time
        bleaching_time = float(np.random.exponential(scale=bleach_time))

        if bleaching_time > recording_time:
            bleaching_time = recording_time

        # Append bleaching time to list
        bleaching_time_lst.append(bleaching_time)

        # Generate fret values
        while frame < bleaching_time:

            # Suggest fret state
            fret_temp = np.random.choice(a=fret_states, p=transition_probability)  # NB. All transitions are are allowed

            # Check for self-transition (NB. Inefficient code, could be optimized)
            while fret_temp == fret:
                fret_temp = np.random.choice(a=fret_states, p=transition_probability)

            # Set fret state
            fret = fret_temp

            # Decide dwell time
            # NB. This part of algorithm should be changed!
            state_probability = transition_probability[fret_states.index(fret)]
            dwell_time = float(np.random.exponential(scale=energy_barrier)) #* state_probability

            for n in range(int(dwell_time)):
                if frame < bleaching_time:
                    # Generate random noise
                    noise = np.random.normal(loc=0, scale=noise_level)

                    # Append to lists
                    frame_lst.append(frame)
                    fret_lst.append(fret + noise)
                    true_fret_lst.append(fret)
                    id_lst.append(trace)

                    frame +=1
                else:
                    break

        if include_bleaching:
            while frame < recording_time:
                # Generate random noise
                noise = np.random.normal(loc=0, scale=noise_level)

                # Append to lists
                frame_lst.append(frame)
                fret_lst.append(noise)
                true_fret_lst.append(0)
                id_lst.append(trace)

                frame += 1

        # Create data frame from lists
        data_tuples = list(zip(id_lst, frame_lst, fret_lst, true_fret_lst))
        df_temp = pd.DataFrame(data_tuples, columns=["ID", "Frame", "FRET (observed)", "FRET (true)"])

        # Concatenate date frames
        df_traces = pd.concat([df_traces, df_temp], ignore_index=True)

        # Plot traces
        if plot_traces:
            os.makedirs("results/plot_traces", exist_ok=True)

            plt.figure(figsize=(6, 3))
            plt.plot(frame_lst, fret_lst, label="Observed")
            plt.plot(frame_lst, true_fret_lst, linestyle="--", linewidth=0.8, color="firebrick", label="True")
            plt.xlabel("Frame")
            plt.ylabel("FRET")
            plt.xlim(0, recording_time)
            plt.ylim(-0.1, 1.1)
            plt.legend(loc="upper right")
            plt.tight_layout()
            plt.savefig("results/plot_traces/random_trace_{:d}.pdf".format(trace))
            plt.close()

    # Export traces to csv
    os.makedirs("results", exist_ok=True)

    df_traces.to_csv("results/traces.csv")

    # Plot other stuff
    if plot_other:
        os.makedirs("results", exist_ok=True)

        # Bleaching time histogram
        plt.figure()
        plt.hist(bleaching_time_lst, bins=15, label="N_traces = {:d}".format(len(bleaching_time_lst)))
        plt.xlabel("Bleaching time (frames)")
        plt.ylabel("#")
        plt.legend()
        plt.savefig("results/bleaching_times.pdf")

        # E FRET histogram
        hist_binwidths = np.arange(-0.1, 1.1, 0.01)

        plt.figure()
        plt.hist(df_traces["FRET (observed)"], bins=hist_binwidths, normed=True)
        plt.xlabel("FRET")
        plt.ylabel("Probability density")
        plt.xlim(-0.1, 1.1)

        # Underlying gaussian distributions
        x_pts = np.linspace(-0.1, 1.1, 1000)
        multi_gauss = 0

        for i in range(len(fret_states)):
            gauss = stats.norm.pdf(x_pts, loc=fret_states[i], scale=noise_level)
            weight = transition_probability[i]

            y_pts = weight * gauss
            multi_gauss += y_pts
            plt.plot(x_pts, y_pts, alpha=0.8)

        plt.plot(x_pts, multi_gauss, color="black", label="True distribution")

        plt.legend()
        plt.savefig("results/FRET_histogram.pdf")
        plt.close()

    return

fret_trace_generator(n_traces=1000)