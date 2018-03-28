import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


np.random.seed(1)

def fret_trace_generator(n_traces            = 50,
                        true_states          = [0.3, 0.5, 0.7, 0.9],  # True underlying states
                        true_occupancies     = [0.2, 0.3, 0.4, 0.1],  # True occupancies (must sum up to one)
                        bleach_time          = 20,                    # Average bleaching time of donor/acceptor
                        recording_time       = 60,                    # Total recording time
                        energy_barrier       = 5,                     # Scales the dwell times
                        noise_level          = 0.05,                  # Noise level
                        plot_traces          = True,
                        plot_bleaching_times = True):

    df_traces = pd.DataFrame()
    bleaching_time_lst = []

    for trace in tqdm(range(n_traces)):

        # Reset frame and lists
        frame           = 0
        id_lst          = []
        frame_lst       = []
        fret_lst        = []
        true_fret_lst   = []

        # Decide bleaching time
        bleaching_time = float(np.random.exponential(scale=bleach_time))

        if bleaching_time > recording_time:
            bleaching_time = recording_time

        while frame < bleaching_time:

            # Decide state
            fret = np.random.choice(a=true_states, p=true_occupancies)    # NB. All transitions are are allowed (including self-transitions)

            # Decide dwell time
            true_occupancy   = true_occupancies[true_states.index(fret)]
            dwell_time = energy_barrier * float(np.random.exponential(scale=true_occupancy))

            for n in range(int(dwell_time)):
                # Generate random noise
                noise = np.random.normal(loc=0, scale=noise_level)

                frame_lst.append(frame)
                fret_lst.append(fret + noise)
                true_fret_lst.append(fret)
                id_lst.append(trace)

                frame +=1

        while frame < recording_time:
            # Generate random noise
            noise = np.random.normal(loc=0, scale=noise_level)

            frame_lst.append(frame)
            fret_lst.append(noise)
            true_fret_lst.append(0)
            id_lst.append(trace)

            frame += 1

        # Append fret values to data frame
        data_tuples = list(zip(id_lst, frame_lst, fret_lst, true_fret_lst))
        df_temp = pd.DataFrame(data_tuples, columns=["ID", "Frame", "FRET (observed)", "FRET (true)"])

        df_traces = pd.concat([df_traces, df_temp], ignore_index=True)

        # Append bleaching time to list
        bleaching_time_lst.append(bleaching_time)

        # Plot traces
        if plot_traces:
            os.makedirs("random_traces", exist_ok=True)

            plt.figure(figsize=(6, 3))
            plt.plot(frame_lst, fret_lst, label="Observed")
            plt.plot(frame_lst, true_fret_lst, linestyle="--", linewidth=0.8, color="firebrick", label="True")
            plt.xlabel("Frame")
            plt.ylabel("FRET")
            plt.xlim(0, recording_time)
            plt.ylim(-0.1, 1.1)
            plt.legend()
            plt.tight_layout()
            plt.savefig("random_traces/random_trace_{:d}.pdf".format(trace))
            plt.close()

    # Export traces to csv file
    os.makedirs("random_traces", exist_ok=True)

    df_traces.to_csv("random_traces/random_traces.csv")

    # Distribution of bleaching times
    if plot_bleaching_times:
        os.makedirs("random_traces", exist_ok=True)

        plt.figure()
        plt.hist(bleaching_time_lst, bins=15, label="N_traces = {:d}".format(len(bleaching_time_lst)))
        plt.xlabel("Bleaching time (frames)")
        plt.ylabel("#")
        plt.legend()
        plt.savefig("random_traces/bleaching_times.pdf")

    return


fret_trace_generator(n_traces=50)