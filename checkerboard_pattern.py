from napatrackmater.homology import vr_entropy_all_frames, vr_entropy_generator, diagrams_over_time
from pathlib import Path 
import os
import matplotlib.pyplot as plt
import pandas as pd


dataset_name = 'Sixth'
home_folder = '/lustre/fsn1/projects/rech/jsy/uzj81mi/'
timelapse_to_track = f'timelapse_{dataset_name.lower()}_dataset'
tracking_directory = f'{home_folder}Mari_Data_Oneat/Mari_{dataset_name}_Dataset_Analysis/nuclei_membrane_tracking/'
channel = 'nuclei_'
data_frames_dir = os.path.join(tracking_directory, f'dataframes/')
save_dir = os.path.join(tracking_directory, f'{channel}checkerboard')
Path(save_dir).mkdir(exist_ok=True)

tracks_dataframe_path = os.path.join(data_frames_dir, f'results_dataframe_normalized_{channel}.csv')
tracks_dataframe = pd.read_csv(tracks_dataframe_path)

# --- diagrams over time + Betti-1 plot --------------------------------------
all_diagrams = diagrams_over_time(tracks_dataframe, max_dim=1)

betti_1 = [len(diag[1]) for diag in all_diagrams.values()]
plt.figure()
plt.plot(list(all_diagrams.keys()), betti_1)
plt.xlabel("time")
plt.ylabel(r"$\beta_1$")
plt.title(r"Betti-1 over Time")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, f"{channel}betti1_over_time.png"))
plt.close()

# --- one-shot: PE vector per time point --------------------------------------
t_sorted, diagrams, entropy = vr_entropy_all_frames(tracks_dataframe)

plt.figure()
plt.plot(t_sorted, entropy[:, 1])  # β₁ entropy
plt.xlabel('time')
plt.ylabel('PE (β₁)')
plt.title('Persistence Entropy (β₁)')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, f"{channel}persistence_entropy_beta1.png"))
plt.close()

# --- optional: print streaming values --------------------------------------
with open(os.path.join(save_dir, f"{channel}persistence_entropy_log.txt"), 'w') as f:
    for t, diag, ent_vec in vr_entropy_generator(tracks_dataframe):
        log_line = f"t={t:4d},  PE β₀={ent_vec[0]:.3f}, β₁={ent_vec[1]:.3f}, β₂={ent_vec[2]:.3f}\n"
        print(log_line, end='')
        f.write(log_line)