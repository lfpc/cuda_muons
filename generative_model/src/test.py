import torch
import h5py


data_dir = 'data/'
material = 'Fe'
data_file = data_dir + f'muon_data_energy_loss_sens_G4_{material}.h5'
with h5py.File(data_file, 'r') as f:
    p_bins = list(f.keys())
    print(list(f[p_bins[0]].keys()))
