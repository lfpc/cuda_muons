import time
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import argparse
import torch
torch.set_default_dtype(torch.float64)

parser = argparse.ArgumentParser(description="Run muon simulations.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--material', type=str, default='G4_Fe', help='Material to use for simulations (G4_Fe, G4_CONCRETE)')
parser.add_argument('--alias', action='store_true', help='Build also alias histograms')
args = parser.parse_args()

material = args.material
# Ensure all necessary directories exist


os.makedirs("plots/hists_geant4", exist_ok=True)

def compute_2d_histo(array_a, array_b, edges_a, edges_b):
    t1 = time.time()
    hist2d, _, _ = np.histogram2d(array_a, array_b, bins=[edges_a, edges_b])
    print("Took", time.time() - t1, "seconds")
    print(hist2d.shape)
    return hist2d


# Load H5 data
h5_filename = f"data/muon_data_energy_loss_sens_{material}.h5"
plotting_enabled = True

nbins = 100
min_bin = -22


def get_min_max_bin(h5_filename):
    min_dpz = float('inf')
    max_dpz = float('-inf')
    min_secondary = float('inf')
    max_secondary = float('-inf')
    with h5py.File(h5_filename, 'r') as f:
        for energy_seg in f.keys():
            pz = f[energy_seg]['pz'][:]
            px = f[energy_seg]['px'][:]
            py = f[energy_seg]['py'][:]
            initial_momenta = f[energy_seg]['initial_momenta'][:]

            delta_pz_f = np.log(np.abs((pz - initial_momenta) / initial_momenta))
            delta_pz_f = np.where(np.isnan(delta_pz_f), min_bin, delta_pz_f)
            delta_pz_f = np.where(delta_pz_f == float('-inf'), min_bin, delta_pz_f)
            delta_pz_f = np.clip(delta_pz_f, min_bin, 0)
            min_dpz = min(min_dpz, np.min(delta_pz_f))
            max_dpz = max(max_dpz, np.max(delta_pz_f))

            secondary_f = np.log(np.sqrt(px**2 + py**2) / initial_momenta)
            secondary_f = np.where(np.isnan(secondary_f), min_bin, secondary_f)
            secondary_f = np.where(secondary_f == float('-inf'), min_bin, secondary_f)
            secondary_f = np.clip(secondary_f, min_bin, 0)
            min_secondary = min(min_secondary, np.min(secondary_f))
            max_secondary = max(max_secondary, np.max(secondary_f))
    min_dpz = np.floor(min_dpz * 10) / 10
    max_dpz = np.ceil(max_dpz * 10) / 10
    min_secondary = np.floor(min_secondary * 10) / 10
    max_secondary = np.ceil(max_secondary * 10) / 10
    max_dpz = max(max_dpz, 0.0)
    max_secondary = max(max_secondary, 0.0)
    return min_dpz, max_dpz, min_secondary, max_secondary
min_delta_pz_f, max_delta_pz_f, min_secondary_f, max_secondary_f = get_min_max_bin(h5_filename)
print(f"Min/Max delta_pz_f: {min_delta_pz_f}, {max_delta_pz_f}; Min/Max secondary_f: {min_secondary_f}, {max_secondary_f}")
edges_dpz = np.linspace(min_delta_pz_f, max_delta_pz_f, nbins + 1)
edges_secondary = np.linspace(min_secondary_f, max_secondary_f, nbins + 1)

min_delta_pz_f = 0
min_secondary_f = 0

hists_dict = {}
with h5py.File(h5_filename, 'r') as f:
    step_length = f.attrs['step_length']/100
    hists_dict['step_length'] = step_length
    for energy_seg in sorted(
        f.keys(),
        key=lambda k: float(k.strip("()").split(",")[0])):

        print(f"Processing energy segment {energy_seg} GeV")
        px = f[energy_seg]['px'][:]
        py = f[energy_seg]['py'][:]
        pz = f[energy_seg]['pz'][:]
        initial_momenta = f[energy_seg]['initial_momenta'][:]

        # Filtered values for delta_pz_f and delta_px_f
        assert (pz <= initial_momenta).all(), "Some final pz values are greater than initial momenta"
        delta_pz_f_filt = np.log(np.abs(pz - initial_momenta) / initial_momenta)
        #delta_pz_f_filt = np.clip(delta_pz_f_filt, min_bin, 0)
        delta_pz_f_filt = np.where(np.isnan(delta_pz_f_filt), min_bin, delta_pz_f_filt)
        delta_pz_f_filt = np.where(delta_pz_f_filt == float('-inf'), min_bin, delta_pz_f_filt)
    
        secondary_f_filt = np.log(np.sqrt(px**2 + py**2) / initial_momenta)
        secondary_f_filt = np.where(np.isnan(secondary_f_filt), min_bin, secondary_f_filt)
    
        total_loss_mag = np.log(np.sqrt(px**2 + py**2 + (pz - initial_momenta)**2) / initial_momenta)
    
        # Compute histograms
        hist_dpz, _ = np.histogram(delta_pz_f_filt, bins=edges_dpz)
        hist_secondary, _ = np.histogram(secondary_f_filt, bins=edges_secondary)

        hist_2d = compute_2d_histo(delta_pz_f_filt, secondary_f_filt, edges_dpz, edges_secondary)

        energy_seg = tuple(round(float(x.strip()), 4) for x in energy_seg.strip("()").split(","))
        assert np.all((energy_seg[0]-0.0001 < initial_momenta) & (initial_momenta < energy_seg[1]+0.0001)), f"Initial momenta not in energy segment range. Min: {initial_momenta.min()}, Max: {initial_momenta.max()}, Segment: {energy_seg}"
        hists_dict[energy_seg] = {
            #'hist_dpz': hist_dpz,
            #'hist_secondary': hist_secondary,
            'hist_2d': hist_2d,
            'edges_dpz': edges_dpz,
            'edges_secondary': edges_secondary
        }
    
        if plotting_enabled:
            fig, ax = plt.subplots(1, 2, figsize=(12, 3.2))
            fig.subplots_adjust(
                top=0.9,
                bottom=0.1,
                left=0.1,
                right=0.9,
                hspace=0.35,
                wspace=0.4
            )
            ax[0].grid(True)
            ax[1].grid(True)
            
            fig.suptitle('        ')
            
            ax[0].stairs(hist_dpz, edges_dpz, label='%.0f < $p_z$ [GeV] < %.0f'%(energy_seg[0], energy_seg[1]), color='firebrick', zorder=5)
            ax[1].stairs(hist_secondary, edges_secondary, label='%.0f < $p_z$ [GeV] < %.0f'%(energy_seg[0], energy_seg[1]), color='firebrick', zorder=5)

            ax[0].legend(loc='upper right', bbox_to_anchor=(1.0, 1.15))
            ax[1].legend(loc='upper right', bbox_to_anchor=(1.0, 1.15))
            
            ax[0].set_ylabel('Frequency (arb.)')
            ax[1].set_ylabel('Frequency (arb.)')
            
            ax[0].set_xlabel(r'$\log\left(\frac{\Delta p_z}{\text{initial } p_z}\right)$')
            ax[1].set_xlabel(r'$\log\left(\frac{\sqrt{p_x^2 + p_y^2}}{\text{initial } p_z}\right)$')
            
            ax[0].set_yscale('log')
            ax[1].set_yscale('log')
            
            print("Plotting...")
            plt.savefig(f'plots/hists_geant4/{energy_seg[0]}-{energy_seg[1]}_dpz_{material}.pdf', bbox_inches='tight')
            plt.close()

# Save histograms
with open(f"data/histograms_{material}.pkl", "wb") as file:
    pickle.dump(hists_dict, file)
print("Histograms built and saved.")


def alias_setup(histogram):
    N = histogram.shape[0]
    n = histogram.shape[1]
    prob_table = torch.zeros_like(histogram)
    alias_table = torch.zeros(histogram.shape, dtype=torch.int32)

    # Normalize probabilities and scale by n
    normalized_prob = histogram*n

    for j in range(N):
        small = []
        large = []

        # Separate bins into small and large
        for i, p in enumerate(normalized_prob[j]):
            if p < 1:
                small.append(i)
            else:
                large.append(i)

        # Distribute probabilities between small and large bins
        while small and large:
            small_bin = small.pop()
            large_bin = large.pop()

            prob_table[j][small_bin] = normalized_prob[j][small_bin]
            alias_table[j][small_bin] = large_bin

            # Adjust the large bin's probability
            normalized_prob[j][large_bin] = (normalized_prob[j][large_bin] +
                                          normalized_prob[j][small_bin] - 1)
            if normalized_prob[j][large_bin] < 1:
                small.append(large_bin)
            else:
                large.append(large_bin)

        # Fill remaining bins
        for remaining in large + small:
            prob_table[j][remaining] = 1
            alias_table[j][remaining] = remaining

    return prob_table, alias_table

if args.alias:
    alias_save_dir = f"data/alias_histograms_{material}.pkl"

    step_length = hists_dict.pop('step_length')
    energy_bins = list(hists_dict.keys())
    keys = hists_dict[energy_bins[0]].keys()
    n_energy_bins = len(energy_bins)
    print(f"Processing {n_energy_bins} energy bins vectorized")
    stored_hist_data = {key:torch.tensor([hists_dict[energy_bin][key].tolist() for energy_bin in energy_bins]) for key in keys}

    edges_dpz = stored_hist_data['edges_dpz'][0]
    edges_secondary = stored_hist_data['edges_secondary'][0]
    centers_dpz = (edges_dpz[:-1] + edges_dpz[1:]) / 2
    centers_secondary = (edges_secondary[:-1] + edges_secondary[1:]) / 2
    widths_dpz = edges_dpz[1:] - edges_dpz[:-1]
    widths_secondary = edges_secondary[1:] - edges_secondary[:-1]
    hist_2d_all = stored_hist_data['hist_2d'].reshape((-1))

    n_bins_dpz = len(centers_dpz)
    n_bins_secondary = len(centers_secondary)
    assert n_bins_dpz == n_bins_secondary
    hist_2d_bin_centers_first_dim = centers_dpz.repeat_interleave(n_bins_secondary)
    hist_2d_bin_centers_second_dim = centers_secondary.repeat(n_bins_dpz)
    hist_2d_bin_widths_first_dim = widths_dpz.repeat_interleave(n_bins_secondary)
    hist_2d_bin_widths_second_dim = widths_secondary.repeat(n_bins_dpz)

    hist_2d_all = hist_2d_all.reshape((-1,100*100))
    hist_2d_all = torch.nn.functional.normalize(hist_2d_all, p=1, dim=1)


    t0 = time.time()
    hist_2d_probability_table, hist_2d_alias_table = alias_setup(hist_2d_all)
    print(f"Alias setup took {time.time() - t0:.4f} seconds")

    output = {
        'hist_2d_probability_table': hist_2d_probability_table,
        'hist_2d_alias_table': hist_2d_alias_table,
        'centers_first_dim': hist_2d_bin_centers_first_dim,
        'centers_second_dim': hist_2d_bin_centers_second_dim,
        'width_first_dim': hist_2d_bin_widths_first_dim,
        'width_second_dim': hist_2d_bin_widths_second_dim,
        'step_length': step_length
    }
    for k, v in output.items():
        if k == 'step_length':
            print(f"{k}: value={v}")
        else:
            print(f"{k}: shape={v.shape}")

    with open(alias_save_dir, 'wb') as f:
        pickle.dump(output, f)
    print("Data saved to", alias_save_dir)




