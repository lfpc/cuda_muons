import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

from muon_slabs import simulate_muon, initialize, kill_secondary_tracks, collect_from_sensitive
import json
import lib.gigantic_sphere as sphere_design
from cuda_muons import propagate_muons_with_cuda
import argparse
import multiprocessing as mp
import torch    
import h5py
import time


sensitive_film = {"name": "SensitiveFilm", "dz": 0.0001, "dx": 20, "dy": 20,"shape": "plane"}

def run_geant4_sim(muons, n_steps=500, mag_field=[0., 0., 0.], material = 'G4_Fe'):
    sensitive_film['z_center'] = 0.02*n_steps
    np.random.seed((os.getpid() * int(time.time())) % 2**16)
    detector = sphere_design.get_design(mag_field=mag_field, sens_film=sensitive_film, material=material)
    detector["store_primary"] = False
    detector["store_all"] = False
    initialize(np.random.randint(32000), np.random.randint(32000), np.random.randint(32000), np.random.randint(32000), json.dumps(detector))
    kill_secondary_tracks(True)
    results = {
        'px': [],
        'py': [],
        'pz': [],
        'x': [],
        'y': [],
        'z': [],
        'step_length': [],
    }
    for muon in muons:
        simulate_muon(*muon[:3], -1, *muon[3:6])
        data = collect_from_sensitive()
        index = 0
        if len(data['z'])>0 and 13 in np.abs(data['pdg_id']): 
            while int(abs(data['pdg_id'][index])) != 13:
                index += 1
        else: continue

        results['px'].append(data['px'][index])
        results['py'].append(data['py'][index])
        results['pz'].append(data['pz'][index])
        results['x'].append(data['x'][index])
        results['y'].append(data['y'][index])
        results['z'].append(data['z'][index])
        #results['step_length'].append(data['step_length'][index])
    return results

def run_cuda_simulation(muons, n_steps=500, mag_field=[0., 0., 0.],histogram_dir = 'data', material = 'G4_Fe'):
    muons = torch.from_numpy(muons)
    muons_momenta = muons[:,:3]
    muons_positions = muons[:,3:6]
    muons_charge = muons[:,6]

    if material == 'G4_Fe':
        hist_file = os.path.join(histogram_dir, 'alias_histograms_G4_Fe.pkl')
    elif material == 'G4_CONCRETE':
        hist_file = os.path.join(histogram_dir, 'alias_histograms_G4_CONCRETE.pkl')

    with open(hist_file, 'rb') as f:
        hist_data = pickle.load(f)
        histograms = [hist_data['hist_2d_probability_table'],hist_data['hist_2d_alias_table'],
                        hist_data['centers_first_dim'], hist_data['centers_second_dim'],
                        hist_data['width_first_dim'], hist_data['width_second_dim']]
        step_length = hist_data['step_length']
    histograms_sec = [torch.zeros_like(s) for s in histograms]

    corners = torch.tensor([[-100, -100, -100], [100, -100, -100], [100, 100, -100], [-100, 100, -100],
                             [-100, -100, 100], [100, -100, 100], [100, 100, 100], [-100, 100, 100]],
                            dtype=torch.float32).reshape(1, 8, 3)
    mag_field = np.tile(np.asarray(mag_field), (12, 12, 12, 1))
    mag_dict = {'B': mag_field,'range_x':[-10_00,100_00,10_00], 'range_y':[-10_00,100_00,10_00], 'range_z':[-10_00,100_00,10_00]}

    out_position, out_momenta = propagate_muons_with_cuda (
            muons_positions,
            muons_momenta,
            muons_charge,
            corners,
            torch.tensor([[-30, 30, -30, 30, 0], [-30, 30, -30, 30, 0]], dtype=torch.float32),
            *histograms,
            *histograms_sec,
            mag_dict,
            n_steps*0.02 - step_length/2,
            n_steps * 2,
            step_length,
        )
    if n_steps > 1:
        in_sens_plane = (out_position[:,0].abs() < sensitive_film['dx']/2) & \
                        (out_position[:,1].abs() < sensitive_film['dy']/2) & \
                        (out_position[:,2] >= (n_steps*0.02 - step_length/2))

        out_momenta = out_momenta[in_sens_plane]
        out_position = out_position[in_sens_plane]
        muons_charge = muons_charge[in_sens_plane].int()
    print("Number of outputs:", out_momenta.shape[0])


    out_position = out_position.cpu().numpy()
    out_momenta = out_momenta.cpu().numpy()
    output = {
        'px': out_momenta[:,0],
        'py': out_momenta[:,1],
        'pz': out_momenta[:,2],
        'x': out_position[:,0],
        'y': out_position[:,1],
        'z': out_position[:,2],
        'pdg_id': muons_charge*(-13)
    }
    return output

def propagate_muons_until_z(px,py,pz,x,y,z, z_target:float):
    """ Propagate muons in a straight line until they reach z_target """
    dz = z_target - z
    y += dz*py/pz
    x += dz*px/pz
    z = np.ones_like(z)*z_target
    return x,y,z

def main(muons, n_steps=500, mag_field=[0., 0., 0.], material = 'G4_Fe'):
    file_cuda = 'data/outputs_cuda.pkl'
    file_geant4 = 'data/outputs_geant4.pkl'
    outputs_cuda = run_cuda_simulation(muons, mag_field = mag_field, histogram_dir=f'data/', n_steps=n_steps, material=material)
    n_cores = 90
    muons_split = np.array_split(muons, n_cores)
    with mp.Pool(n_cores) as pool:
        geant4_results = pool.starmap(run_geant4_sim, [(muon_batch, n_steps, mag_field, material) for muon_batch in muons_split])
    print("Geant4 simulation completed.")
    outputs_geant4 = {}
    for key in geant4_results[0].keys():
        outputs_geant4[key] = np.concatenate([np.array(res[key]) for res in geant4_results], axis=0)
    print("CUDA simulation completed.")
    print("Data collection completed.")
    with open(file_cuda, 'wb') as f:
        pickle.dump(outputs_cuda, f)
    print("CUDA data saved to ", file_cuda)
    with open(file_geant4, 'wb') as f:
        pickle.dump(outputs_geant4, f)
    print("Geant4 data saved to ", file_geant4)
    return outputs_geant4, outputs_cuda



import numpy as np
import matplotlib.pyplot as plt
import os

def filter_data(px, py, pz, x, y, z, weights):
    """Filters the data based on momentum and sensor plane conditions."""
    p_mag = np.sqrt(px**2 + py**2 + pz**2)
    
    # Apply momentum filter
    mask = (p_mag >= args.filter_p[0]) & (p_mag <= args.filter_p[1])
    
    # Apply sensor plane filter if enabled
    if args.sens_plane:
        mask &= (np.abs(x) <= 2) & (np.abs(y) <= 3)
        
    # Return filtered data
    if weights is not None:
        return px[mask], py[mask], pz[mask], x[mask], y[mask], z[mask], weights[mask]
    else:
        return px[mask], py[mask], pz[mask], x[mask], y[mask], z[mask], None

def plot_histograms(output_filename, 
                    px_g4, py_g4, pz_g4, x_g4, y_g4, z_g4,
                    px_cuda, py_cuda, pz_cuda, x_cuda, y_cuda, z_cuda, weights_g4 = None, weights_cuda = None):


    px_g4, py_g4, pz_g4, x_g4, y_g4, z_g4, weights_g4 = filter_data(
        px_g4, py_g4, pz_g4, x_g4, y_g4, z_g4, weights_g4)
    
    px_cuda, py_cuda, pz_cuda, x_cuda, y_cuda, z_cuda, weights_cuda = filter_data(
        px_cuda, py_cuda, pz_cuda, x_cuda, y_cuda, z_cuda, weights_cuda)
    
    p_mag_g4 = np.sqrt(px_g4 ** 2 + py_g4 ** 2 + pz_g4 ** 2)
    p_mag_cuda = np.sqrt(px_cuda ** 2 + py_cuda ** 2 + pz_cuda ** 2)

    print(f"Number of GEANT4 samples after filter: {px_g4.shape[0]}")
    if weights_g4 is not None:
        print(f"Sum of GEANT4 weights after filter: {np.sum(weights_g4)}")
    print(f"Number of CUDA samples after filter: {px_cuda.shape[0]}")
    if weights_cuda is not None:
        print(f"Sum of CUDA weights after filter: {np.sum(weights_cuda)}")

    print(f"Difference in number of muons in output:", np.abs(px_g4.shape[0] - px_cuda.shape[0]) / px_g4.shape[0] * 100, "%")

    p_transverse_g4 = np.sqrt(px_g4 ** 2 + py_g4 ** 2)
    p_transverse_cuda = np.sqrt(px_cuda ** 2 + py_cuda ** 2)

    num_bins = 100
    bins_px = np.linspace(min(px_g4.min(), px_cuda.min()), max(px_g4.max(), px_cuda.max()), num_bins)
    bins_py = np.linspace(min(py_g4.min(), py_cuda.min()), max(py_g4.max(), py_cuda.max()), num_bins)
    bins_pz = np.linspace(min(pz_g4.min(), pz_cuda.min()), max(pz_g4.max(), pz_cuda.max()), num_bins)
    bins_p_mag = np.linspace(min(p_mag_g4.min(), p_mag_cuda.min()), max(p_mag_g4.max(), p_mag_cuda.max()), num_bins)
    bins_p_transverse = np.linspace(min(p_transverse_g4.min(), p_transverse_cuda.min()), max(p_transverse_g4.max(), p_transverse_cuda.max()), num_bins)
    bins_x = np.linspace(min(x_g4.min(), x_cuda.min()), max(x_g4.max(), x_cuda.max()), num_bins)
    bins_y = np.linspace(min(y_g4.min(), y_cuda.min()), max(y_g4.max(), y_cuda.max()), num_bins)
    bins_z = np.linspace(min(z_g4.min(), z_cuda.min()), max(z_g4.max(), z_cuda.max()), num_bins)

    fig, axs = plt.subplots(3, 3, figsize=(15, 12))
    density = args.density

    ylabel = 'Density' if density else 'Count'

    title_fontsize = 18
    label_fontsize = 16
    tick_fontsize = 14
    legend_fontsize = 14

    hist_g4, bins_g4, _ = axs[0, 0].hist(px_g4, bins=bins_px, density=density, histtype='step', color='firebrick', label='Geant4', log=True)
    hist_cuda, bins_cuda, _ = axs[0, 0].hist(px_cuda, bins=bins_px, density=density, histtype='step', color='steelblue', label='CUDA', log=True)
    axs[0, 0].set_title(r'$p_x$ Component', fontsize=title_fontsize)
    axs[0, 0].set_xlabel(r'$p_x$', fontsize=label_fontsize)
    axs[0, 0].set_ylabel(ylabel, fontsize=label_fontsize)
    axs[0, 0].legend(fontsize=legend_fontsize)

    mse = np.mean((hist_g4 - hist_cuda)**2)
    print(f"MSE of Px: {mse:.3e}")

    hist_g4, bins_g4, _ = axs[0, 1].hist(py_g4, bins=bins_py, density=density, histtype='step', color='firebrick', label='Geant4', log=True)
    hist_cuda, bins_cuda, _ = axs[0, 1].hist(py_cuda, bins=bins_py, density=density, histtype='step', color='steelblue', label='CUDA', log=True)
    axs[0, 1].set_title(r'$p_y$ Component', fontsize=title_fontsize)
    axs[0, 1].set_xlabel(r'$p_y$', fontsize=label_fontsize)
    axs[0, 1].set_ylabel(ylabel, fontsize=label_fontsize)
    axs[0, 1].legend(fontsize=legend_fontsize)

    mse = np.mean((hist_g4 - hist_cuda)**2)
    print(f"MSE of Py: {mse:.3e}")

    hist_g4, bins_g4, _ = axs[0, 2].hist(pz_g4, bins=bins_pz, density=density, histtype='step', color='firebrick', label='Geant4', log=True)
    hist_cuda, bins_cuda, _ = axs[0, 2].hist(pz_cuda, bins=bins_pz, density=density, histtype='step', color='steelblue', label='CUDA', log=True)
    axs[0, 2].set_title(r'$p_z$ Component', fontsize=title_fontsize)
    axs[0, 2].set_xlabel(r'$p_z$', fontsize=label_fontsize)
    axs[0, 2].set_ylabel(ylabel, fontsize=label_fontsize)
    axs[0, 2].legend(fontsize=legend_fontsize)

    mse = np.mean((hist_g4 - hist_cuda)**2)
    print(f"MSE of Pz: {mse:.3e}")

    hist_g4, bins_g4, _ = axs[1, 0].hist(p_mag_g4, bins=bins_p_mag, density=density, histtype='step', color='firebrick', label='Geant4', log=True)
    hist_cuda, bins_cuda, _ = axs[1, 0].hist(p_mag_cuda, bins=bins_p_mag, density=density, histtype='step', color='steelblue', label='CUDA', log=True)
    axs[1, 0].set_title(r'Momentum Magnitude $(|p|)$')
    axs[1, 0].set_xlabel(r'$|p|$')
    axs[1, 0].set_ylabel(ylabel)
    axs[1, 0].legend()

    mse = np.mean((hist_g4 - hist_cuda)**2)
    print(f"MSE of |P|: {mse:.3e}")

    hist_g4, bins_g4, _ = axs[1, 1].hist(p_transverse_g4, bins=bins_p_transverse, density=density, histtype='step', color='firebrick', label='Geant4', log=True)
    hist_cuda, bins_cuda, _ = axs[1, 1].hist(p_transverse_cuda, bins=bins_p_transverse, density=density, histtype='step', color='steelblue', label='CUDA', log=True)
    axs[1, 1].set_title(r'Transverse Momentum $(\sqrt{p_x^2 + p_y^2})$')
    axs[1, 1].set_xlabel(r'$\sqrt{p_x^2 + p_y^2}$')
    axs[1, 1].set_ylabel(ylabel)
    axs[1, 1].legend()

    mse = np.mean((hist_g4 - hist_cuda)**2)
    print(f"MSE of Pt: {mse:.3e}")

    hist_g4, bins_g4, _ = axs[2, 0].hist(x_g4, bins=bins_x, density=density, histtype='step', color='firebrick', label='Geant4', log=True)
    hist_cuda, bins_cuda, _ = axs[2, 0].hist(x_cuda, bins=bins_x, density=density, histtype='step', color='steelblue', label='CUDA', log=True)
    axs[2, 0].set_title(r'$x$ Position')
    axs[2, 0].set_xlabel(r'$x$')
    axs[2, 0].set_ylabel(ylabel)
    axs[2, 0].legend()

    mse = np.mean((hist_g4 - hist_cuda)**2)
    print(f"MSE of x: {mse:.3e}")

    hist_g4, bins_g4, _ = axs[2, 1].hist(y_g4, bins=bins_y, density=density, histtype='step', color='firebrick', label='Geant4', log=True)
    hist_cuda, bins_cuda, _ = axs[2, 1].hist(y_cuda, bins=bins_y, density=density, histtype='step', color='steelblue', label='CUDA', log=True)
    axs[2, 1].set_title(r'$y$ Position')
    axs[2, 1].set_xlabel(r'$y$')
    axs[2, 1].set_ylabel(ylabel)
    axs[2, 1].legend()

    mse = np.mean((hist_g4 - hist_cuda)**2)
    print(f"MSE of y: {mse:.3e}")

    hist_g4, bins_g4, _ = axs[2, 2].hist(z_g4, bins=bins_z, density=density, histtype='step', color='firebrick', label='Geant4', log=True)
    hist_cuda, bins_cuda, _ = axs[2, 2].hist(z_cuda, bins=bins_z, density=density, histtype='step', color='steelblue', label='CUDA', log=True)
    axs[2, 2].set_title(r'$z$ Position')
    axs[2, 2].set_xlabel(r'$z$')
    axs[2, 2].set_ylabel(ylabel)
    axs[2, 2].legend()

    mse = np.mean((hist_g4 - hist_cuda)**2)
    print(f"MSE of z: {mse:.3e}")

    n = int(1e6)
    axs[1, 2].scatter(x_g4[:n], y_g4[:n], s=1, alpha=0.1, c=p_mag_g4[:n], cmap='Reds', vmin=max(0, args.filter_p[0]), vmax=min(250, args.filter_p[1]), label='Geant4')
    axs[1, 2].scatter(x_cuda[:n], y_cuda[:n], s=1, alpha=0.1, c=p_mag_cuda[:n], cmap='Blues', vmin=max(0, args.filter_p[0]), vmax=min(250, args.filter_p[1]), label='CUDA')
    axs[1, 2].set_title(r'Y vs X', fontsize=title_fontsize)
    axs[1, 2].set_xlabel(r'$x$ [m]', fontsize=label_fontsize)
    axs[1, 2].set_ylabel(r'$y$ [m]', fontsize=label_fontsize)

    for ax in axs.flatten():
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(output_filename)
    print(f'Plot saved to {output_filename}')
    plt.close()

def plot_positions(output_filename, 
                   px_g4, py_g4, pz_g4, x_g4, y_g4, z_g4,
                   px_cuda, py_cuda, pz_cuda, x_cuda, y_cuda, z_cuda,  weights_g4=None, weights_cuda=None):
    """
    Plots histograms of x and y position components side-by-side.
    Saves the plot as a PDF with a transparent background.
    """
    # --- Filter Data ---
    px_g4, py_g4, pz_g4, x_g4, y_g4, z_g4, weights_g4 = filter_data(
        px_g4, py_g4, pz_g4, x_g4, y_g4, z_g4, weights_g4)
    
    px_cuda, py_cuda, pz_cuda, x_cuda, y_cuda, z_cuda, weights_cuda = filter_data(
        px_cuda, py_cuda, pz_cuda, x_cuda, y_cuda, z_cuda, weights_cuda)

    # --- Plotting Setup ---
    plt.style.use('seaborn-v0_8-paper') # A good style for papers
    fig, axs = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
    
    density = args.density
    ylabel = 'Density' if density else 'Count'
    num_bins = 100

    # --- Font Sizes for Publication Quality ---
    title_fontsize = 18
    label_fontsize = 16
    tick_fontsize = 14
    legend_fontsize = 14

    # --- Plot X Position ---
    bins_x = np.linspace(min(x_g4.min(), x_cuda.min()), max(x_g4.max(), x_cuda.max()), num_bins)
    axs[0].hist(x_g4, bins=bins_x, weights=weights_g4, density=density, histtype='step', color='firebrick', label='Geant4', lw=1.5, log=True)
    axs[0].hist(x_cuda, bins=bins_x, weights=weights_cuda, density=density, histtype='step', color='steelblue', label='CUDA', lw=1.5, log=True)
    axs[0].set_title(r'$x$ Position', fontsize=title_fontsize)
    axs[0].set_xlabel(r'$x$ [m]', fontsize=label_fontsize)
    axs[0].set_ylabel(ylabel, fontsize=label_fontsize)
    axs[0].tick_params(axis='both', which='major', labelsize=tick_fontsize)
    axs[0].legend(fontsize=legend_fontsize)
    axs[0].grid(True, linestyle='--', alpha=0.6)

    # --- Plot Y Position ---
    bins_y = np.linspace(min(y_g4.min(), y_cuda.min()), max(y_g4.max(), y_cuda.max()), num_bins)
    axs[1].hist(y_g4, bins=bins_y, weights=weights_g4, density=density, histtype='step', color='firebrick', label='Geant4', lw=1.5, log=True)
    axs[1].hist(y_cuda, bins=bins_y, weights=weights_cuda, density=density, histtype='step', color='steelblue', label='CUDA', lw=1.5, log=True)
    axs[1].set_title(r'$y$ Position', fontsize=title_fontsize)
    axs[1].set_xlabel(r'$y$ [m]', fontsize=label_fontsize)
    axs[1].tick_params(axis='both', which='major', labelsize=tick_fontsize)
    axs[1].legend(fontsize=legend_fontsize)
    axs[1].grid(True, linestyle='--', alpha=0.6)

    # --- Save Figure ---
    plt.tight_layout()
    plt.savefig(output_filename, format='pdf', transparent=True)
    print(f'Position plot saved to {output_filename}')
    plt.close()

def plot_momenta(output_filename, 
                 px_g4, py_g4, pz_g4, x_g4, y_g4, z_g4,
                 px_cuda, py_cuda, pz_cuda, x_cuda, y_cuda, z_cuda, 
                 weights_g4=None, weights_cuda=None):
    """
    Plots histograms of px, py, and pz momentum components side-by-side.
    Saves the plot as a PDF with a transparent background.
    """
    # --- Filter Data ---
    px_g4, py_g4, pz_g4, x_g4, y_g4, z_g4, weights_g4 = filter_data(
        px_g4, py_g4, pz_g4, x_g4, y_g4, z_g4, weights_g4)
    
    px_cuda, py_cuda, pz_cuda, x_cuda, y_cuda, z_cuda, weights_cuda = filter_data(
        px_cuda, py_cuda, pz_cuda, x_cuda, y_cuda, z_cuda, weights_cuda)
    
    # --- Plotting Setup ---
    plt.style.use('seaborn-v0_8-paper')
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    density = args.density
    ylabel = 'Density' if density else 'Count'
    num_bins = 100

    # --- Font Sizes for Publication Quality ---
    title_fontsize = 18
    label_fontsize = 16
    tick_fontsize = 14
    legend_fontsize = 14

    # --- Plot Px ---
    bins_px = np.linspace(min(px_g4.min(), px_cuda.min()), max(px_g4.max(), px_cuda.max()), num_bins)
    axs[0].hist(px_g4, bins=bins_px, weights=weights_g4, density=density, histtype='step', color='firebrick', label='Geant4', lw=1.5, log=True)
    axs[0].hist(px_cuda, bins=bins_px, weights=weights_cuda, density=density, histtype='step', color='steelblue', label='CUDA', lw=1.5, log=True)
    axs[0].set_title(r'$p_x$ Component', fontsize=title_fontsize)
    axs[0].set_xlabel(r'$p_x$ [GeV]', fontsize=label_fontsize)
    axs[0].set_ylabel(ylabel, fontsize=label_fontsize)
    axs[0].tick_params(axis='both', which='major', labelsize=tick_fontsize)
    axs[0].legend(fontsize=legend_fontsize)
    axs[0].grid(True, linestyle='--', alpha=0.6)

    # --- Plot Py ---
    bins_py = np.linspace(min(py_g4.min(), py_cuda.min()), max(py_g4.max(), py_cuda.max()), num_bins)
    axs[1].hist(py_g4, bins=bins_py, weights=weights_g4, density=density, histtype='step', color='firebrick', label='Geant4', lw=1.5, log=True)
    axs[1].hist(py_cuda, bins=bins_py, weights=weights_cuda, density=density, histtype='step', color='steelblue', label='CUDA', lw=1.5, log=True)
    axs[1].set_title(r'$p_y$ Component', fontsize=title_fontsize)
    axs[1].set_xlabel(r'$p_y$ [GeV]', fontsize=label_fontsize)
    axs[1].tick_params(axis='both', which='major', labelsize=tick_fontsize)
    axs[1].legend(fontsize=legend_fontsize)
    axs[1].grid(True, linestyle='--', alpha=0.6)
    
    # --- Plot Pz ---
    bins_pz = np.linspace(min(pz_g4.min(), pz_cuda.min()), max(pz_g4.max(), pz_cuda.max()), num_bins)
    axs[2].hist(pz_g4, bins=bins_pz, weights=weights_g4, density=density, histtype='step', color='firebrick', label='Geant4', lw=1.5, log=True)
    axs[2].hist(pz_cuda, bins=bins_pz, weights=weights_cuda, density=density, histtype='step', color='steelblue', label='CUDA', lw=1.5, log=True)
    axs[2].set_title(r'$p_z$ Component', fontsize=title_fontsize)
    axs[2].set_xlabel(r'$p_z$ [GeV]', fontsize=label_fontsize)
    axs[2].tick_params(axis='both', which='major', labelsize=tick_fontsize)
    axs[2].legend(fontsize=legend_fontsize)
    axs[2].grid(True, linestyle='--', alpha=0.6)

    # --- Save Figure ---
    plt.tight_layout()
    plt.savefig(output_filename, format='pdf', transparent=True)
    print(f'Momentum plot saved to {output_filename}')
    plt.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run muon simulations with specified number of muons.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n_muons', type=int, default=int(5e6), help='Number of muons to simulate')
    parser.add_argument('--n_steps', type=int, default=500, help='Number of steps to simulate')
    parser.add_argument('--mag_field', type=float, nargs=3, default=[0., 0., 0.], help='Magnetic field vector components (Bx, By, Bz)')
    parser.add_argument('--material', type=str, default='G4_Fe', help='Material to use for simulations (G4_Fe, G4_CONCRETE)')
    parser.add_argument('--load_data', action='store_true', help='Load existing data instead of running simulations')
    parser.add_argument('--initial_momenta', type=float, default=195., help='Initial momenta of muons in GeV/c')
    parser.add_argument('--output_filename', type=str, default='histograms_comparison_geant4_cuda.png', help='Output filename for the histogram plot')
    parser.add_argument('--file_name_g4', type=str, help='Path to GEANT4 HDF5 file', default='../data/outputs/results/final_concatenated_results.h5')
    parser.add_argument('--file_name_cuda', type=str, help='Path to CUDA pickle file', default =  '../data/outputs/outputs_cuda.pkl')
    parser.add_argument('--density', action='store_true', help='Plot histograms with density normalization')
    parser.add_argument('--filter_p', type=float, nargs=2, default=[0.0, 500.0], help='Thresholds to filter muons by min_p <= |P| <= max_p')
    parser.add_argument('--sens_plane', action='store_true', help='Apply sensitive plane cut (|x|<2, |y|<3)')
    parser.add_argument('--propagate_until', type=float, default=None, help='If >0, propagate the loaded data until this z')
    args = parser.parse_args()
    n_muons = args.n_muons

    if args.load_data:
        print("Loading GEANT4 data...")
        file_name_g4 = args.file_name_g4
        file_name_cuda = args.file_name_cuda
        with h5py.File(file_name_g4,'r') as f:
            data_geant4 = {key: f[key][:] for key in f.keys()}
        print("Loaded G4 data. Shape of data:", len(data_geant4['px']))

        print("Loading CUDA data...")
        if file_name_cuda.endswith('.h5'):
            with h5py.File(file_name_cuda,'r') as f:
                data_cuda = {key: f[key][:] for key in f.keys()}
        else:
            with open(file_name_cuda, 'rb') as f:
                data_cuda = pickle.load(f)
        for key in data_cuda:
            if not isinstance(data_cuda[key], np.ndarray): data_cuda[key] = data_cuda[key].numpy()
        print("Loaded CUDA data. Shape of data:", len(data_cuda['px']))
    else:
        initial_momenta = np.array([[0.,0.,args.initial_momenta]])
        initial_positions_charge = np.array([[0.,0.,0., -1]])
        print("Generating %d muons..."%n_muons)
        muons = np.concatenate((initial_momenta, initial_positions_charge), axis=1)*np.ones((n_muons,1))
        data_geant4, data_cuda = main(muons, n_steps=args.n_steps, mag_field=list(args.mag_field), material=args.material)

    for key in data_cuda:
        if key in ['pdg_id', 'W']: continue
        print("=" * 30, key, "=" * 30)
        cuda_vals = np.array(data_cuda[key])
        geant4_vals = np.array(data_geant4[key])
        print(f"Number of entries - CUDA: {len(cuda_vals)}, Geant4: {len(geant4_vals)}")
        print(f"CUDA {key} values: {cuda_vals}, Geant4 {key} values: {geant4_vals}")
        print(f"CUDA {key} average: {cuda_vals.mean()}, Geant4 {key} average: {geant4_vals.mean()}")
        print(f"CUDA {key} std: {cuda_vals.std()}, Geant4 {key} std: {geant4_vals.std()}")


    px_g4 = data_geant4['px']
    py_g4 = data_geant4['py']
    pz_g4 = data_geant4['pz']
    x_g4 = data_geant4['x']
    y_g4 = data_geant4['y']
    z_g4 = data_geant4['z']

    px_cuda = data_cuda['px']
    py_cuda = data_cuda['py']
    pz_cuda = data_cuda['pz']
    x_cuda = data_cuda['x']
    y_cuda = data_cuda['y']
    z_cuda = data_cuda['z']

    if args.propagate_until is not None:
        x_g4, y_g4, z_g4 = propagate_muons_until_z(px_g4, py_g4, pz_g4, x_g4, y_g4, z_g4, args.propagate_until)
        x_cuda, y_cuda, z_cuda = propagate_muons_until_z(px_cuda, py_cuda, pz_cuda, x_cuda, y_cuda, z_cuda, args.propagate_until)

    print("=" * 60)
    print('TOTAL number of samples GEANT4:', px_g4.shape[0])
    print('TOTAL number of samples CUDA:', px_cuda.shape[0])
    weights_g4 = None
    weights_cuda = None
    if 'weight' in data_geant4:
        weights_g4 = data_geant4['weight']
        print('TOTAL sum of weights GEANT4:', np.sum(weights_g4))
    if 'weight' in data_cuda:
        weights_cuda = data_cuda['weight']
        print('TOTAL sum of weights CUDA:', np.sum(weights_cuda))

    output_filename = os.path.join('/home/hep/lprate/projects/MuonsAndMatter/cuda_muons/plots', args.output_filename)
    plot_histograms(output_filename, 
                px_g4, py_g4, pz_g4, x_g4, y_g4, z_g4,
                px_cuda, py_cuda, pz_cuda, x_cuda, y_cuda, z_cuda,
                weights_g4, weights_cuda)
    plot_positions(output_filename.replace('.png','_positions.pdf'),
                px_g4, py_g4, pz_g4, x_g4, y_g4, z_g4,
                px_cuda, py_cuda, pz_cuda, x_cuda, y_cuda, z_cuda,
                weights_g4, weights_cuda)
    plot_momenta(output_filename.replace('.png','_momenta.pdf'),
                px_g4, py_g4, pz_g4, x_g4, y_g4, z_g4,
                px_cuda, py_cuda, pz_cuda, x_cuda, y_cuda, z_cuda,
                weights_g4, weights_cuda)
    
    if args.n_steps == 1:
        log_start = np.log10(0.18)
        log_end = np.log10(400)
        inv_log_step = 95/(log_end - log_start)
        index = int((np.log10(args.initial_momenta) - log_start)*inv_log_step)
        with h5py.File("/home/hep/lprate/projects/MuonsAndMatter/cuda_muons/data/muon_data_energy_loss_sens_{}.h5".format(args.material), "r") as f:
            keys = sorted(list(f.keys()),key=lambda k: float(k.strip("()").split(",")[0]))
            pz = f[keys[index]]["pz"][:]
            initial_momenta = f[keys[index]]["initial_momenta"][:]
            pt = np.sqrt(f[keys[index]]["px"][:]**2 + f[keys[index]]["py"][:]**2)
        pt_g4 = np.sqrt(px_g4**2 + py_g4**2)
        pt_cuda = np.sqrt(px_cuda**2 + py_cuda**2)
        bins = np.linspace(-5,1, 500)
        plt.hist(np.log(np.abs(pz_g4-args.initial_momenta)), bins=bins, density=True, histtype='step', color='firebrick', label='Geant4', log=True)
        plt.hist(np.log(np.abs(pz-initial_momenta)), bins=bins, density=True, histtype='step', color='green', label='GEANT4 (HDF5)', log=True)
        plt.hist(np.log(np.abs(pz_cuda-args.initial_momenta)), bins=bins, density=True, histtype='step', color='steelblue', label='CUDA', log=True)
        plt.title(r'$p_z$ Component', fontsize=18)
        plt.xlabel(r'$p_z$ [GeV/c]', fontsize=16)
        plt.ylabel('Density', fontsize=16)
        plt.legend(fontsize=14)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join('/home/hep/lprate/projects/MuonsAndMatter/cuda_muons/plots', 'pz_comparison_geant4_cuda.png'))
        plt.close()
        print("Plot saved to ", os.path.join('/home/hep/lprate/projects/MuonsAndMatter/cuda_muons/plots', 'pz_comparison_geant4_cuda.png'))

        
        bins = np.linspace(min(np.min(np.log(pt_g4)), np.min(np.log(pt_cuda))), 0.1, 200)
        plt.hist(np.log(pt_g4/args.initial_momenta), bins=bins, density=True, histtype='step', color='firebrick', label='Geant4', log=True)
        plt.hist(np.log(pt/initial_momenta), bins=bins, density=True, histtype='step', color='green', label='GEANT4 (HDF5)', log=True)
        plt.hist(np.log(pt_cuda/args.initial_momenta), bins=bins, density=True, histtype='step', color='steelblue', label='CUDA', log=True)
        plt.title(r'$p_T$ Component', fontsize=18)
        plt.xlabel(r'$p_T$ [GeV/c]', fontsize=16)
        plt.ylabel('Density', fontsize=16)
        plt.legend(fontsize=14)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join('/home/hep/lprate/projects/MuonsAndMatter/cuda_muons/plots', 'log_pt_comparison_geant4_cuda.png'))
        plt.close()

        bins = np.linspace(min(np.min(pt_g4), np.min(pt_cuda)), max(np.max(pt_g4), np.max(pt_cuda)), 200)
        plt.hist(pt_g4, bins=bins, density=True, histtype='step', color='firebrick', label='Geant4', log=True)
        plt.hist(pt, bins=bins, density=True, histtype='step', color='green', label='GEANT4 (HDF5)', log=True)
        plt.hist(pt_cuda, bins=bins, density=True, histtype='step', color='steelblue', label='CUDA', log=True)
        plt.title(r'$p_T$ Component', fontsize=18)
        plt.xlabel(r'$p_T$ [GeV/c]', fontsize=16)
        plt.ylabel('Density', fontsize=16)
        plt.legend(fontsize=14)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join('/home/hep/lprate/projects/MuonsAndMatter/cuda_muons/plots', 'pt_comparison_geant4_cuda.png'))
        plt.close()




