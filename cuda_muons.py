import time
import numpy as np
import pickle
import torch
import h5py
import os
from utils_cuda_muons.get_geometry import (
    get_corners_from_params,
    get_cavern_from_params,
    create_z_axis_grid,
)
from utils_cuda_muons.get_magnetic_field import get_magnetic_field_from_params
import faster_muons_torch
assert torch.cuda.is_available(), f"CUDA is not available. Torch version: {torch.__version__} \n Torch cuda version: {print(torch.version.cuda)}"

import warnings
# Suppress only the specific torch.storage FutureWarning
warnings.filterwarnings(
    "ignore",
    message=".*weights_only=False.*",
    category=FutureWarning,
    module="torch.storage"
)

def propagate_muons_with_cuda(
    muons_positions,
    muons_momenta,
    muons_charge,
    arb8_corners,
    cavern_params,
    material_histograms,  # Dict: {'iron': [prob, alias, centers1, centers2, widths1, widths2], 'concrete': [...]}
    magnetic_field,  # Either a dict (field map) or a tensor of shape (N_arb8s, 3) for uniform field per block
    sensitive_plane_z: float = 82,
    num_steps=100,
    step_length_fixed=0.02,
    use_symmetry=True,
    seed=1234,
    device='cuda',
    ):

    muons_positions_cuda = muons_positions.float().to(device)
    muons_momenta_cuda = muons_momenta.float().to(device)
    muons_charge = muons_charge.float().to(device)
    
    material_histograms_cuda = {}
    for material, hists in material_histograms.items():
        # hists[1] (alias_table) should be int, rest are float
        material_histograms_cuda[material] = [
            hists[0].float().to(device).contiguous(),   # probability_table
            hists[1].int().to(device).contiguous(),     # alias_table
            hists[2].float().to(device).contiguous(),   # bin_centers_first_dim
            hists[3].float().to(device).contiguous(),   # bin_centers_second_dim
            hists[4].float().to(device).contiguous(),   # bin_widths_first_dim
            hists[5].float().to(device).contiguous(),   # bin_widths_second_dim
        ]
    
    cells_arb8, hashed_arb8 = create_z_axis_grid(arb8_corners, sz=15)
    cells_arb8 = cells_arb8.int().contiguous().to(device)
    hashed_arb8 = hashed_arb8.int().contiguous().to(device)
    arb8_corners = arb8_corners.float().to(device)
    cavern_params = cavern_params.float().to(device)

    kill_at = 0.18
    assert not arb8_corners.isnan().any(), "Arb8 corners contain NaN values."

    # Check if magnetic_field is a dict (field map) or tensor (uniform field per ARB8)
    use_field_map = isinstance(magnetic_field, dict)

    if use_field_map:
        # Field map mode: magnetic_field is a dict with 'B' and range info
        magnetic_field_B = torch.from_numpy(magnetic_field['B'])
        magnetic_field_ranges = [magnetic_field['range_x'][0], magnetic_field['range_x'][1],
                                 magnetic_field['range_y'][0], magnetic_field['range_y'][1],
                                 magnetic_field['range_z'][0], magnetic_field['range_z'][1]]
        nx = int(round((magnetic_field_ranges[1] - magnetic_field_ranges[0]) / magnetic_field['range_x'][2])) + 1
        ny = int(round((magnetic_field_ranges[3] - magnetic_field_ranges[2]) / magnetic_field['range_y'][2])) + 1
        nz = int(round((magnetic_field_ranges[5] - magnetic_field_ranges[4]) / magnetic_field['range_z'][2])) + 1
        magnetic_field_B = magnetic_field_B.view(nx, ny, nz, 3).float().to(device).contiguous()
        magnetic_field_ranges = torch.tensor([magnetic_field_ranges]).div(100).float().cpu().contiguous()
        
        assert not magnetic_field_B.isnan().any(), "Magnetic field contains NaN values."
        
        cuda_muons_fn = faster_muons_torch.propagate_muons_with_alias_sampling
        field_args = (magnetic_field_B, magnetic_field_ranges)
    else:
        # Uniform field mode: magnetic_field is a tensor of shape (N_arb8s, 3)
        arb8s_fields = magnetic_field.float().to(device).contiguous()
        assert arb8s_fields.shape[0] == arb8_corners.shape[0], \
            f"arb8s_fields must have the same number of ARB8s as arb8_corners. Got {arb8s_fields.shape[0]} vs {arb8_corners.shape[0]}"
        assert arb8s_fields.shape[1] == 3, \
            f"arb8s_fields must have shape (N, 3) for [Bx, By, Bz]. Got shape {arb8s_fields.shape}"
        
        cuda_muons_fn = faster_muons_torch.propagate_muons_with_alias_sampling_uniform_field
        field_args = (arb8s_fields,)

    t1 = time.time()
    cuda_muons_fn(
        muons_positions_cuda,
        muons_momenta_cuda,
        muons_charge,
        material_histograms_cuda,
        arb8_corners,
        cells_arb8,
        hashed_arb8,
        *field_args,
        cavern_params,
        use_symmetry,
        sensitive_plane_z,
        kill_at,
        num_steps,
        step_length_fixed,
        seed
    )
    
    torch.cuda.synchronize()
    mode_str = "field map" if use_field_map else "uniform field"
    print(f"Took {time.time() - t1:.3f} seconds for {len(muons_positions_cuda):.2e} muons and {num_steps} steps ({mode_str}).")

    # Convert results back to numpy arrays and return (on CPU)
    return muons_positions_cuda.cpu(), muons_momenta_cuda.cpu()

def run_from_params(params,
    muons:np.array,
        sensitive_plane={'dz': 0.02, 'dx': 4, 'dy': 6, 'position': 82.0},
        histogram_dir='cuda_muons/data',
        save_dir = None,
        n_steps=500,
        fSC_mag = False,
        field_map_file = None,
        NI_from_B = True,
        use_diluted = False,
        add_cavern = True,
        use_uniform_field = True,
        cores_field = 8,
        return_all = False,
        seed = 0,
        device='cuda'):
    """
    Run muon propagation from magnet parameters.
    
    Args:
        params: Array of shape (N_magnets, 15) with magnet parameters.
        muons: Input muon array with columns [px, py, pz, x, y, z, pdg_id, weight].
        sensitive_plane: Dict with 'dz', 'dx', 'dy', 'position' or list of such dicts.
        histogram_dir: Directory containing alias histogram files.
        save_dir: If provided, save output to this path.
        n_steps: Number of propagation steps.
        fSC_mag: Whether superconducting magnets are used.
        field_map_file: Path to field map file (only used if use_uniform_field=False).
        NI_from_B: Whether NI is derived from B (affects SC threshold).
        use_diluted: Whether to use diluted steel in FEM simulation.
        add_cavern: Whether to include cavern geometry.
        use_uniform_field: If True, use uniform field per ARB8 block (fast).
                          If False, use FEM-simulated field map (slow, requires snoopy).
        cores_field: Number of CPU cores for field simulation (only if use_uniform_field=False).
        return_all: If True, return all muons; if False, filter by sensitive plane.
        seed: Random seed for propagation.
        device: Device to run on ('cuda' or GPU index).
    
    Returns:
        Dict with 'px', 'py', 'pz', 'x', 'y', 'z', 'pdg_id', and optionally 'weight'.
    """
    if isinstance(sensitive_plane, list):
        for plane in sensitive_plane:
            output = run_from_params(params, muons, sensitive_plane=plane,
                         histogram_dir=histogram_dir, n_steps=n_steps,
                         fSC_mag=fSC_mag,
                         field_map_file=field_map_file, NI_from_B=NI_from_B,
                         use_diluted=use_diluted, add_cavern=add_cavern,
                         use_uniform_field=use_uniform_field, cores_field=cores_field,
                         return_all=return_all, seed=seed, device=device)
            muons = torch.stack([output['px'], output['py'], output['pz'],
                                  output['x'], output['y'], output['z'],
                                  output['pdg_id'], output['weight']], dim=1)
            if return_all:
                in_sens_plane = (muons[:,3].abs() < plane['dx']/2) & \
                            (muons[:,4].abs() < plane['dy']/2) & \
                            (muons[:,5] >= (plane['position'] - plane['dz']/2)) 
                muons[~in_sens_plane,:3] = 0.0
            elif muons.shape[0]==0: break
        return output
    
    t0 = time.time()
    if seed is None: seed = np.random.randint(0, 10000)
    if not torch.is_tensor(muons): muons = torch.from_numpy(muons).float()
    muons_momenta = muons[:,:3]
    muons_positions = muons[:,3:6]
    muons_charge = muons[:,6]
    if muons_charge.abs().eq(13).all(): muons_charge = muons_charge.div(-13)
    assert muons_charge.abs().eq(1).all(), f"PDG IDs or charges in the input file are not correct. They should be either +/-13 or +/-1., {muons_charge.unique(return_counts=True)}" 

    # Get geometry and field directly from params (no detector creation)
    params = np.asarray(params).reshape(-1, 15)
    
    corners = get_corners_from_params(params, fSC_mag=fSC_mag, NI_from_B=NI_from_B)
    cavern = get_cavern_from_params(add_cavern=add_cavern)
    
    # Get magnetic field: either uniform (fast) or simulated field map (slow)
    simulate_fields = not use_uniform_field
    mag_field = get_magnetic_field_from_params(
        params,
        simulate_fields=simulate_fields,
        field_map_file=field_map_file if not use_uniform_field else None,
        fSC_mag=fSC_mag,
        NI_from_B=NI_from_B,
        use_diluted=use_diluted,
        cores_field=cores_field,
    )
    
    mode_str = "uniform field" if use_uniform_field else "field map"
    print(f"Geometry + {mode_str} setup took {time.time() - t0:.2f} seconds.")

    use_symmetry = True
    
    # Load histograms into a dict
    def load_histogram(filepath):
        with open(filepath, 'rb') as f:
            hist_data = pickle.load(f)
        return [
            hist_data['hist_2d_probability_table'],
            hist_data['hist_2d_alias_table'],
            hist_data['centers_first_dim'],
            hist_data['centers_second_dim'],
            hist_data['width_first_dim'],
            hist_data['width_second_dim']
        ], hist_data['step_length']
    
    iron_hists, step_length = load_histogram(os.path.join(histogram_dir, 'alias_histograms_G4_Fe.pkl'))
    
    material_histograms = {'iron': iron_hists}
    
    if add_cavern:
        concrete_hists, concrete_step = load_histogram(os.path.join(histogram_dir, 'alias_histograms_G4_CONCRETE.pkl'))
        assert step_length == concrete_step, "Step lengths in the two histogram files are different"
        material_histograms['concrete'] = concrete_hists
    else:
        # Create dummy concrete histograms (zeros with same shape as iron)
        material_histograms['concrete'] = [torch.zeros_like(h) for h in iron_hists]


    sensitive_plane_z = -2.0 if sensitive_plane is None else sensitive_plane['position']

    print("Using CUDA for propagation... (server)")
    #torch.cuda.synchronize()
    out_position, out_momenta = propagate_muons_with_cuda (
            muons_positions,
            muons_momenta,
            muons_charge,
            corners,
            cavern,
            material_histograms,
            mag_field,
            sensitive_plane_z - sensitive_plane['dz']/2,
            n_steps,
            step_length,
            use_symmetry,
            seed,
            device,
        )   
    weights = muons[:,7] if (muons.shape[1]>7) else None
    if sensitive_plane is not None and not return_all:
        in_sens_plane = (out_position[:,0].abs() < sensitive_plane['dx']/2) & \
                        (out_position[:,1].abs() < sensitive_plane['dy']/2) & \
                        (out_position[:,2] >= (sensitive_plane['position'] - sensitive_plane['dz']/2)) #& \
                        #(out_position[:,2] <= (sensitive_plane['position'] + sensitive_plane['dz']/2))

        out_momenta = out_momenta[in_sens_plane]
        out_position = out_position[in_sens_plane]
        muons_charge = muons_charge[in_sens_plane].int()
        print("Number of outputs:", out_momenta.shape[0])
        weights = weights[in_sens_plane] if weights is not None else None

    out_position = out_position.cpu()
    out_momenta = out_momenta.cpu()
    output = {
        'px': out_momenta[:,0],
        'py': out_momenta[:,1],
        'pz': out_momenta[:,2],
        'x': out_position[:,0],
        'y': out_position[:,1],
        'z': out_position[:,2],
        'pdg_id': muons_charge*(-13)
    }
    if muons.shape[1]>7:
        output['weight'] = weights
        print("Number of HITS (weighted)", weights.sum().item())
    if save_dir is not None:
        t1 = time.time()
        with open(save_dir, 'wb') as f:
            pickle.dump(output, f)
        print("Data saved to", save_dir, "took", time.time() - t1, "seconds to save.")
    return output  



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--h',dest = 'histogram_dir', type=str, default='cuda_muons/data/',
                        help='Path to the histogram file')
    parser.add_argument('-muons', '--f', dest='input_file', type=str, default="data/muons/full_sample_after_target.h5",
                        help='Path to input muon file (.npy, .pkl, .h5). If not provided a synthetic example will be used.')
    parser.add_argument('-n_muons', '--n', dest='n_muons', type=int, default=0,
                        help='Maximum number of muons to load from the input file; 0 means all')
    parser.add_argument('--n_steps', type=int, default=5000,
                        help='Number of steps for simulation')
    parser.add_argument('-sens_plane', type=float, default=82.0,
                        help='Z position of the sensitive plane')
    parser.add_argument("-remove_cavern", dest="add_cavern", action='store_false', help="Remove the cavern from simulation")
    parser.add_argument('-plot', action='store_true',
                        help='Plot histograms')
    parser.add_argument('-use_field_map', action='store_true',
                        help='Use FEM-simulated field map instead of uniform field (slower, requires snoopy)')
    parser.add_argument("-params", type=str, default='tokanut_v5.txt', help="Magnet parameters configuration - name or file path. If 'input', will prompt for input.")
    parser.add_argument('--gpu', dest='gpu', type=int, default=0,
                        help='GPU index to use (e.g., 0, 1, ...).')
    args = parser.parse_args()
    
    if args.params == 'input':
        params_input = input("Enter the params as a Python list (e.g., [1.0, 2.0, 3.0]): ")
        params = eval(params_input)
    elif os.path.isfile(args.params):
        with open(args.params, "r") as txt_file:
            params = [float(line.strip()) for line in txt_file]
    else: 
        raise ValueError(f"Invalid params: {args.params}. Must be a valid parameter name or a file path.")
    params = np.asarray(params).reshape(-1, 15)
    time0 = time.time()
    if args.input_file is not None:
        input_file = args.input_file
        print(f"Loading muons from {input_file}")
        if input_file.endswith('.npy') or input_file.endswith('.pkl'):
            muons_iter = np.load(input_file, allow_pickle=True, mmap_mode='r')
            muons = muons_iter[:args.n_muons].copy() if args.n_muons > 0 else muons_iter[:].copy()
        elif input_file.endswith('.h5'):
            with h5py.File(input_file, 'r') as f:
                px = f['px'][: args.n_muons] if args.n_muons > 0 else f['px'][:]
                py = f['py'][: args.n_muons] if args.n_muons > 0 else f['py'][:]
                pz = f['pz'][: args.n_muons] if args.n_muons > 0 else f['pz'][:]
                x = f['x'][: args.n_muons] if args.n_muons > 0 else f['x'][:]
                y = f['y'][: args.n_muons] if args.n_muons > 0 else f['y'][:]
                z = f['z'][: args.n_muons] if args.n_muons > 0 else f['z'][:]
                pdg = f['pdg'][: args.n_muons] if args.n_muons > 0 else f['pdg'][:]
                weight = f['weight'][: args.n_muons] if args.n_muons > 0 else f['weight'][:]
                muons = np.stack([px, py, pz, x, y, z, pdg, weight], axis=1).astype(np.float64)
    print(f"Loaded {muons.shape[0]} muons. Took {time.time() - time0:.2f} seconds to load.")
    dx, dy = 9.0, 6.0
    sensitive_film_params = {'dz': 0.01, 'dx': dx, 'dy': dy, 'position':args.sens_plane} if args.sens_plane >0 else None
    t_run_start = time.time()
    output = run_from_params(params, muons, sensitive_film_params,
                 histogram_dir=args.histogram_dir, n_steps=args.n_steps,
                 fSC_mag=False, NI_from_B=True, 
                 use_diluted=False, add_cavern=args.add_cavern,
                 field_map_file=None, use_uniform_field=not args.use_field_map,
                 save_dir="data/outputs/outputs_cuda.pkl",
                 device=args.gpu)
    print(f"Run completed in {time.time() - t_run_start:.2f} seconds.")
    if args.plot:
        import matplotlib.pyplot as plt
        out_dir = "plots/outputs"

        os.makedirs(out_dir, exist_ok=True)
        input_data = {
            'px': muons[:,0],
            'py': muons[:,1],
            'pz': muons[:,2],
            'x': muons[:,3],
            'y': muons[:,4],
            'z': muons[:,5],
            'pdg_id': muons[:,6],
            'weight': muons[:,7]
        }
        print('Number of INPUTS:', muons.shape[0], ' (weighted: ', muons[:,7].sum().item())
        print('Number of OUTPUTS:', output['x'].shape[0], ' (weighted: ', output['weight'].sum().item())
        for key, values in output.items():
            plt.figure()
            plt.hist(input_data[key], bins='auto', histtype='step', label='Input', linewidth=1.5, log=True)
            plt.hist(values, bins='auto', histtype='step', label='Output', linewidth=1.5, log=True)
            plt.title(f"Histogram of {key} (CUDA_MUONS)")
            plt.xlabel(key)
            plt.ylabel("Counts")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{out_dir}/{key}_hist.png")
            plt.close()
        print(f"Histograms saved in {out_dir}")
