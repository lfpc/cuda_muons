"""
Magnetic field utilities for CUDA muon propagation.

This module provides functions to get magnetic fields from magnet parameters:
- Uniform field per ARB8 block (fast, no simulation)
- Simulated field map via FEM (requires snoopy)
"""
import os
import numpy as np
import torch
import h5py
import multiprocessing as mp
from time import time
from typing import Union, Dict
from os.path import exists
import pandas as pd
from scipy.spatial import cKDTree

# Try to import snoopy for FEM simulation (optional)
try:
    import snoopy
    SNOOPY_AVAILABLE = True
except ImportError:
    SNOOPY_AVAILABLE = False

# Constants
SC_YMGAP = 15  # cm, gap for superconducting magnets
RESOL_DEF = (2, 2, 5)  # Default resolution in cm (x, y, z)


def get_magnetic_field_from_params(
    params: np.ndarray,
    simulate_fields: bool = False,
    field_map_file: str = None,
    fSC_mag: bool = False,
    NI_from_B: bool = True,
    use_diluted: bool = False,
    cores_field: int = 1,
) -> Union[Dict, torch.Tensor]:
    """
    Get magnetic field from magnet parameters.
    
    Args:
        params: Array of shape (N_magnets, 15) with magnet parameters.
                Each row: [zgap, dZ, dXIn, dXOut, dYIn, dYOut, gapIn, gapOut,
                          ratio_yokesIn, ratio_yokesOut, dY_yokeIn, dY_yokeOut,
                          midGapIn, midGapOut, NI]
        simulate_fields: If True, run FEM simulation and return field map dict.
        field_map_file: Path to save/load field map. If exists and simulate_fields=False,
                       loads from file.
        fSC_mag: Whether superconducting magnets are used.
        NI_from_B: Whether NI is derived from B (affects SC threshold).
        use_diluted: Whether to use diluted steel.
        cores_field: Number of CPU cores for field simulation.
    
    Returns:
        If simulate_fields=True or field_map_file is provided:
            Dict with 'B' (field array), 'range_x', 'range_y', 'range_z'
        Otherwise:
            Tensor of shape (N_arb8s, 3) with uniform [Bx, By, Bz] per ARB8.
            With use_symmetry=True, returns 3 fields per magnet (MainL, TopLeft, RetL).
    """
    params = np.asarray(params)
    if params.ndim == 1:
        params = params.reshape(1, -1)
    params = np.round(params, 2)
    
    # If simulating fields or loading from file, return field map dict
    if simulate_fields or field_map_file is not None:
        return _get_field_map(params, simulate_fields, field_map_file, 
                              fSC_mag, NI_from_B, use_diluted, cores_field)
    else:
        # Return uniform fields per ARB8
        return _get_uniform_fields(params, fSC_mag, NI_from_B)


# =============================================================================
# Uniform Field Mode
# =============================================================================

def _get_uniform_fields(
    params: np.ndarray,
    fSC_mag: bool = False,
    NI_from_B: bool = True,
) -> torch.Tensor:
    """
    Generate uniform magnetic field per ARB8 block.
    
    The field assignment follows the same logic as ship_muon_shield_customfield.py:
    - For each magnet with use_symmetry=True, we have 3 components:
      1. MainL (MiddleMagL): B = [0, ironField, 0]
      2. TopLeft: B = [ironField/ratio_yoke, 0, 0]  
      3. RetL (MagRetL): B = [0, -ironField/ratio_yoke, 0]
    
    Args:
        params: Array of shape (N_magnets, 15) with magnet parameters.
        fSC_mag: Whether superconducting magnets are used.
        NI_from_B: Whether NI is derived from B (affects SC threshold).
    
    Returns:
        Tensor of shape (N_arb8s, 3) with [Bx, By, Bz] per ARB8 in Tesla.
    """
    SC_threshold = 3.0 if NI_from_B else 1e6
    
    all_fields = []
    
    for magnet in params:
        dZ = magnet[1]
        dXIn = magnet[2]
        ratio_yokesIn = magnet[8]
        NI = magnet[14]
        
        # Skip invalid magnets
        if dZ < 1 or dXIn < 1:
            continue

        
        ironField = NI
        
        # Use ratio_yoke for scaling return/corner fields
        ratio_yoke = ratio_yokesIn if ratio_yokesIn > 0 else 1.0
        
        # Field assignments matching the order from get_corners_from_params:
        # 1. MainL (MiddleMagL): vertical field in iron core
        magFieldIron = [0.0, ironField, 0.0]
        
        # 2. TopLeft: horizontal field in top corner (ConLField)
        conLField = [ironField / ratio_yoke, 0.0, 0.0]
        
        # 3. RetL (MagRetL): vertical field in return yoke (opposite direction)
        retField = [0.0, -ironField / ratio_yoke, 0.0]
        
        # Append in the same order as corners are built
        all_fields.append(magFieldIron)
        all_fields.append(conLField)
        all_fields.append(retField)
    
    return torch.tensor(all_fields, dtype=torch.float32)


# =============================================================================
# Field Map Simulation Mode (requires snoopy)
# =============================================================================

def _get_field_map(
    params: np.ndarray,
    simulate_fields: bool,
    field_map_file: str,
    fSC_mag: bool,
    NI_from_B: bool,
    use_diluted: bool,
    cores_field: int,
) -> Dict:
    """Get field map from simulation or file."""
    
    # Calculate d_space extents
    d_space, resol = _compute_field_extents(params, fSC_mag, NI_from_B)
    
    # Determine if we need to simulate or can load from file
    should_simulate = simulate_fields or (field_map_file is not None and not exists(field_map_file))
    
    if should_simulate:
        if not SNOOPY_AVAILABLE:
            raise ImportError("snoopy module not available. Cannot simulate fields. "
                             "Install snoopy or provide a pre-computed field_map_file.")
        fields = _simulate_field(
            params, 
            file_name=field_map_file,
            fSC_mag=fSC_mag,
            d_space=d_space,
            NI_from_B=NI_from_B,
            cores=cores_field,
            use_diluted=use_diluted
        )
    elif field_map_file is not None and exists(field_map_file):
        print('Using field map from file', field_map_file)
        with h5py.File(field_map_file, 'r') as f:
            fields = f["B"][:]
    else:
        raise ValueError(f"Field map file {field_map_file} does not exist and simulate_fields=False")
    
    # Return in the format expected by propagate_muons_with_cuda
    mag_dict = {
        'B': fields,
        'range_x': [d_space[0][0], d_space[0][1], resol[0]],
        'range_y': [d_space[1][0], d_space[1][1], resol[1]],
        'range_z': [d_space[2][0], d_space[2][1], resol[2]]
    }
    
    return mag_dict


def _compute_field_extents(params: np.ndarray, fSC_mag: bool, NI_from_B: bool):
    """Compute the spatial extents for field simulation."""
    SC_threshold = 3.0 if NI_from_B else 1e6
    
    max_x = 0.0
    max_y = 0.0
    length = 2 * params[:, 1].sum()  # 2 * sum of dZ values
    
    for magnet in params:
        dZ = magnet[1]
        dXIn = magnet[2]
        dXOut = magnet[3]
        dYIn = magnet[4]
        dYOut = magnet[5]
        gapIn = magnet[6]
        gapOut = magnet[7]
        ratio_yokesIn = magnet[8]
        ratio_yokesOut = magnet[9]
        dY_yokeIn = magnet[10]
        dY_yokeOut = magnet[11]
        midGapIn = magnet[12]
        midGapOut = magnet[13]
        NI = magnet[14]
        
        if dZ < 1 or dXIn < 1:
            continue
            
        is_SC = fSC_mag and (abs(NI) > SC_threshold)
        Ymgap = SC_YMGAP if is_SC else 0.0
        
        max_x = max(max_x, 
                   dXIn + dXIn * ratio_yokesIn + gapIn + midGapIn,
                   dXOut + dXOut * ratio_yokesOut + gapOut + midGapOut)
        max_y = max(max_y,
                   dYIn + dY_yokeIn + Ymgap,
                   dYOut + dY_yokeOut + Ymgap)
    
    resol = RESOL_DEF
    max_x = int((max_x // resol[0]) * resol[0])
    max_y = int((max_y // resol[1]) * resol[1])
    d_space = ((0, max_x + 50), (0, max_y + 50), 
               (-50, int(((length + 200) // resol[2]) * resol[2])))
    
    return d_space, resol


def _get_fixed_params(yoke_type: str = 'Mag1', mesh_size_parameter: float = 0.15) -> dict:
    """Get fixed parameters for magnet simulation."""
    SC = (yoke_type == 'Mag2')
    return {
        'yoke_type': yoke_type,
        'coil_material': 'hts_pencake.json' if SC else 'copper_water_cooled.json',
        'max_turns': 12 if SC else 10,
        'J_tar(A/mm2)': 583 if SC else -1,
        'coil_diam(mm)': 40 if SC else 9,
        'insulation(mm)': 1 if SC else 0.5,
        'winding_radius(mm)': 200 if SC else 0,
        'yoke_spacer(mm)': 5,
        'material': 'aisi1010.json',
        'field_density': 5,
        'delta_x(m)': 1 if SC else 0.5,
        'delta_y(m)': 1 if SC else 0.5,
        'delta_z(m)': 1 if SC else 0.5,
        'mesh_size_parameter': mesh_size_parameter,
    }


def _get_magnet_params(
    params: np.ndarray,
    Ymgap: float = 0.0,
    yoke_type: str = 'Mag1',
    resol: tuple = RESOL_DEF,
    use_B_goal: bool = False,
    materials_directory: str = None,
    use_diluted: bool = False,
) -> dict:
    """Build parameter dict for a single magnet."""
    
    ratio_yoke_1 = params[8]
    ratio_yoke_2 = params[9]
    B_NI = params[14]
    params_m = params / 100  # Convert to meters
    Xmgap_1 = params_m[12]
    Xmgap_2 = params_m[13]
    
    d = _get_fixed_params(yoke_type)
    d.update({
        'resol_x(m)': resol[0] / 100,
        'resol_y(m)': resol[1] / 100,
        'resol_z(m)': resol[2] / 100,
        'Z_pos(m)': -1 * params_m[1],
        'Xmgap1(m)': Xmgap_1,
        'Xmgap2(m)': Xmgap_2,
        'Z_len(m)': 2 * params_m[1],
        'Xcore1(m)': params_m[2] + Xmgap_1,
        'Xvoid1(m)': params_m[2] + params_m[6] + Xmgap_2,
        'Xyoke1(m)': params_m[2] + params_m[6] + ratio_yoke_1 * params_m[2] + Xmgap_1,
        'Xcore2(m)': params_m[3] + Xmgap_2,
        'Xvoid2(m)': params_m[3] + params_m[7] + Xmgap_2,
        'Xyoke2(m)': params_m[3] + params_m[7] + ratio_yoke_2 * params_m[3] + Xmgap_2,
        'Ycore1(m)': params_m[4],
        'Yvoid1(m)': params_m[4] + Ymgap / 100,
        'Yyoke1(m)': params_m[4] + params_m[10] + Ymgap / 100,
        'Ycore2(m)': params_m[5],
        'Yvoid2(m)': params_m[5] + Ymgap / 100,
        'Yyoke2(m)': params_m[5] + params_m[11] + Ymgap / 100
    })
    
    if use_B_goal and SNOOPY_AVAILABLE:
        if materials_directory is None:
            materials_directory = os.path.join(
                os.getenv('PROJECTS_DIR', ''), 
                'MuonsAndMatter/data/materials'
            )
        d['NI(A)'] = snoopy.get_NI(abs(B_NI), pd.DataFrame([d]), 0, 
                                   materials_directory=materials_directory)[0].item()
        d['NI(A)'] = min(d['NI(A)'], 4e6)
        if (B_NI > 0 and d['yoke_type'] == 'Mag3') or (B_NI < 0 and d['yoke_type'] == 'Mag1'):
            d['NI(A)'] = -d['NI(A)']
    elif use_diluted:
        d['NI(A)'] = B_NI
    else:
        d['NI(A)'] = abs(B_NI)

    if use_diluted and d['yoke_type'] == 'Mag3':
        d['yoke_type'] = 'Mag1'
    
    return d


def _construct_grid(limits: tuple, resol: tuple = RESOL_DEF):
    """Construct a 3D grid for field interpolation."""
    (min_x, min_y, min_z), (max_x, max_y, max_z) = limits
    r_x, r_y, r_z = resol
    
    nx = int(round((max_x - min_x) / r_x)) + 1
    ny = int(round((max_y - min_y) / r_y)) + 1
    nz = int(round((max_z - min_z) / r_z)) + 1
    
    X = np.linspace(min_x, max_x, nx)
    Y = np.linspace(min_y, max_y, ny)
    Z = np.linspace(min_z, max_z, nz)
    X, Y, Z = np.meshgrid(X, Y, Z)
    
    return X / 100, Y / 100, Z / 100


def _get_grid_data(points: np.ndarray, B: np.ndarray, new_points: tuple, method: str = 'nearest'):
    """Interpolate magnetic field data to a new grid."""
    t1 = time()
    
    new_points_stacked = np.column_stack((
        new_points[0].ravel(),
        new_points[1].ravel(),
        new_points[2].ravel()
    ))
    
    if method == 'nearest':
        Bx_out, By_out, Bz_out = np.zeros_like(new_points_stacked).T
        
        hull = ((new_points_stacked[:, 0] <= points[:, 0].max()) & 
                (new_points_stacked[:, 1] <= points[:, 1].max()) & 
                (new_points_stacked[:, 2] >= points[:, 2].min()) & 
                (new_points_stacked[:, 2] <= points[:, 2].max()))
        
        tree = cKDTree(points)
        _, idx = tree.query(new_points_stacked[hull], k=1)
        Bx_out[hull] = B[idx, 0]
        By_out[hull] = B[idx, 1]
        Bz_out[hull] = B[idx, 2]
        
        new_B = np.column_stack((Bx_out, By_out, Bz_out))
    else:
        raise ValueError(f"Unknown method: {method}. Use 'nearest'.")
    
    print(f'Gridding / Interpolation time ({method}) = {time() - t1:.4f} sec')
    return new_points_stacked, new_B


def _get_vector_field(magn_params: dict, materials_dir: str, use_diluted: bool = False):
    """Get vector field from FEM simulation via snoopy."""
    if 'Mag2' in magn_params['yoke_type']:
        points, B, M_i, M_c, Q, J = snoopy.get_vector_field_ncsc(
            magn_params, 0, materials_directory=materials_dir)
    elif magn_params['yoke_type'][0] == 'Mag1':
        points, B, M_i, M_c, Q, J = snoopy.get_vector_field_mag_1(
            magn_params, 0, materials_directory=materials_dir, use_diluted_steel=use_diluted)
    elif magn_params['yoke_type'][0] == 'Mag3':
        points, B, M_i, M_c, Q, J = snoopy.get_vector_field_mag_3(
            magn_params, 0, materials_directory=materials_dir, use_diluted_steel=use_diluted)
    else:
        raise ValueError(f'Invalid yoke type - Received {magn_params["yoke_type"][0]}')
    return points, B.round(4)


def _run_fem(magn_params: dict, use_diluted: bool = False):
    """Run FEM simulation for a single magnet configuration."""
    materials_dir = os.path.join(
        os.getenv('PROJECTS_DIR', ''),
        'MuonsAndMatter/data/materials'
    )
    start = time()
    points, B = _get_vector_field(magn_params, materials_dir, use_diluted=use_diluted)
    print(f'FEM Computation time = {time() - start:.2f} sec')
    return {'points': points, 'B': B}


def _simulate_and_grid(params: dict, points: tuple, use_diluted: bool = False):
    """Simulate field for one magnet and interpolate to grid."""
    result = _run_fem(params, use_diluted=use_diluted)
    return _get_grid_data(result['points'], result['B'], new_points=points)[1]


def _run_magnets(
    magn_params: dict,
    d_space: tuple,
    cores: int = 1,
    use_diluted: bool = False
) -> dict:
    """Run FEM simulation for all magnets and combine results."""
    n_magnets = len(magn_params['yoke_type'])
    print(f'Starting simulation for {n_magnets} magnets')
    
    limits_quadrant = (
        (d_space[0][0], d_space[1][0], d_space[2][0]), 
        (d_space[0][1], d_space[1][1], d_space[2][1])
    )
    points = _construct_grid(limits=limits_quadrant, resol=RESOL_DEF)
    
    # Split parameters for each magnet
    params_split = [
        ({k: [v[i]] for k, v in magn_params.items()}, points, use_diluted) 
        for i in range(n_magnets)
    ]
    
    # Handle SC magnet pairs (Mag2)
    if n_magnets > 1 and params_split[1][0]['yoke_type'][0] == 'Mag2':
        for k in magn_params.keys():
            params_split[0][0][k] += params_split[1][0][k]
        params_split.pop(1)
    
    # Run simulations in parallel
    with mp.Pool(cores) as pool:
        B = pool.starmap(_simulate_and_grid, params_split)
    B = np.sum(B, axis=0)
    
    points = np.column_stack([points[i].ravel() for i in range(3)])
    
    return {
        'points': points.round(4).astype(np.float16), 
        'B': B.round(4).astype(np.float16)
    }


def _simulate_field(
    params: np.ndarray,
    Z_init: float = 0,
    fSC_mag: bool = True,
    d_space: tuple = ((0., 400.), (0., 400.), (-100, 300.)),
    NI_from_B: bool = True,
    file_name: str = None,
    cores: int = 1,
    use_diluted: bool = False
) -> np.ndarray:
    """
    Simulate magnetic field for all magnets.
    
    Returns:
        Field array B of shape (N_points, 3).
    """
    t1 = time()
    all_params = pd.DataFrame()
    Z_pos = 0.0
    SC_threshold = 3.0 if NI_from_B else 1e6
    
    for mag_params in params:
        is_SC = fSC_mag and (abs(mag_params[14]) > SC_threshold)
        Z_pos += mag_params[0] / 100
        
        if mag_params[1] < 1:
            continue
        if mag_params[2] < 1:
            Z_pos += 2 * mag_params[1] / 100
            continue
        
        if is_SC:
            Ymgap = SC_YMGAP
            yoke_type = 'Mag2'
        elif use_diluted:
            Ymgap = 0.0
            yoke_type = 'Mag1'
        else:
            Ymgap = 0.0
            yoke_type = 'Mag3' if mag_params[14] < 0 else 'Mag1'
        
        p = _get_magnet_params(
            mag_params, 
            Ymgap=Ymgap, 
            use_B_goal=NI_from_B, 
            yoke_type=yoke_type,
            use_diluted=use_diluted
        )
        p['Z_pos(m)'] = Z_pos
        all_params = pd.concat([all_params, pd.DataFrame([p])], ignore_index=True)
        Z_pos += p['Z_len(m)']
    
    # Run simulation
    fields = _run_magnets(
        all_params.to_dict(orient='list'), 
        d_space=d_space, 
        cores=cores, 
        use_diluted=use_diluted
    )
    fields['points'][:, 2] += Z_init / 100
    
    print(f'Magnetic field simulation took {time() - t1:.2f} seconds')
    
    # Save to file if requested
    if file_name is not None:
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        all_params.to_csv(
            os.path.join(os.path.dirname(file_name), 'magnet_params.csv'), 
            index=False
        )
        
        t_save = time()
        with h5py.File(file_name, "w") as f:
            f.create_dataset("B", data=fields['B'].astype(np.float16), compression=None)
            d_space_arr = np.array([
                [d_space[0][0], d_space[0][1], RESOL_DEF[0]],
                [d_space[1][0], d_space[1][1], RESOL_DEF[1]],
                [d_space[2][0], d_space[2][1], RESOL_DEF[2]]
            ], dtype=np.int16)
            f.create_dataset("d_space", data=d_space_arr, compression=None)
        
        print(f'Fields saved to {file_name} ({time() - t_save:.2f} sec)')
    
    return fields['B']


# =============================================================================
# Legacy Interface
# =============================================================================

def get_uniform_fields_from_detector(detector) -> torch.Tensor:
    """
    Extract uniform magnetic fields from a detector dict (legacy interface).
    
    This extracts the 'field' values from each magnet component when
    field_profile is 'uniform'. Only returns first 3 components per magnet
    (use_symmetry=True mode).
    
    Args:
        detector: Detector dict from get_design_from_params with uniform field_profile.
    
    Returns:
        Tensor of shape (N_arb8s, 3) with [Bx, By, Bz] per ARB8.
    """
    all_fields = []
    
    for magnet in detector['magnets']:
        components = magnet['components'][:3]  # use_symmetry=True
        for component in components:
            field = component.get('field', [0.0, 0.0, 0.0])
            all_fields.append(field)
    
    return torch.tensor(all_fields, dtype=torch.float32)
