import numpy as np
import torch
import h5py

def get_sphere_design(mag_field, sens_film = None, material="G4_Fe"):
    detector = {
        # "worldPositionX": 0, "worldPositionY": 0, "worldPositionZ": 0, "worldSizeX": 11, "worldSizeY": 11,
        # "worldSizeZ": 100,
        # "magnets": magnets,
        "material": material,
        "magnetic_field": mag_field,
        "type": 3,
        "store_all": True,
        "limits": {
            "max_step_length": -1,
            "minimum_kinetic_energy": -1
        }
    }
    if sens_film is not None:
        detector["sensitive_film"] = sens_film
    return detector


def get_corners_from_params(params: np.ndarray, fSC_mag: bool = False, NI_from_B: bool = True) -> torch.Tensor:
    """
    Build ARB8 corners directly from the magnet parameters, without going through
    the full detector creation. This replicates the corner construction from
    ship_muon_shield_customfield.py but returns only the corners tensor.
    
    Args:
        params: Array of shape (N_magnets, 15) with magnet parameters.
                Each row: [zgap, dZ, dXIn, dXOut, dYIn, dYOut, gapIn, gapOut,
                          ratio_yokesIn, ratio_yokesOut, dY_yokeIn, dY_yokeOut,
                          midGapIn, midGapOut, NI]
        fSC_mag: Whether superconducting magnets are used (affects Ymgap).
        NI_from_B: Whether NI is derived from B (affects SC threshold).
    
    Returns:
        Tensor of shape (N_arb8s, 8, 3) with corner coordinates in meters.
        With use_symmetry=True, returns 3 components per magnet (MainL, TopLeft, RetL).
    """
    SC_Ymgap = 5.0  # cm, gap for superconducting magnets
    anti_overlap = 0.01
    
    params = np.asarray(params)
    if params.ndim == 1:
        params = params.reshape(1, -1)
    
    all_corners = []
    Z = 0.0
    
    for magnet in params:
        zgap = magnet[0]
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
        
        # Skip invalid magnets
        if dZ < 1 or dXIn < 1:
            Z += dZ + zgap
            continue
        
        # Determine if superconducting
        SC_threshold = 3.0 if NI_from_B else 1e6
        is_SC = fSC_mag and (abs(NI) > SC_threshold)
        Ymgap = SC_Ymgap if is_SC else 0.0
        
        # Update Z position
        Z += zgap + dZ
        z_center = Z  # This is the z_center for the magnet
        
        # Apply Ymgap to dY values
        dY = dYIn + Ymgap
        dY2 = dYOut + Ymgap
        dX = dXIn
        dX2 = dXOut
        middleGap = midGapIn
        middleGap2 = midGapOut
        coil_gap = gapIn
        coil_gap2 = gapOut
        ratio_yoke_1 = ratio_yokesIn
        ratio_yoke_2 = ratio_yokesOut
        dY_yoke_1 = dY_yokeIn
        dY_yoke_2 = dY_yokeOut
        
        # Build cornersMainL (16 values: 8 points x 2 coords)
        cornersMainL = np.array([
            middleGap, -(dY + dY_yoke_1) - anti_overlap,
            middleGap, dY + dY_yoke_1 - anti_overlap,
            dX + middleGap, dY - anti_overlap,
            dX + middleGap, -(dY - anti_overlap),
            middleGap2, -(dY2 + dY_yoke_2 - anti_overlap),
            middleGap2, dY2 + dY_yoke_2 - anti_overlap,
            dX2 + middleGap2, dY2 - anti_overlap,
            dX2 + middleGap2, -(dY2 - anti_overlap)
        ])
        
        # Build cornersTL (TopLeft)
        cornersTL = np.array([
            middleGap + dX, dY,
            middleGap, dY + dY_yoke_1,
            dX + ratio_yoke_1 * dX + middleGap + coil_gap, dY + dY_yoke_1,
            dX + middleGap + coil_gap, dY,
            middleGap2 + dX2, dY2,
            middleGap2, dY2 + dY_yoke_2,
            dX2 + ratio_yoke_2 * dX2 + middleGap2 + coil_gap2, dY2 + dY_yoke_2,
            dX2 + middleGap2 + coil_gap2, dY2
        ])
        
        # Build cornersMainSideL (RetL)
        cornersMainSideL = np.array([
            dX + middleGap + gapIn, -dY,
            dX + middleGap + gapIn, dY,
            dX + ratio_yoke_1 * dX + middleGap + gapIn, dY + dY_yoke_1,
            dX + ratio_yoke_1 * dX + middleGap + gapIn, -(dY + dY_yoke_1),
            dX2 + middleGap2 + gapOut, -dY2,
            dX2 + middleGap2 + gapOut, dY2,
            dX2 + ratio_yoke_2 * dX2 + middleGap2 + gapOut, dY2 + dY_yoke_2,
            dX2 + ratio_yoke_2 * dX2 + middleGap2 + gapOut, -(dY2 + dY_yoke_2)
        ])
        
        # Expand corners to 3D (add z coordinates) and convert to meters
        for corners_2d in [cornersMainL, cornersTL, cornersMainSideL]:
            corners_3d = _expand_corners_from_params(corners_2d / 100, dZ / 100, z_center / 100)
            all_corners.append(corners_3d)
        
        # Update Z for next magnet
        Z += dZ
    
    return torch.from_numpy(np.array(all_corners, dtype=np.float32))


def _expand_corners_from_params(corners_2d: np.ndarray, dz: float, z_center: float) -> np.ndarray:
    """
    Expand 2D corners (16 values) to 3D corners (8x3) by adding z coordinates.
    
    Args:
        corners_2d: Array of 16 values (8 points x 2 coords for x,y) in meters.
        dz: Half-length in z in meters.
        z_center: Center z position in meters.
        
    Returns:
        Array of shape (8, 3) with 3D corner coordinates.
    """
    corners_2d = corners_2d.reshape(8, 2)
    z_min = z_center - dz
    z_max = z_center + dz
    z_coords = np.array([z_min] * 4 + [z_max] * 4).reshape(8, 1)
    corners_3d = np.hstack([corners_2d, z_coords])
    return corners_3d


def get_corners_from_detector(detector, use_symmetry=True):

    def expand_corners(corners, dz, z_center):
        corners = np.array(corners)
        z_min = z_center - dz
        z_max = z_center + dz
        corners = corners.reshape(8, 2)
        z = np.full((8, 1), z_min)
        z[4:] = z_max
        corners = np.hstack([corners, z])
        return corners

    all_corners = []
    for magnet in detector['magnets']:
        if use_symmetry: components = magnet['components'][:3]
        else: components = magnet['components']
        for component in components:
            corners = component['corners']
            dz = component['dz']
            z_center = component['z_center'] 
            corners = expand_corners(corners, dz, z_center)
            all_corners.append(corners)
    return torch.from_numpy(np.array(all_corners))

def get_cavern_from_detector(detector):
    if 'cavern' not in detector:
        return torch.tensor([[-30, 30, -30, 30, 0], [-30, 30, -30, 30, 0]], dtype=torch.float32)
    cavern = detector['cavern']
    TCC8 = cavern[0]
    ECN3 = cavern[1]
    TCC8_params = [TCC8['x1'], TCC8['x2'], TCC8['y1'], TCC8['y2'], TCC8['z_center']+TCC8['dz']]
    ECN3_params = [ECN3['x1'], ECN3['x2'], ECN3['y1'], ECN3['y2'], ECN3['z_center']-ECN3['dz']]
    return torch.tensor([TCC8_params,ECN3_params], dtype=torch.float32)


def get_cavern_from_params(add_cavern: bool = True, world_dz: float = 200.0) -> torch.Tensor:
    """
    Build cavern parameters directly without going through detector creation.
    
    This replicates the logic from CreateCavern in ship_muon_shield_customfield.py
    and returns the tensor format expected by the CUDA propagation.
    
    Args:
        add_cavern: If False, returns dummy values that effectively disable cavern.
        world_dz: World size in Z (meters), used to determine cavern length.
    
    Returns:
        Tensor of shape (2, 5) with [x1, x2, y1, y2, z_boundary] for TCC8 and ECN3.
        - For TCC8: z_boundary is z_max (z_center + dz)
        - For ECN3: z_boundary is z_min (z_center - dz)
    """
    if not add_cavern:
        # Return dummy values that effectively disable cavern collision
        return torch.tensor([[-30, 30, -30, 30, 0], [-30, 30, -30, 30, 0]], dtype=torch.float32)
    
    # Constants from ship_muon_shield_customfield.py
    SHIFT = -214  # cm
    CAVERN_TRANSITION = 2051.8 + SHIFT  # cm -> 1837.8 cm
    shift = CAVERN_TRANSITION / 100  # Convert to meters: 18.378 m
    
    length = world_dz  # meters
    
    # TCC8 parameters
    TCC8_length = max(length, 170.0)  # meters
    dX_TCC8 = 5.0  # meters
    dY_TCC8 = 3.75  # meters
    TCC8_x_shift = 1.43  # meters
    TCC8_y_shift = 2.05  # meters
    TCC8_z_shift = -TCC8_length / 2  # meters
    
    TCC8_dz = TCC8_length / 2
    TCC8_z_center = TCC8_z_shift + shift
    TCC8_x1 = TCC8_x_shift - dX_TCC8
    TCC8_x2 = TCC8_x_shift + dX_TCC8
    TCC8_y1 = TCC8_y_shift - dY_TCC8
    TCC8_y2 = TCC8_y_shift + dY_TCC8
    TCC8_z_max = TCC8_z_center + TCC8_dz  # This is the z boundary we care about
    
    # ECN3 parameters
    ECN3_length = max(length, 100.0)  # meters
    dX_ECN3 = 8.0  # meters
    dY_ECN3 = 7.0  # meters
    ECN3_x_shift = 3.43  # meters
    ECN3_y_shift = 3.64  # meters
    ECN3_z_shift = ECN3_length / 2  # meters
    
    ECN3_dz = ECN3_length / 2
    ECN3_z_center = ECN3_z_shift + shift
    ECN3_x1 = ECN3_x_shift - dX_ECN3
    ECN3_x2 = ECN3_x_shift + dX_ECN3
    ECN3_y1 = ECN3_y_shift - dY_ECN3
    ECN3_y2 = ECN3_y_shift + dY_ECN3
    ECN3_z_min = ECN3_z_center - ECN3_dz  # This is the z boundary we care about
    
    TCC8_params = [TCC8_x1, TCC8_x2, TCC8_y1, TCC8_y2, TCC8_z_max]
    ECN3_params = [ECN3_x1, ECN3_x2, ECN3_y1, ECN3_y2, ECN3_z_min]
    
    return torch.tensor([TCC8_params, ECN3_params], dtype=torch.float32)

def get_magnetic_field_from_detector(detector):
    mag_dict = detector['global_field_map']
    if isinstance(mag_dict['B'], str):
        with h5py.File(mag_dict['B'], 'r') as f:
            mag_dict['B'] = f["B"][:]
    return mag_dict

def create_z_axis_grid(corners_tensor: torch.Tensor, sz: int) -> list[list[int]]:
    """
    Builds a Z-axis spatial grid and returns it in a flattened, GPU-friendly
    format (CRS) directly, along with the grid's metadata.

    This function performs the entire grid construction in a single, vectorized
    operation without creating intermediate Python list structures.

    Args:
        corners_tensor (torch.Tensor): A tensor of shape (N, 8, 3) containing the
                                       vertex coordinates for N ARB8 geometries.
        sz (int): The number of cells (slices) to divide the Z-axis into.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]: A tuple containing:
            - cell_starts (torch.Tensor, int32): Start indices for each cell. Shape: (sz + 1,).
            - item_indices (torch.Tensor, int32): Flat array of all geometry indices, grouped by cell.
    """
    all_z_coords = corners_tensor[:, :, 2]
    z_min_global = all_z_coords.min()
    z_max_global = max(all_z_coords.max(),30.0)
    cell_boundaries = torch.linspace(z_min_global, z_max_global, sz + 1)
    cell_z_starts = cell_boundaries[:-1]
    cell_z_ends = cell_boundaries[1:]
    geom_z0 = corners_tensor[:, 0, 2]
    geom_z1 = corners_tensor[:, 7, 2]
    overlap_matrix = torch.logical_not((geom_z0.unsqueeze(0) > cell_z_ends.unsqueeze(1)).logical_or(geom_z1.unsqueeze(0) < cell_z_starts.unsqueeze(1)))
    cell_indices_flat, geom_indices_flat = torch.where(overlap_matrix)
    item_indices = geom_indices_flat.to(torch.int32)
    counts = torch.bincount(cell_indices_flat, minlength=sz)
    zero_prefix = torch.tensor([0], dtype=torch.int32)
    cell_starts = torch.cat((zero_prefix, counts.cumsum(dim=0))).to(torch.int32)
    return cell_starts, item_indices