import numpy as np
import torch
import h5py

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

def get_cavern(detector):
    if 'cavern' not in detector:
        return torch.tensor([[-30, 30, -30, 30, 0], [-30, 30, -30, 30, 0]], dtype=torch.float32)
    cavern = detector['cavern']
    TCC8 = cavern[0]
    ECN3 = cavern[1]
    TCC8_params = [TCC8['x1'], TCC8['x2'], TCC8['y1'], TCC8['y2'], TCC8['z_center']+TCC8['dz']]
    ECN3_params = [ECN3['x1'], ECN3['x2'], ECN3['y1'], ECN3['y2'], ECN3['z_center']-ECN3['dz']]
    return torch.tensor([TCC8_params,ECN3_params], dtype=torch.float32)

def get_magnetic_field(detector):
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