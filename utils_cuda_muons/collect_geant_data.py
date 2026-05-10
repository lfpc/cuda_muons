import argparse
import h5py
import time

import numpy as np
import json
import multiprocessing as mp
import os
from muon_slabs import simulate_muon, initialize, kill_secondary_tracks, collect_from_sensitive, collect, is_single_step
from get_geometry import get_sphere_design
import functools


def simulate_muon_batch(num_simulations, detector, step_size=None, initial_momenta_bounds=(10, 400)):
    np.random.seed((os.getpid() * int(time.time())) % 2**16)
    single_step = step_size is None or step_size == 0
    batch_data = {
        'initial_momenta': [],
        'px': [],
        'py': [],
        'pz': [],
    }
    initialize(np.random.randint(32000), np.random.randint(32000), np.random.randint(32000), np.random.randint(32000), json.dumps(detector))
    #kill_secondary_tracks(True)
    if single_step:
        is_single_step(True)
        batch_data['step_length'] = []
    i = 0
    while i < num_simulations:
        initial_momenta = np.random.uniform(max(initial_momenta_bounds[0], 0.18), initial_momenta_bounds[1])
        simulate_muon(0, 0, initial_momenta, 1, 0, 0, 0)
        if not single_step: 
            data = collect_from_sensitive()
            index = 0
            if len(data['z']) > 0 and 13 in np.abs(data['pdg_id']):
                while int(abs(data['pdg_id'][index])) != 13:
                    index += 1
            else:
                continue
        else: 
            data = collect()
            index = 0; 
            assert len(data['px']) == 1, f"Expected single step data, got {len(data['px'])} entries"
        batch_data['initial_momenta'].append(initial_momenta)
        batch_data['px'].append(data['px'][index])
        batch_data['py'].append(data['py'][index])
        batch_data['pz'].append(data['pz'][index])
        if single_step:
            step_length = data['step_length'][index]
            batch_data['step_length'].append(step_length)
        i += 1
    assert len(batch_data['px']) == num_simulations, f"Expected {num_simulations} simulations, got {len(batch_data['px'])}"
    return batch_data

def parallel_simulations(num_sims, detector, num_processes=4, step_size=None, output_file='muons_data.h5', initial_momenta_bounds=(10, 400)):
    single_step = step_size is None or step_size == 0
    with h5py.File(output_file, 'a') as f:
        if not single_step:
            f.attrs['step_length'] = int(step_size * 100)
        grp = f.create_group(str(initial_momenta_bounds))
        grp.create_dataset('initial_momenta', (num_sims,), dtype='f8')
        grp.create_dataset('px', (num_sims,), dtype='f8')
        grp.create_dataset('py', (num_sims,), dtype='f8')
        grp.create_dataset('pz', (num_sims,), dtype='f8')
        if single_step:
            grp.create_dataset('step_length', (num_sims,), dtype='f8')
        simulate_fn = functools.partial(simulate_muon_batch, detector=detector, step_size=step_size, initial_momenta_bounds=initial_momenta_bounds)
        chunk_size = num_sims // num_processes
        remainder = num_sims % num_processes
        chunks = [chunk_size + (1 if i < remainder else 0) for i in range(num_processes)]
        total_written = 0
        assert len(chunks) == num_processes, f"Expected {num_processes} chunks, got {len(chunks)}"
        with mp.Pool(processes=num_processes) as pool:
            for batch_result in pool.imap_unordered(simulate_fn, chunks):
                batch_size = len(batch_result['pz'])
                new_size = total_written + batch_size
                grp['initial_momenta'][total_written:new_size] = batch_result['initial_momenta']
                grp['px'][total_written:new_size] = batch_result['px']
                grp['py'][total_written:new_size] = batch_result['py']
                grp['pz'][total_written:new_size] = batch_result['pz']
                if single_step:
                    grp['step_length'][total_written:new_size] = batch_result['step_length']
                f.flush()
                total_written += batch_size
                print(f"Written batch of {batch_size} simulations. Total: {total_written}/{num_sims}")
    print(f"Completed writing {total_written} simulations to {output_file}")
    return output_file

def main(cores=16, num_sims=1_000_000, step_size=0.02, initial_momenta=(10, 400), tag:str = '5', material='G4_Fe', logspace=True):
    folder = f"data/"
    if not os.path.exists(folder):
        os.makedirs(folder)
    single_step = step_size is None or step_size == 0
    if single_step:
        detector = get_sphere_design(mag_field=[0., 0., 0.], material=material)
    else:
        sensitive_film = {"name": "SensitiveFilm", "radius": step_size, "dr": 0.001, "shape": "sphere"}
        detector = get_sphere_design(mag_field=[0., 0., 0.], sens_film=sensitive_film, material=material)
        detector["limits"]["max_step_length"] = step_size
    detector["store_primary"] = single_step
    detector["store_all"] = False
    output_file = os.path.join(folder, f"muon_data_energy_loss_sens_{tag}.h5")
    with h5py.File(output_file, 'w') as f:
        pass
    if logspace:
        momenta_points = np.logspace(np.log10(max(initial_momenta[0], 0.18)), np.log10(initial_momenta[1]), num=int(initial_momenta[2]+1))
        for i, m0 in enumerate(momenta_points[:-1]):
            m0 = float(m0)
            m1 = float(momenta_points[i+1])
            print(f"Simulating for initial momenta in range ({m0}, {m1})")
            parallel_simulations(num_sims, detector, num_processes=cores, step_size=step_size, output_file=output_file, initial_momenta_bounds=(m0, m1))
    else:
        m0 = initial_momenta[0]
        while m0 < initial_momenta[1]:
            m0 = float(m0)
            print(f"Simulating for initial momenta in range ({m0}, {m0+initial_momenta[2]})")
            parallel_simulations(num_sims, detector, num_processes=cores, step_size=step_size, output_file=output_file, initial_momenta_bounds=(m0, m0+initial_momenta[2]))
            m0 += initial_momenta[2]
    print(f"Data saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run muon simulations.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cores', type=int, default=64, help='Number of CPU cores to use')
    parser.add_argument('--material', type=str, default='G4_Fe', help='Material to use for simulations (G4_Fe, G4_CONCRETE)')
    parser.add_argument('--num_sims', type=int, default=5_000_000, help='Number of simulations to run')
    parser.add_argument('--step_size', type=float, default=0.02, help='Step size for simulation; 0 for single natural step mode')
    parser.add_argument('--tag', type=str, default='', help='Tag to identify the output file')
    parser.add_argument('--initial_momenta', type=float, nargs=3, default=(0.18, 400, 95), help='Range of initial momenta for muons (min, max, bins)')
    parser.add_argument('--linearspace', dest='logspace', action='store_false', help='Use linear spacing for initial momenta ranges (default is logspace)')
    args = parser.parse_args()
    main(cores=args.cores, num_sims=args.num_sims, step_size=args.step_size, tag=str(args.material)+args.tag, initial_momenta=args.initial_momenta, material=args.material, logspace=args.logspace)
