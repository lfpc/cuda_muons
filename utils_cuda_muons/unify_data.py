import h5py
import argparse
import os

parser = argparse.ArgumentParser(description="Run muon simulations.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--material', type=str, default='G4_Fe', help='Material to use for simulations (G4_Fe, G4_CONCRETE)')
args = parser.parse_args()

dir_data = "data"
file_name_root =  "muon_data_energy_loss_sens_" + args.material + '_'
output_file = os.path.join(dir_data, f"muon_data_energy_loss_sens_{args.material}.h5")

with h5py.File(output_file, 'a') as dst_file:
    for file_name in os.listdir(dir_data):
        if (not file_name.startswith(file_name_root)) or (not file_name.endswith('.h5')):
            continue
        print("Processing file:", file_name)
        with h5py.File(os.path.join(dir_data, file_name), 'r') as src_file:
            for group_name in src_file:
                print("Copying group:", group_name)
                if group_name not in dst_file:
                    src_file.copy(group_name, dst_file)
                    print("Size:", len(src_file[group_name]['px']))
                else:
                    print(f"Group {group_name} already exists in output file, skipping.")
            step_length = src_file.attrs.get('step_length', 0.02)
            dst_file.attrs['step_length'] = step_length
print(f"Unified data saved to {output_file}")