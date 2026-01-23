#include <torch/extension.h>



// Forward declaration of the CUDA function
torch::Tensor propagate_muons_cuda(torch::Tensor muon_data, torch::Tensor loss_hists, torch::Tensor bin_widths, torch::Tensor bin_centers, int num_steps);

//torch::Tensor propagate_muons_cuda(torch::Tensor input1, torch::Tensor input2, int num_steps);
// Define a wrapper function for the propagate_muons operation
torch::Tensor propagate_muons(torch::Tensor muon_data, torch::Tensor loss_hists, torch::Tensor bin_widths, torch::Tensor bin_centers, int num_steps) {
    // Check that inputs are CUDA tensors
    if (!muon_data.is_cuda() || !loss_hists.is_cuda() || !bin_widths.is_cuda() || !bin_centers.is_cuda()) {
        throw std::runtime_error("All tensors must be CUDA tensors.");
    }

    // Call the CUDA implementation
    return propagate_muons_cuda(muon_data, loss_hists, bin_widths, bin_centers, num_steps);
}



// Forward declaration of the CUDA function for uniform field per ARB8
void propagate_muons_with_alias_sampling_cuda_uniform_field(
    torch::Tensor muon_data_positions,
    torch::Tensor muon_data_momenta,
    torch::Tensor charges,
    torch::Tensor hist_2d_probability_table_iron,
    torch::Tensor hist_2d_alias_table_iron,
    torch::Tensor hist_2d_bin_centers_first_dim_iron,
    torch::Tensor hist_2d_bin_centers_second_dim_iron,
    torch::Tensor hist_2d_bin_widths_first_dim_iron,
    torch::Tensor hist_2d_bin_widths_second_dim_iron,
    torch::Tensor hist_2d_probability_table_concrete,
    torch::Tensor hist_2d_alias_table_concrete,
    torch::Tensor hist_2d_bin_centers_first_dim_concrete,
    torch::Tensor hist_2d_bin_centers_second_dim_concrete,
    torch::Tensor hist_2d_bin_widths_first_dim_concrete,
    torch::Tensor hist_2d_bin_widths_second_dim_concrete,
    torch::Tensor arb8s,
    torch::Tensor hashed3d_arb8s_cells,
    torch::Tensor hashed3d_arb8s_indices,
    torch::Tensor arb8s_fields,
    torch::Tensor cavern_params,
    bool use_symmetry,
    float sensitive_plane_z,
    float kill_at,
    int num_steps,
    float step_length_fixed,
    int seed
);

// Forward declaration of the CUDA function for magnetic field map
void propagate_muons_with_alias_sampling_cuda(
    torch::Tensor muon_data_positions,
    torch::Tensor muon_data_momenta,
    torch::Tensor charges,
    torch::Tensor hist_2d_probability_table_iron,
    torch::Tensor hist_2d_alias_table_iron,
    torch::Tensor hist_2d_bin_centers_first_dim_iron,
    torch::Tensor hist_2d_bin_centers_second_dim_iron,
    torch::Tensor hist_2d_bin_widths_first_dim_iron,
    torch::Tensor hist_2d_bin_widths_second_dim_iron,
    torch::Tensor hist_2d_probability_table_concrete,
    torch::Tensor hist_2d_alias_table_concrete,
    torch::Tensor hist_2d_bin_centers_first_dim_concrete,
    torch::Tensor hist_2d_bin_centers_second_dim_concrete,
    torch::Tensor hist_2d_bin_widths_first_dim_concrete,
    torch::Tensor hist_2d_bin_widths_second_dim_concrete,
    torch::Tensor arb8s,
    torch::Tensor hashed3d_arb8s_cells,
    torch::Tensor hashed3d_arb8s_indices,
    torch::Tensor magnetic_field,
    torch::Tensor magnetic_field_range,
    torch::Tensor cavern_params,
    bool use_symmetry,
    float sensitive_plane_z,
    float kill_at,
    int num_steps,
    float step_length_fixed,
    int seed
);

//torch::Tensor propagate_muons_with_alias_sampling_cuda(torch::Tensor input1, torch::Tensor input2, int num_steps);
// Define a wrapper function for the propagate_muons_with_alias_sampling operation (uniform field version)
void propagate_muons_with_alias_sampling_uniform_field(
    torch::Tensor muon_data_positions,
    torch::Tensor muon_data_momenta,
    torch::Tensor charges,
    torch::Tensor hist_2d_probability_table_iron,
    torch::Tensor hist_2d_alias_table_iron,
    torch::Tensor hist_2d_bin_centers_first_dim_iron,
    torch::Tensor hist_2d_bin_centers_second_dim_iron,
    torch::Tensor hist_2d_bin_widths_first_dim_iron,
    torch::Tensor hist_2d_bin_widths_second_dim_iron,
    torch::Tensor hist_2d_probability_table_concrete,
    torch::Tensor hist_2d_alias_table_concrete,
    torch::Tensor hist_2d_bin_centers_first_dim_concrete,
    torch::Tensor hist_2d_bin_centers_second_dim_concrete,
    torch::Tensor hist_2d_bin_widths_first_dim_concrete,
    torch::Tensor hist_2d_bin_widths_second_dim_concrete,
    torch::Tensor arb8s,
    torch::Tensor hashed3d_arb8s_cells,
    torch::Tensor hashed3d_arb8s_indices,
    torch::Tensor arb8s_fields,
    torch::Tensor cavern_params,
    bool use_symmetry,
    float sensitive_plane_z,
    float kill_at,
    int num_steps,
    float step_length_fixed,
    int seed
) {
    // Check that inputs are CUDA tensors
    if (!muon_data_positions.is_cuda() ||
        !muon_data_momenta.is_cuda() ||
        !hist_2d_probability_table_iron.is_cuda() ||
        !hist_2d_alias_table_iron.is_cuda() ||
        !hist_2d_bin_centers_first_dim_iron.is_cuda() ||
        !hist_2d_bin_centers_second_dim_iron.is_cuda() ||
        !hist_2d_bin_widths_first_dim_iron.is_cuda() ||
        !hist_2d_bin_widths_second_dim_iron.is_cuda() ||
        !hist_2d_probability_table_concrete.is_cuda() ||
        !hist_2d_alias_table_concrete.is_cuda() ||
        !hist_2d_bin_centers_first_dim_concrete.is_cuda() ||
        !hist_2d_bin_centers_second_dim_concrete.is_cuda() ||
        !hist_2d_bin_widths_first_dim_concrete.is_cuda() ||
        !hist_2d_bin_widths_second_dim_concrete.is_cuda() ||
        !arb8s.is_cuda() ||
        !arb8s_fields.is_cuda() ||
        !cavern_params.is_cuda()
        )
    {
        throw std::runtime_error("All tensors must be CUDA tensors.");
    }

    // Call the CUDA implementation for uniform field
    propagate_muons_with_alias_sampling_cuda_uniform_field(
        muon_data_positions,
        muon_data_momenta,
        charges,
        hist_2d_probability_table_iron,
        hist_2d_alias_table_iron,
        hist_2d_bin_centers_first_dim_iron,
        hist_2d_bin_centers_second_dim_iron,
        hist_2d_bin_widths_first_dim_iron,
        hist_2d_bin_widths_second_dim_iron,
        hist_2d_probability_table_concrete,
        hist_2d_alias_table_concrete,
        hist_2d_bin_centers_first_dim_concrete,
        hist_2d_bin_centers_second_dim_concrete,
        hist_2d_bin_widths_first_dim_concrete,
        hist_2d_bin_widths_second_dim_concrete,
        arb8s,
        hashed3d_arb8s_cells,
        hashed3d_arb8s_indices,
        arb8s_fields,
        cavern_params,
        use_symmetry,
        sensitive_plane_z,
        kill_at,
        num_steps,
        step_length_fixed,
        seed
    );
}

// Define a wrapper function for the propagate_muons_with_alias_sampling operation (field map version)
void propagate_muons_with_alias_sampling(
    torch::Tensor muon_data_positions,
    torch::Tensor muon_data_momenta,
    torch::Tensor charges,
    torch::Tensor hist_2d_probability_table_iron,
    torch::Tensor hist_2d_alias_table_iron,
    torch::Tensor hist_2d_bin_centers_first_dim_iron,
    torch::Tensor hist_2d_bin_centers_second_dim_iron,
    torch::Tensor hist_2d_bin_widths_first_dim_iron,
    torch::Tensor hist_2d_bin_widths_second_dim_iron,
    torch::Tensor hist_2d_probability_table_concrete,
    torch::Tensor hist_2d_alias_table_concrete,
    torch::Tensor hist_2d_bin_centers_first_dim_concrete,
    torch::Tensor hist_2d_bin_centers_second_dim_concrete,
    torch::Tensor hist_2d_bin_widths_first_dim_concrete,
    torch::Tensor hist_2d_bin_widths_second_dim_concrete,
    torch::Tensor arb8s,
    torch::Tensor hashed3d_arb8s_cells,
    torch::Tensor hashed3d_arb8s_indices,
    torch::Tensor magnetic_field,
    torch::Tensor magnetic_field_range,
    torch::Tensor cavern_params,
    bool use_symmetry,
    float sensitive_plane_z,
    float kill_at,
    int num_steps,
    float step_length_fixed,
    int seed
) {
    // Check that inputs are CUDA tensors
    if (!muon_data_positions.is_cuda() ||
        !muon_data_momenta.is_cuda() ||
        !hist_2d_probability_table_iron.is_cuda() ||
        !hist_2d_alias_table_iron.is_cuda() ||
        !hist_2d_bin_centers_first_dim_iron.is_cuda() ||
        !hist_2d_bin_centers_second_dim_iron.is_cuda() ||
        !hist_2d_bin_widths_first_dim_iron.is_cuda() ||
        !hist_2d_bin_widths_second_dim_iron.is_cuda() ||
        !hist_2d_probability_table_concrete.is_cuda() ||
        !hist_2d_alias_table_concrete.is_cuda() ||
        !hist_2d_bin_centers_first_dim_concrete.is_cuda() ||
        !hist_2d_bin_centers_second_dim_concrete.is_cuda() ||
        !hist_2d_bin_widths_first_dim_concrete.is_cuda() ||
        !hist_2d_bin_widths_second_dim_concrete.is_cuda() ||
        !magnetic_field.is_cuda() ||
        !arb8s.is_cuda() ||
        !cavern_params.is_cuda()
        )
    {
        throw std::runtime_error("All tensors except magnetic_field_range must be CUDA tensors.");
    }

    // Call the CUDA implementation for field map
    propagate_muons_with_alias_sampling_cuda(
        muon_data_positions,
        muon_data_momenta,
        charges,
        hist_2d_probability_table_iron,
        hist_2d_alias_table_iron,
        hist_2d_bin_centers_first_dim_iron,
        hist_2d_bin_centers_second_dim_iron,
        hist_2d_bin_widths_first_dim_iron,
        hist_2d_bin_widths_second_dim_iron,
        hist_2d_probability_table_concrete,
        hist_2d_alias_table_concrete,
        hist_2d_bin_centers_first_dim_concrete,
        hist_2d_bin_centers_second_dim_concrete,
        hist_2d_bin_widths_first_dim_concrete,
        hist_2d_bin_widths_second_dim_concrete,
        arb8s,
        hashed3d_arb8s_cells,
        hashed3d_arb8s_indices,
        magnetic_field,
        magnetic_field_range,
        cavern_params,
        use_symmetry,
        sensitive_plane_z,
        kill_at,
        num_steps,
        step_length_fixed,
        seed
    );
}


void add_cuda(torch::Tensor x, torch::Tensor y, torch::Tensor out);

// Define a wrapper function that can be called from Python for addition
void add(torch::Tensor x, torch::Tensor y, torch::Tensor out) {
    // Check that inputs are on the same CUDA device
    if (!x.is_cuda() || !y.is_cuda() || !out.is_cuda()) {
        throw std::runtime_error("All tensors must be CUDA tensors.");
    }
    add_cuda(x, y, out);
}
// Register the functions as PyTorch extensions
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add", &add, "Add two tensors (CUDA)");
    m.def("propagate_muons", &propagate_muons, "Propagate muons with two input tensors (CUDA)");
    m.def("propagate_muons_with_alias_sampling", &propagate_muons_with_alias_sampling, "Propagate muons with magnetic field map (CUDA)");
    m.def("propagate_muons_with_alias_sampling_uniform_field", &propagate_muons_with_alias_sampling_uniform_field, "Propagate muons with uniform magnetic field per ARB8 (CUDA)");
}
