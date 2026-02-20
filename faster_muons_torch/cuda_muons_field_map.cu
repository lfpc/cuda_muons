#include "common.cuh"

// ============================================================
//  Field-map-specific kernels and functions
// ============================================================

__global__ void fill_arb8s_kernel(
    const float* arb8s_tensor, // shape (N,8,3), row-major, contiguous
    ARB8_Data* arb8s_out,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // Each ARB8 has 8 vertices, each with 3 floats (x, y, z)
    const float* verts = arb8s_tensor + i * 8 * 3;

    // Fill vertices
    for (int j = 0; j < 4; ++j) {
        arb8s_out[i].vertices_neg_z[j] = make_float2(verts[j*3 + 0], verts[j*3 + 1]);
        arb8s_out[i].vertices_pos_z[j] = make_float2(verts[(j+4)*3 + 0], verts[(j+4)*3 + 1]);
    }
    // Compute z_center and dz from z values
    float z_neg = verts[0*3 + 2];
    float z_pos = verts[4*3 + 2];
    arb8s_out[i].z_center = 0.5f * (z_neg + z_pos);
    arb8s_out[i].dz = 0.5f * fabsf(z_pos - z_neg);
}


__device__ MaterialType get_material(
    float x, float y, float z, 
    const ARB8_Data* arb8s, 
    const int* cell_starts,
    const int* item_indices,
    const float* cavern_params,
    const ZGridMeta& m
) {
    if (
        (z <= cavern_params[4] && (x <= cavern_params[0] || x >= cavern_params[1] || y <= cavern_params[2] || y >= cavern_params[3])) ||
        (z > cavern_params[4] && (x <= cavern_params[5] || x >= cavern_params[6] || y <= cavern_params[7] || y >= cavern_params[8]))
    ) {
        return MATERIAL_CONCRETE;
    }
    if (z >= m.z_max_global || z < m.z_min_global) {
        return MATERIAL_AIR;
    }
    if (_use_symmetry) {
        x = fabsf(x);
        y = fabsf(y);
    }
    int cell_idx = (int)((z - m.z_min_global) * m.z_cell_height_inv);

    int start_idx = cell_starts[cell_idx];
    int end_idx = cell_starts[cell_idx + 1];
    for (int i = start_idx; i < end_idx; ++i) {
        int arb8_idx = item_indices[i];
        if (is_inside_arb8(x, y, z, arb8s[arb8_idx])) {
            return MATERIAL_IRON;
        }
    }
    return MATERIAL_AIR;
}


__global__ void cuda_propagate_muons_field_map_k(float* muon_data_positions,
                               float* muon_data_momenta,
                               const float* charges,
                               const float* hist_2d_probability_table_iron,
                               const int* hist_2d_alias_table_iron,
                               const float* hist_2d_bin_centers_first_dim_iron,
                               const float* hist_2d_bin_centers_second_dim_iron,
                               const float* hist_2d_bin_widths_first_dim_iron,
                               const float* hist_2d_bin_widths_second_dim_iron,
                               const float* hist_2d_probability_table_concrete,
                               const int* hist_2d_alias_table_concrete,
                               const float* hist_2d_bin_centers_first_dim_concrete,
                               const float* hist_2d_bin_centers_second_dim_concrete,
                               const float* hist_2d_bin_widths_first_dim_concrete,
                               const float* hist_2d_bin_widths_second_dim_concrete,
                               const float* magnetic_field,
                               const float sensitive_plane_z,
                               const float kill_at,
                               const int N,
                               const int H_2d,
                               const ARB8_Data* arb8s,
                               const int* hashed3d_arb8s_cells,
                               const int* hashed3d_arb8s_indices,
                               const FieldMeta field_meta,
                               const ZGridMeta grid_meta,
                               const float* cavern_params,
                               int num_steps,
                               float step_length_fixed,
                               int seed)
                               {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;


    // Initialize random number generator for each thread
    curandState state;
    curand_init(seed, idx, 0, &state);  // Use idx as seed for unique randomness

    int offset = idx * 3;
    float delta_P[3] = {0,0,0};
    float output[3] = {0, 0, 0};

    float muon_data_momenta_this_cached[3] = {muon_data_momenta[offset+0], muon_data_momenta[offset+1], muon_data_momenta[offset+2]};
    float muon_data_positions_this_cached[3] = {muon_data_positions[offset+0], muon_data_positions[offset+1], muon_data_positions[offset+2]};
    float charge_this_cached = charges[idx];
    float step_length = step_length_fixed;

    for (int step = 0; step < num_steps; step++) {
        float mag_P = norm(muon_data_momenta_this_cached);
        if (mag_P < kill_at)
            break;

        if (fabsf(muon_data_positions_this_cached[0]) > 10.0f || fabsf(muon_data_positions_this_cached[1]) > 10.0f) {
            break;
        }
        // Normalize P to get the direction unit vector
        float P_unit[3];
        normalize(muon_data_momenta_this_cached, P_unit);
        int hist_idx = get_first_bin(mag_P);

        // 2. Generate a uniform random number in [0, 1] for sampling from CDF
        float rand_value = curand_uniform(&state);

        MaterialType material = get_material(
            muon_data_positions_this_cached[0],
            muon_data_positions_this_cached[1],
            muon_data_positions_this_cached[2],
            arb8s,
            hashed3d_arb8s_cells, 
            hashed3d_arb8s_indices,
            cavern_params,
            grid_meta
        );

        float delta = 0.0f;
        float delta_second_dim = 0.0f;
        if (material == MATERIAL_IRON) {
            int bin_idx;
            int tbin = curand(&state) % H_2d;

            if (rand_value < hist_2d_probability_table_iron[tbin+hist_idx*H_2d])
                bin_idx = tbin;
            else
                bin_idx =  hist_2d_alias_table_iron[tbin+hist_idx*H_2d];
            // 3. Retrieve the bin value and add it to muon_valu
            float bin_value_first_dim = hist_2d_bin_centers_first_dim_iron[bin_idx]; // Initialize bin_value at the bin center
            float bin_jitter_first_dim = (curand_uniform(&state) - 0.5f) * hist_2d_bin_widths_first_dim_iron[bin_idx]; // Calculate jitter in range [-bin_width/2, bin_width/2]
            bin_value_first_dim += bin_jitter_first_dim; // Apply jitter to the bin value

            float bin_value_second_dim = hist_2d_bin_centers_second_dim_iron[bin_idx]; // Initialize bin_value at the bin center
            float bin_jitter_second_dim = (curand_uniform(&state) - 0.5f) * hist_2d_bin_widths_second_dim_iron[bin_idx]; // Calculate jitter in range [-bin_width/2, bin_width/2]
            bin_value_second_dim += bin_jitter_second_dim; // Apply jitter to the bin value

            delta = mag_P * exp(bin_value_first_dim);
            delta_second_dim = mag_P * exp(bin_value_second_dim);
        }
        else if (material == MATERIAL_CONCRETE) {
            int bin_idx;
            int tbin = curand(&state) % H_2d;

            if (rand_value < hist_2d_probability_table_concrete[tbin+hist_idx*H_2d])
                bin_idx = tbin;
            else
                bin_idx =  hist_2d_alias_table_concrete[tbin+hist_idx*H_2d];
            float bin_value_first_dim = hist_2d_bin_centers_first_dim_concrete[bin_idx]; // Initialize bin_value at the bin center
            float bin_jitter_first_dim = (curand_uniform(&state) - 0.5f) * hist_2d_bin_widths_first_dim_concrete[bin_idx]; // Calculate jitter in range [-bin_width/2, bin_width/2]
            bin_value_first_dim += bin_jitter_first_dim; // Apply jitter to the bin value

            float bin_value_second_dim = hist_2d_bin_centers_second_dim_concrete[bin_idx]; // Initialize bin_value at the bin center
            float bin_jitter_second_dim = (curand_uniform(&state) - 0.5f) * hist_2d_bin_widths_second_dim_concrete[bin_idx]; // Calculate jitter in range [-bin_width/2, bin_width/2]
            
            
            bin_value_second_dim += bin_jitter_second_dim; // Apply jitter to the bin value

            delta = mag_P * expf(bin_value_first_dim);
            delta_second_dim = mag_P * expf(bin_value_second_dim);

        }

        float phi = curand_uniform(&state) * 2 * M_PI;

        // Convert polar coordinates to Cartesian coordinates
        float x = delta_second_dim * __cosf(phi);
        float y = delta_second_dim * __sinf(phi);

        delta_P[0] = x;
        delta_P[1] = -y;
        delta_P[2] = -delta;

        rotateVector(P_unit, delta_P, muon_data_momenta_this_cached, output);
        // works till here

        muon_data_momenta_this_cached[0] = output[0];
        muon_data_momenta_this_cached[1] = output[1];
        muon_data_momenta_this_cached[2] = output[2];
        if (muon_data_momenta_this_cached[2] < 0.0f) {
            break;
        }
        if (muon_data_positions_this_cached[0] >= (cavern_params[5]+BIG_STEP) && (muon_data_positions_this_cached[0] <= (cavern_params[6]-BIG_STEP))
            && (muon_data_positions_this_cached[1] >= (cavern_params[7]+BIG_STEP)) && (muon_data_positions_this_cached[1] <= (cavern_params[8]-BIG_STEP))
            && (muon_data_positions_this_cached[2] >= grid_meta.z_max_global) && (muon_data_positions_this_cached[2] >= field_meta.endZ)
            && (muon_data_positions_this_cached[2] <= sensitive_plane_z - BIG_STEP)) {
            step_length = BIG_STEP;
        } else {step_length = step_length_fixed;}

        rk4_step_cached(muon_data_positions_this_cached, muon_data_momenta_this_cached,
              charge_this_cached, step_length,
              magnetic_field,
              muon_data_positions_this_cached, muon_data_momenta_this_cached,
              field_meta);
        if (sensitive_plane_z >= 0 && (muon_data_positions_this_cached[2] >= sensitive_plane_z)) {
            break;
        }
    }
    muon_data_momenta[offset+0] = muon_data_momenta_this_cached[0];
    muon_data_momenta[offset+1] = muon_data_momenta_this_cached[1];
    muon_data_momenta[offset+2] = muon_data_momenta_this_cached[2];

    muon_data_positions[offset+0] = muon_data_positions_this_cached[0];
    muon_data_positions[offset+1] = muon_data_positions_this_cached[1];
    muon_data_positions[offset+2] = muon_data_positions_this_cached[2];
}


void propagate_muons_with_alias_sampling_cuda(
    torch::Tensor muon_data_positions,
    torch::Tensor muon_data_momenta,
    torch::Tensor charges,
    const std::vector<MaterialHistograms>& material_histograms,
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
    TORCH_CHECK(material_histograms.size() >= 2, "Expected at least 2 materials (iron and concrete)");
    TORCH_CHECK(magnetic_field_range.size(1) == 6, "Expected the second dimension of magnetic_field_range to be 6, but got ", magnetic_field_range.size(1));
    
    // Extract histogram tensors from the struct (index 0 = iron, index 1 = concrete)
    const auto& iron = material_histograms[0];
    const auto& concrete = material_histograms[1];
    
    const auto N = muon_data_positions.size(0);
    const auto H_2D = iron.probability_table.size(1);
    const auto N_momentum_bins = iron.probability_table.size(0);

    cudaMemcpyToSymbol(_use_symmetry, &use_symmetry, sizeof(bool));

    // Define grid and block sizes
    const int threads_per_block = BLOCK_SIZE;
    const int num_blocks = (N + threads_per_block - 1) / threads_per_block;

    float* data_ptr_B = magnetic_field_range.data_ptr<float>();
    FieldMeta magfield_data;
    magfield_data.binsX = magnetic_field.size(0);
    magfield_data.binsY = magnetic_field.size(1);
    magfield_data.binsZ = magnetic_field.size(2);
    magfield_data.startX = data_ptr_B[0];
    magfield_data.endX   = data_ptr_B[1];
    magfield_data.startY = data_ptr_B[2];
    magfield_data.endY   = data_ptr_B[3];
    magfield_data.startZ = data_ptr_B[4];
    magfield_data.endZ   = data_ptr_B[5];
    magfield_data.invX   = 1.0f / (magfield_data.endX - magfield_data.startX);
    magfield_data.invY   = 1.0f / (magfield_data.endY - magfield_data.startY);
    magfield_data.invZ   = 1.0f / (magfield_data.endZ - magfield_data.startZ);
    magfield_data.stride_y  = magfield_data.binsX * magfield_data.binsZ;
    cudaError_t _err = cudaMemcpyToSymbol(d_field_meta, &magfield_data, sizeof(FieldMeta));
    if (_err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyToSymbol(d_field_meta) failed: %s\n", cudaGetErrorString(_err));
    }

    float log_start_val = log10f(0.18f);
    float log_stop_val = log10f(400.0f);
    float inv_log_step_val = N_momentum_bins / (log_stop_val - log_start_val);
    cudaMemcpyToSymbol(LOG_START, &log_start_val, sizeof(float));
    cudaMemcpyToSymbol(LOG_STOP, &log_stop_val, sizeof(float));
    cudaMemcpyToSymbol(INV_LOG_STEP, &inv_log_step_val, sizeof(float));

    auto z_neg_vals = arb8s.select(1, 0).select(1, 2); 
    auto z_pos_vals = arb8s.select(1, 4).select(1, 2); 
    float z_min = std::min(z_neg_vals.min().item<float>(), z_pos_vals.min().item<float>());
    float z_max = std::max(z_neg_vals.max().item<float>(), z_pos_vals.max().item<float>());
    z_max = std::max(z_max, 30.0f);


    int N_arbs = arb8s.size(0);
    ARB8_Data* arb8s_device;
    cudaMalloc(&arb8s_device, N_arbs * sizeof(ARB8_Data));

    const int threads = 256;
    const int blocks = (N_arbs + threads - 1) / threads;
    fill_arb8s_kernel<<<blocks, threads>>>(
        arb8s.data_ptr<float>(),
        arb8s_device,
        N_arbs
    );
    
    const int sz = hashed3d_arb8s_cells.size(0) - 1;
    ZGridMeta grid_data;
    grid_data.sz = sz;
    grid_data.z_min_global = z_min;
    grid_data.z_max_global = z_max;
    grid_data.z_cell_height_inv = (float)sz / (z_max - z_min);
    cudaMemcpyToSymbol(d_grid_meta, &grid_data, sizeof(ZGridMeta));



    cudaDeviceSynchronize();
    

    #define CUDA_CHECK(call)                                          \
    do {                                                          \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",           \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

    cuda_propagate_muons_field_map_k<<<num_blocks, threads_per_block>>>(
        muon_data_positions.data_ptr<float>(),
        muon_data_momenta.data_ptr<float>(),
        charges.data_ptr<float>(),
        iron.probability_table.data_ptr<float>(),
        iron.alias_table.data_ptr<int>(),
        iron.bin_centers_first_dim.data_ptr<float>(),
        iron.bin_centers_second_dim.data_ptr<float>(),
        iron.bin_widths_first_dim.data_ptr<float>(),
        iron.bin_widths_second_dim.data_ptr<float>(),
        concrete.probability_table.data_ptr<float>(),
        concrete.alias_table.data_ptr<int>(),
        concrete.bin_centers_first_dim.data_ptr<float>(),
        concrete.bin_centers_second_dim.data_ptr<float>(),
        concrete.bin_widths_first_dim.data_ptr<float>(),
        concrete.bin_widths_second_dim.data_ptr<float>(),
        magnetic_field.data_ptr<float>(),
        sensitive_plane_z,
        kill_at,
        N,
        H_2D,
        arb8s_device,
        hashed3d_arb8s_cells.data_ptr<int>(),
        hashed3d_arb8s_indices.data_ptr<int>(),
        magfield_data,
        grid_data,
        cavern_params.data_ptr<float>(),
        num_steps,
        step_length_fixed,
        seed
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(arb8s_device);
}
