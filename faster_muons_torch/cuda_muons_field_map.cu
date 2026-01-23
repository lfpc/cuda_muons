#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>
#include <vector_types.h>


__global__ void add_kernel(float *x, float *y, float *out, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        out[index] = x[index] + y[index];
    }
}

void add_cuda(torch::Tensor x, torch::Tensor y, torch::Tensor out) {
    int size = x.numel();
    int threads = 1024;
    int blocks = (size + threads - 1) / threads;

    add_kernel<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), out.data_ptr<float>(), size);
}


#define BLOCK_SIZE 256

__global__ void propagate_muons_kernel(
    float* muon_data,        // Input tensor
    float* muon_data_output, // Output tensor
    float* loss_hists,       // CDF histogram data
    float* bin_widths,       // Width of bins in histograms
    float* bin_centers,      // Center of bins in histograms
    int N,                   // Number of muons
    int M,                   // Number of histograms
    int H,                   // Number of bins in each histogram
    int num_steps            // Number of propagation steps
) {
//     extern __shared__ float loss_hists_shared[];
//
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
//
//     // Copy global data to shared memory - usually only a subset of threads are used
//     if (idx < M*H) {
//         sharedData[idx] = globalData[idx];
//     }
//
//     // Synchronize to ensure shared memory is fully populated
//     __syncthreads();


    if (idx >= N) return;

    // Initialize random number generator for each thread
    curandState state;
    curand_init(1234, idx, 0, &state);  // Use idx as seed for unique randomness

    // Initialize output value for this muon
    float muon_value = muon_data[idx];

    // Loop over propagation steps
    for (int step = 0; step < num_steps; ++step) {
        // 1. Select a random histogram
        int hist_idx = curand(&state) % M;

        // 2. Generate a uniform random number in [0, 1] for sampling from CDF
        float rand_value = curand_uniform(&state);

        // Linear version
//         int start = hist_idx * H;
//         int end = start + H - 1;
//         int bin_idx = start;
//
//         while (bin_idx <= end) {
//             if (loss_hists[bin_idx] >= rand_value) {
//                 break; // Found the element
//             }
//             bin_idx++;
//         }


        // Binary search version

        // 3. Perform binary search in the selected histogram's CDF
        int start = hist_idx * H;
        int end = start + H - 1;
        int left = start;
        int right = end;
        while (left < right) {
            int mid = (left + right) / 2;
            if (loss_hists[mid] < rand_value) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        // 4. Retrieve the bin value and add it to muon_value
        int bin_idx = left;

        float bin_value = bin_centers[bin_idx]; // Initialize bin_value at the bin center
        float bin_jitter = (curand_uniform(&state) - 0.5f) * bin_widths[bin_idx]; // Calculate jitter in range [-bin_width/2, bin_width/2]
        bin_value += bin_jitter; // Apply jitter to the bin value

        muon_value += bin_value;
    }

    // Store the final result after all steps
    muon_data_output[idx] = muon_value;
}


torch::Tensor propagate_muons_cuda(
    torch::Tensor muon_data,
    torch::Tensor loss_hists,
    torch::Tensor bin_widths,
    torch::Tensor bin_centers,
    int num_steps  // Number of propagation steps
) {
    const auto N = muon_data.size(0);
    const auto M = loss_hists.size(0);
    const auto H = loss_hists.size(1);

    // Allocate output tensor
    auto muon_data_output = torch::empty_like(muon_data);

    // Define grid and block sizes
    const int threads_per_block = BLOCK_SIZE;
    const int num_blocks = (N + threads_per_block - 1) / threads_per_block;

    // Launch the kernel
    propagate_muons_kernel<<<num_blocks, threads_per_block>>>(
        muon_data.data_ptr<float>(),
        muon_data_output.data_ptr<float>(),
        loss_hists.data_ptr<float>(),
        bin_widths.data_ptr<float>(),
        bin_centers.data_ptr<float>(),
        N, M, H, num_steps
    );

    // Synchronize to ensure kernel completion
    cudaDeviceSynchronize();
    return muon_data_output;
}

struct ARB8_Data {
    float2 vertices_neg_z[4];
    float2 vertices_pos_z[4];
    float z_center;
    float dz;
};

struct FieldMeta {
    int binsX;
    int binsY;
    int binsZ;
    float startX;
    float endX;
    float startY;
    float endY;
    float startZ;
    float endZ;
    float invX;
    float invY;
    float invZ;
    int stride_y;   // binsX * binsZ
};
struct ZGridMeta { float z_min_global; float z_max_global; float z_cell_height_inv; int sz; };

__constant__ FieldMeta d_field_meta;
__constant__ ZGridMeta d_grid_meta;
__constant__ float BIG_STEP = 0.2f;


void load_arb8s_from_tensor(const at::Tensor& arb8s_tensor, std::vector<ARB8_Data>& out) {
    int N = arb8s_tensor.size(0);
    auto arb8s_acc = arb8s_tensor.accessor<float, 3>();
    out.resize(N);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < 4; ++j) {
            out[i].vertices_neg_z[j] = make_float2(arb8s_acc[i][j][0], arb8s_acc[i][j][1]);
            out[i].vertices_pos_z[j] = make_float2(arb8s_acc[i][j+4][0], arb8s_acc[i][j+4][1]);
        }
        // Compute z_center and dz from z values
        float z_neg = arb8s_acc[i][0][2];
        float z_pos = arb8s_acc[i][4][2];
        out[i].z_center = 0.5f * (z_neg + z_pos);
        out[i].dz = 0.5f * fabs(z_pos - z_neg);
    }
}
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



__constant__ float LOG_START;
__constant__ float LOG_STOP;
__constant__ float INV_LOG_STEP;
__device__ __constant__ bool _use_symmetry = true;

__device__ int get_first_bin(float num) {
    // Clip num to be within [10, 300]

    num = fmaxf(0.18f, num);
    num = fminf(400.0f, num);
    // Calculate the range index
    //int index = num / 5;
    int index = static_cast<int>((log10f(num) - LOG_START) * INV_LOG_STEP);
    return index;
}


// Helper function to compute the dot product of two vectors
__device__ float dotProduct(const float a[3], const float b[3]) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

// Helper function to compute the cross product of two vectors
__device__ void crossProduct(const float a[3], const float b[3], float result[3]) {
    result[0] = a[1] * b[2] - a[2] * b[1];
    result[1] = a[2] * b[0] - a[0] * b[2];
    result[2] = a[0] * b[1] - a[1] * b[0];
}

// Helper function to compute the norm of a vector
__device__ float norm(const float v[3]) {
    return sqrtf(dotProduct(v, v));
}

// Helper function to normalize a vector
__device__ void normalize(const float v[3], float result[3]) {
    float inv_norm = 1.0f/norm(v); //rsqrtf(dotProduct(v, v));  // 1/sqrt(x)
    result[0] = v[0] * inv_norm;
    result[1] = v[1] * inv_norm;
    result[2] = v[2] * inv_norm;
}


// Function to rotate a vector delta_P to align with the direction of P
__device__ void rotateVector(const float P_unit[3], const float delta_P[3], const float P[3], float P_new[3]) {
    // Define the z-axis unit vector
    float z_axis[3] = {0, 0, 1};

    // Calculate the rotation axis (cross product of z-axis and P_unit)
    float rotation_axis[3];
    crossProduct(z_axis, P_unit, rotation_axis);
    float rotation_axis_norm = norm(rotation_axis);

     // Check if rotation is needed
     if (rotation_axis_norm == 0) {
         // P is aligned with z-axis; no rotation is needed
         P_new[0] = P[0] + delta_P[0];
         P_new[1] = P[1] + delta_P[1];
         P_new[2] = P[2] + delta_P[2];
         return;
     }

    // Normalize the rotation axis
    normalize(rotation_axis, rotation_axis);

    // Calculate the rotation angle
    float cos_theta = dotProduct(z_axis, P_unit);
    float theta = acosf(cos_theta);

    // Construct the rotation matrix using Rodrigues' rotation formula
    float K[3][3] = {
            {0, -rotation_axis[2], rotation_axis[1]},
            {rotation_axis[2], 0, -rotation_axis[0]},
            {-rotation_axis[1], rotation_axis[0], 0}
    };

    float R[3][3];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            R[i][j] = (i == j ? 1 : 0) +
                      sinf(theta) * K[i][j] +
                      (1 - cosf(theta)) * (K[i][0] * K[0][j] + K[i][1] * K[1][j] + K[i][2] * K[2][j]);
        }
    }

    // Rotate delta_P using the rotation matrix
    float delta_P_rotated[3] = {
            R[0][0] * delta_P[0] + R[0][1] * delta_P[1] + R[0][2] * delta_P[2],
            R[1][0] * delta_P[0] + R[1][1] * delta_P[1] + R[1][2] * delta_P[2],
            R[2][0] * delta_P[0] + R[2][1] * delta_P[1] + R[2][2] * delta_P[2]
    };


    // Update P by adding the rotated delta_P to get the new position
    P_new[0] = P[0] + delta_P_rotated[0];
    P_new[1] = P[1] + delta_P_rotated[1];
    P_new[2] = P[2] + delta_P_rotated[2];
}

// For device and host compatibility.
__device__ inline float3 getFieldAt(const float *field,
                                    const float x, const float y, const float z,
                                    const FieldMeta& m)
{
    //const FieldMeta &m = d_field_meta;

    // Map to first quadrant (use symmetry of stored field)
    float sx, sy, sz;
    if (_use_symmetry) {
        sx = fabsf(x);
        sy = fabsf(y);
        sz = z;
        
    } else {
        sx = x;
        sy = y;
        sz = z;
    }

    // Quick bounds test in mapped coordinates
    if (sx < m.startX || sx > m.endX || sy < m.startY || sy > m.endY || sz < m.startZ || sz > m.endZ) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    // Normalize to [0,1] using precomputed reciprocals
    const float normX = (sx - m.startX) * m.invX;
    const float normY = (sy - m.startY) * m.invY;
    const float normZ = (sz - m.startZ) * m.invZ;

    // Map the normalized coordinate to a bin index.
    int i = __float2int_rn(normX * (m.binsX - 1)); 
    int j = __float2int_rn(normY * (m.binsY - 1));
    int k = __float2int_rn(normZ * (m.binsZ - 1));


    // Compute the flat array index.
    // Each field point consists of 3 floats.
    int index = (j * m.stride_y + i * m.binsZ + k) * 3;

    float sign_x = 1.0f, sign_z = 1.0f;
    if (_use_symmetry) {
        sign_x = ((x >= 0.0f && y >= 0.0f) || (x <= 0.0f && y <= 0.0f)) ? 1.0f : -1.0f;
        sign_z = ((x >= 0.0f && y >= 0.0f) || (x <= 0.0f && y >= 0.0f)) ? 1.0f : -1.0f;
    } 
    
    float3 fieldVal;
    fieldVal.x = field[index] * sign_x;
    fieldVal.y = field[index + 1];
    fieldVal.z = field[index + 2] * sign_z;

    return fieldVal;
}

enum MaterialType { MATERIAL_IRON, MATERIAL_CONCRETE, MATERIAL_AIR };

__device__ bool is_inside_arb8(const float x, const float y, const float z, const ARB8_Data& block) {
    if (fabsf(z - block.z_center) > block.dz) {
        return false;
    }
    float f = (z - (block.z_center - block.dz)) / (2.0f * block.dz);
    float2 interpolated_verts[4];
    for (int i = 0; i < 4; ++i) {
        interpolated_verts[i].x = (1.0f - f) * block.vertices_neg_z[i].x + f * block.vertices_pos_z[i].x;
        interpolated_verts[i].y = (1.0f - f) * block.vertices_neg_z[i].y + f * block.vertices_pos_z[i].y;
    }
    float signs[4];
    float2 edge, p_vec;
    for (int i = 0; i < 4; ++i) {
        int j = (i + 1) % 4;
        edge.x = interpolated_verts[j].x - interpolated_verts[i].x;
        edge.y = interpolated_verts[j].y - interpolated_verts[i].y;
        p_vec.x = x - interpolated_verts[i].x;
        p_vec.y = y - interpolated_verts[i].y;
        signs[i] = edge.x * p_vec.y - edge.y * p_vec.x;
    }
    if ((signs[0] >= 0 && signs[1] >= 0 && signs[2] >= 0 && signs[3] >= 0) ||
        (signs[0] <= 0 && signs[1] <= 0 && signs[2] <= 0 && signs[3] <= 0)) {
        return true;
    }
    return false;
}
__device__ MaterialType get_material(
    float x, float y, float z, 
    const ARB8_Data* arb8s, 
    const int* cell_starts,      // Add these parameters
    const int* item_indices,     // Add these parameters
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

__constant__ float MUON_MASS = 0.1056583755f;  // GeV/c²
__constant__ float c_speed_of_light = 299792458.0f;  // Speed of light in m/s
__constant__ float e_charge = 1.602176634e-19f;  // Elementary charge in Coulombs
__constant__ float GeV_over_c_to_SI  = 5.3443e-19f;  // ≈ 5.3443e-19 kg·m/s

__device__ void derivative(float* state, float charge, const float* B, float* deriv,
              const float p_mag, const float energy, const FieldMeta& field_meta) {
    // Unpack the state array
    float x  = state[0];
    float y  = state[1];
    float z  = state[2];
    float px = state[3];
    float py = state[4];
    float pz = state[5];


    float vx = (px / energy) * c_speed_of_light;
    float vy = (py / energy) * c_speed_of_light;
    float vz = (pz / energy) * c_speed_of_light;

    // Multiply the charge by the elementary charge (e_charge)
    float q = charge * e_charge;

    float3 B_ = getFieldAt(B, x, y, z, field_meta);

    // Lorentz force: dp/dt = q * (v x B) scaled by the conversion factor.
    float dpx = q * (vy * B_.z - vz * B_.y) / GeV_over_c_to_SI;
    float dpy = q * (vz * B_.x - vx * B_.z) / GeV_over_c_to_SI;
    float dpz = q * (vx * B_.y - vy * B_.x) / GeV_over_c_to_SI;

    // Pack the derivatives into the output array
    deriv[0] = vx;
    deriv[1] = vy;
    deriv[2] = vz;
    deriv[3] = dpx;
    deriv[4] = dpy;
    deriv[5] = dpz;

}

__device__ void rk4_step(float pos[3], float mom[3],
              float charge, float step_length_fixed,
              const float* B,
              float new_pos[3], float new_mom[3],
              const FieldMeta& field_meta)
{
    // Combine position and momentum into a single state vector.
    float state[6] = { pos[0], pos[1], pos[2], mom[0], mom[1], mom[2] };

    // Compute the magnitude of the momentum.
    float p_mag = sqrtf(mom[0]*mom[0] + mom[1]*mom[1] + mom[2]*mom[2]);

//     float p_mag  = sqrt(px * px + py * py + pz * pz);
    float energy = sqrtf(p_mag * p_mag + MUON_MASS * MUON_MASS);

    // Calculate the energy: E^2 = p^2 + m^2
//     float energy = sqrt(p_mag*p_mag + MUON_MASS*MUON_MASS);

    // Determine the velocity magnitude: v = (p/E) * c (for ultra-relativistic particles, v ~ c)
    float v_mag = (p_mag / energy) * c_speed_of_light;

    // Calculate time step dt = step_length / v
    float dt = step_length_fixed / v_mag;

    // Allocate arrays for RK4 slopes.
    float k1[6], k2[6],
          k3[6], k4[6];
    float temp[6];

//     // First RK4 step: k1
    derivative(state, charge, B, k1,
              p_mag, energy, field_meta);

    // Second RK4 step: k2
    for (int i = 0; i < 6; i++) {
        temp[i] = state[i] + 0.5 * dt * k1[i];
    }

    derivative(temp, charge, B, k2,
              p_mag, energy, field_meta);


    // Third RK4 step: k3
    for (int i = 0; i < 6; i++) {
        temp[i] = state[i] + 0.5 * dt * k2[i];
    }
    derivative(temp, charge, B, k3,
              p_mag, energy, field_meta);

    // Fourth RK4 step: k4
    for (int i = 0; i < 6; i++) {
        temp[i] = state[i] + dt * k3[i];
    }

    derivative(temp, charge, B, k4,
              p_mag, energy, field_meta);



    // Combine the intermediate slopes to compute the new state.
    float new_state[6];
    for (int i = 0; i < 6; i++) {
        new_state[i] = state[i]+ dt/6.0 * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]);
    }

    // Write the updated state into the output arrays.
    // The first three elements are position, and the last three are momentum.
    new_pos[0] = new_state[0];
    new_pos[1] = new_state[1];
    new_pos[2] = new_state[2];

    new_mom[0] = new_state[3];
    new_mom[1] = new_state[4];
    new_mom[2] = new_state[5];
}

__device__ void derivative_cached(float* state, float charge, float3 B_vec, float* deriv,
              const float p_mag, const float energy) {
    // Unpack the state array
    float px = state[3];
    float py = state[4];
    float pz = state[5];

    float vx = (px / energy) * c_speed_of_light;
    float vy = (py / energy) * c_speed_of_light;
    float vz = (pz / energy) * c_speed_of_light;

    // Multiply the charge by the elementary charge (e_charge)
    float q = charge * e_charge;

    // Lorentz force: dp/dt = q * (v x B) scaled by the conversion factor.
    float dpx = q * (vy * B_vec.z - vz * B_vec.y) / GeV_over_c_to_SI;
    float dpy = q * (vz * B_vec.x - vx * B_vec.z) / GeV_over_c_to_SI;
    float dpz = q * (vx * B_vec.y - vy * B_vec.x) / GeV_over_c_to_SI;

    // Pack the derivatives into the output array
    deriv[0] = vx;
    deriv[1] = vy;
    deriv[2] = vz;
    deriv[3] = dpx;
    deriv[4] = dpy;
    deriv[5] = dpz;
}

__device__ void rk4_step_cached(float pos[3], float mom[3],
              float charge, float step_length_fixed,
              const float* B,
              float new_pos[3], float new_mom[3],
              const FieldMeta& field_meta)
{
    float state[6] = { pos[0], pos[1], pos[2], mom[0], mom[1], mom[2] };
    float p_mag = sqrtf(mom[0]*mom[0] + mom[1]*mom[1] + mom[2]*mom[2]);
    float energy = sqrtf(p_mag * p_mag + MUON_MASS * MUON_MASS);
    float v_mag = (p_mag / energy) * c_speed_of_light;
    float dt = step_length_fixed / v_mag;

    float k1[6], k2[6], k3[6], k4[6];
    float temp[6];

    // Field at START of step
    float3 B_start = getFieldAt(B, pos[0], pos[1], pos[2], field_meta);
    derivative_cached(state, charge, B_start, k1, p_mag, energy);

    // Estimate endpoint position
    float end_pos[3];
    for (int i = 0; i < 3; i++) {
        end_pos[i] = pos[i] + dt * k1[i];  // Rough estimate using k1
    }
    
    // Field at END of step
    float3 B_end = getFieldAt(B, end_pos[0], end_pos[1], end_pos[2], field_meta);
    
    // Average field for intermediate calculations
    float3 B_avg;
    B_avg.x = 0.5f * (B_start.x + B_end.x);
    B_avg.y = 0.5f * (B_start.y + B_end.y);
    B_avg.z = 0.5f * (B_start.z + B_end.z);

    // k2, k3, k4 use averaged field
    for (int i = 0; i < 6; i++) temp[i] = state[i] + 0.5f * dt * k1[i];
    derivative_cached(temp, charge, B_avg, k2, p_mag, energy);

    for (int i = 0; i < 6; i++) temp[i] = state[i] + 0.5f * dt * k2[i];
    derivative_cached(temp, charge, B_avg, k3, p_mag, energy);

    for (int i = 0; i < 6; i++) temp[i] = state[i] + dt * k3[i];
    derivative_cached(temp, charge, B_avg, k4, p_mag, energy);

    // Combine
    for (int i = 0; i < 6; i++) {
        state[i] += dt/6.0f * (k1[i] + 2.0f*k2[i] + 2.0f*k3[i] + k4[i]);
    }

    new_pos[0] = state[0]; new_pos[1] = state[1]; new_pos[2] = state[2];
    new_mom[0] = state[3]; new_mom[1] = state[4]; new_mom[2] = state[5];
}




__global__ void cuda_test_propagate_muons_k(float* muon_data_positions,
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

//     float *muon_data_momenta_this = muon_data_momenta + offset;
//     float *muon_data_positions_this = muon_data_positions + offset;

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
    TORCH_CHECK(magnetic_field_range.size(1) == 6, "Expected the second dimension of magnetic_field_range to be 6, but got ", magnetic_field_range.size(1));
    const auto N = muon_data_positions.size(0);
    const auto H_2D = hist_2d_probability_table_iron.size(1);

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
    float inv_log_step_val = 95 / (log_stop_val - log_start_val);
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

    cuda_test_propagate_muons_k<<<num_blocks, threads_per_block>>>(
        muon_data_positions.data_ptr<float>(),
        muon_data_momenta.data_ptr<float>(),
        charges.data_ptr<float>(),
        hist_2d_probability_table_iron.data_ptr<float>(),
        hist_2d_alias_table_iron.data_ptr<int>(),
        hist_2d_bin_centers_first_dim_iron.data_ptr<float>(),
        hist_2d_bin_centers_second_dim_iron.data_ptr<float>(),
        hist_2d_bin_widths_first_dim_iron.data_ptr<float>(),
        hist_2d_bin_widths_second_dim_iron.data_ptr<float>(),
        hist_2d_probability_table_concrete.data_ptr<float>(),
        hist_2d_alias_table_concrete.data_ptr<int>(),
        hist_2d_bin_centers_first_dim_concrete.data_ptr<float>(),
        hist_2d_bin_centers_second_dim_concrete.data_ptr<float>(),
        hist_2d_bin_widths_first_dim_concrete.data_ptr<float>(),
        hist_2d_bin_widths_second_dim_concrete.data_ptr<float>(),
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