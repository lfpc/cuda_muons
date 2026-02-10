#include "common.cuh"

// ============================================================
//  Shared kernels and host functions (defined once to avoid
//  multiple-definition linker errors)
// ============================================================

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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

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
