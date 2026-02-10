#pragma once

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>
#include <vector_types.h>
#include "material_histograms.h"

#define BLOCK_SIZE 256

// ============================================================
//  Structs
// ============================================================

struct ARB8_Data {
    float2 vertices_neg_z[4];
    float2 vertices_pos_z[4];
    float z_center;
    float dz;
    float3 B_field;  // Uniform magnetic field for this block (only used in uniform-field mode)
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

struct ZGridMeta {
    float z_min_global;
    float z_max_global;
    float z_cell_height_inv;
    int sz;
};

// ============================================================
//  Enums
// ============================================================

enum MaterialType { MATERIAL_IRON, MATERIAL_CONCRETE, MATERIAL_AIR };

// ============================================================
//  Device constant memory  (static: each translation unit
//  gets its own copy – set via cudaMemcpyToSymbol from the
//  host launcher that lives in the same TU)
// ============================================================

// Physics constants (compile-time, never changed at runtime)
static __constant__ float MUON_MASS          = 0.1056583755f;  // GeV/c²
static __constant__ float c_speed_of_light   = 299792458.0f;   // m/s
static __constant__ float e_charge           = 1.602176634e-19f;
static __constant__ float GeV_over_c_to_SI   = 5.3443e-19f;    // kg·m/s

// Simulation constants
static __constant__ float BIG_STEP = 0.2f;

// Runtime-set constants (set via cudaMemcpyToSymbol in host launchers)
static __constant__ float LOG_START;
static __constant__ float LOG_STOP;
static __constant__ float INV_LOG_STEP;
static __device__ __constant__ bool _use_symmetry = true;
static __constant__ FieldMeta d_field_meta;
static __constant__ ZGridMeta d_grid_meta;

// ============================================================
//  Shared device helper functions
// ============================================================

static __device__ __forceinline__ float dotProduct(const float a[3], const float b[3]) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

static __device__ __forceinline__ void crossProduct(const float a[3], const float b[3], float result[3]) {
    result[0] = a[1] * b[2] - a[2] * b[1];
    result[1] = a[2] * b[0] - a[0] * b[2];
    result[2] = a[0] * b[1] - a[1] * b[0];
}

static __device__ __forceinline__ float norm(const float v[3]) {
    return sqrtf(dotProduct(v, v));
}

static __device__ __forceinline__ void normalize(const float v[3], float result[3]) {
    float inv_norm = 1.0f / norm(v);
    result[0] = v[0] * inv_norm;
    result[1] = v[1] * inv_norm;
    result[2] = v[2] * inv_norm;
}

static __device__ __forceinline__ void rotateVector(const float P_unit[3], const float delta_P[3], const float P[3], float P_new[3]) {
    float z_axis[3] = {0, 0, 1};

    float rotation_axis[3];
    crossProduct(z_axis, P_unit, rotation_axis);
    float rotation_axis_norm = norm(rotation_axis);

     if (rotation_axis_norm == 0) {
         P_new[0] = P[0] + delta_P[0];
         P_new[1] = P[1] + delta_P[1];
         P_new[2] = P[2] + delta_P[2];
         return;
     }

    normalize(rotation_axis, rotation_axis);

    float cos_theta = dotProduct(z_axis, P_unit);
    float theta = acosf(cos_theta);

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

    float delta_P_rotated[3] = {
            R[0][0] * delta_P[0] + R[0][1] * delta_P[1] + R[0][2] * delta_P[2],
            R[1][0] * delta_P[0] + R[1][1] * delta_P[1] + R[1][2] * delta_P[2],
            R[2][0] * delta_P[0] + R[2][1] * delta_P[1] + R[2][2] * delta_P[2]
    };

    P_new[0] = P[0] + delta_P_rotated[0];
    P_new[1] = P[1] + delta_P_rotated[1];
    P_new[2] = P[2] + delta_P_rotated[2];
}

static __device__ __forceinline__ int get_first_bin(float num) {
    num = fmaxf(0.18f, num);
    num = fminf(400.0f, num);
    int index = static_cast<int>((log10f(num) - LOG_START) * INV_LOG_STEP);
    return index;
}

static __device__ __forceinline__ float3 getFieldAt(const float *field,
                                    const float x, const float y, const float z,
                                    const FieldMeta& m)
{
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

    if (sx < m.startX || sx > m.endX || sy < m.startY || sy > m.endY || sz < m.startZ || sz > m.endZ) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    const float normX = (sx - m.startX) * m.invX;
    const float normY = (sy - m.startY) * m.invY;
    const float normZ = (sz - m.startZ) * m.invZ;

    int i = __float2int_rn(normX * (m.binsX - 1)); 
    int j = __float2int_rn(normY * (m.binsY - 1));
    int k = __float2int_rn(normZ * (m.binsZ - 1));

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

static __device__ __forceinline__ bool is_inside_arb8(const float x, const float y, const float z, const ARB8_Data& block) {
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

// ============================================================
//  Derivative & RK4 functions
// ============================================================

static __device__ __forceinline__ void derivative(float* state, float charge, const float* B, float* deriv,
              const float p_mag, const float energy, const FieldMeta& field_meta) {
    float x  = state[0];
    float y  = state[1];
    float z  = state[2];
    float px = state[3];
    float py = state[4];
    float pz = state[5];

    float vx = (px / energy) * c_speed_of_light;
    float vy = (py / energy) * c_speed_of_light;
    float vz = (pz / energy) * c_speed_of_light;

    float q = charge * e_charge;

    float3 B_ = getFieldAt(B, x, y, z, field_meta);

    float dpx = q * (vy * B_.z - vz * B_.y) / GeV_over_c_to_SI;
    float dpy = q * (vz * B_.x - vx * B_.z) / GeV_over_c_to_SI;
    float dpz = q * (vx * B_.y - vy * B_.x) / GeV_over_c_to_SI;

    deriv[0] = vx;
    deriv[1] = vy;
    deriv[2] = vz;
    deriv[3] = dpx;
    deriv[4] = dpy;
    deriv[5] = dpz;
}

static __device__ __forceinline__ void derivative_cached(float* state, float charge, float3 B_vec, float* deriv,
              const float p_mag, const float energy) {
    float px = state[3];
    float py = state[4];
    float pz = state[5];

    float vx = (px / energy) * c_speed_of_light;
    float vy = (py / energy) * c_speed_of_light;
    float vz = (pz / energy) * c_speed_of_light;

    float q = charge * e_charge;

    float dpx = q * (vy * B_vec.z - vz * B_vec.y) / GeV_over_c_to_SI;
    float dpy = q * (vz * B_vec.x - vx * B_vec.z) / GeV_over_c_to_SI;
    float dpz = q * (vx * B_vec.y - vy * B_vec.x) / GeV_over_c_to_SI;

    deriv[0] = vx;
    deriv[1] = vy;
    deriv[2] = vz;
    deriv[3] = dpx;
    deriv[4] = dpy;
    deriv[5] = dpz;
}

static __device__ __forceinline__ void rk4_step(float pos[3], float mom[3],
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

    derivative(state, charge, B, k1, p_mag, energy, field_meta);

    for (int i = 0; i < 6; i++) temp[i] = state[i] + 0.5 * dt * k1[i];
    derivative(temp, charge, B, k2, p_mag, energy, field_meta);

    for (int i = 0; i < 6; i++) temp[i] = state[i] + 0.5 * dt * k2[i];
    derivative(temp, charge, B, k3, p_mag, energy, field_meta);

    for (int i = 0; i < 6; i++) temp[i] = state[i] + dt * k3[i];
    derivative(temp, charge, B, k4, p_mag, energy, field_meta);

    float new_state[6];
    for (int i = 0; i < 6; i++) {
        new_state[i] = state[i] + dt / 6.0 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
    }

    new_pos[0] = new_state[0];
    new_pos[1] = new_state[1];
    new_pos[2] = new_state[2];
    new_mom[0] = new_state[3];
    new_mom[1] = new_state[4];
    new_mom[2] = new_state[5];
}

static __device__ __forceinline__ void rk4_step_cached(float pos[3], float mom[3],
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
        state[i] += dt / 6.0f * (k1[i] + 2.0f * k2[i] + 2.0f * k3[i] + k4[i]);
    }

    new_pos[0] = state[0]; new_pos[1] = state[1]; new_pos[2] = state[2];
    new_mom[0] = state[3]; new_mom[1] = state[4]; new_mom[2] = state[5];
}

// ============================================================
//  Host helper
// ============================================================

static inline void load_arb8s_from_tensor(const at::Tensor& arb8s_tensor, std::vector<ARB8_Data>& out) {
    int N = arb8s_tensor.size(0);
    auto arb8s_acc = arb8s_tensor.accessor<float, 3>();
    out.resize(N);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < 4; ++j) {
            out[i].vertices_neg_z[j] = make_float2(arb8s_acc[i][j][0], arb8s_acc[i][j][1]);
            out[i].vertices_pos_z[j] = make_float2(arb8s_acc[i][j+4][0], arb8s_acc[i][j+4][1]);
        }
        float z_neg = arb8s_acc[i][0][2];
        float z_pos = arb8s_acc[i][4][2];
        out[i].z_center = 0.5f * (z_neg + z_pos);
        out[i].dz = 0.5f * fabs(z_pos - z_neg);
    }
}
