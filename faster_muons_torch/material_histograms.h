#ifndef MATERIAL_HISTOGRAMS_H
#define MATERIAL_HISTOGRAMS_H

#include <torch/extension.h>
#include <vector>

// Struct to hold all histogram data for a single material
struct MaterialHistograms {
    torch::Tensor probability_table;
    torch::Tensor alias_table;
    torch::Tensor bin_centers_first_dim;
    torch::Tensor bin_centers_second_dim;
    torch::Tensor bin_widths_first_dim;
    torch::Tensor bin_widths_second_dim;
};

#endif // MATERIAL_HISTOGRAMS_H
