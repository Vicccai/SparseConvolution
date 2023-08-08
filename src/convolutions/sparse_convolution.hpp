#include "../data_handling/genotype_data.hpp"
#include <torch/torch.h>
#include <tuple>
#include <vector>

using std::vector;

/**
 * @brief handles sparse convolution with batches and channels
 *
 * @param input (minibatch, in_channels, GenotypeData)
 * @param weight (out_channels, in_channels, width, height)
 * @param bias optional bias
 * @param stride
 * @param dilation
 * @return torch::Tensor
 */
torch::Tensor sparse_convolution(
    const vector<vector<GenotypeData>> &input,
    const vector<vector<vector<vector<double>>>> &weight,
    const double &bias = 0,
    const std::tuple<int, int> &stride = std::make_tuple(1, 1),
    const std::tuple<int, int> &dilation = std::make_tuple(1, 1));

torch::Tensor
sparse_convolution(const vector<vector<GenotypeData>> &input,
                   const vector<vector<vector<vector<double>>>> &weight,
                   const double &bias = 0,
                   const std::tuple<int, int> &stride = std::make_tuple(1, 1),
                   const int &dilation = 1);

torch::Tensor sparse_convolution(
    const vector<vector<GenotypeData>> &input,
    const vector<vector<vector<vector<double>>>> &weight,
    const double &bias = 0, const int &stride = 1,
    const std::tuple<int, int> &dilation = std::make_tuple(1, 1));

torch::Tensor
sparse_convolution(const vector<vector<GenotypeData>> &input,
                   const vector<vector<vector<vector<double>>>> &weight,
                   const double &bias = 0, const int &stride = 1,
                   const int &dilation = 1);

/**
 * @brief Performs single convolution operation for one 2d input and one 2d
 * weight
 *
 * @param homo_snps
 * @param hetero_snps
 * @param num_snps
 * @param num_individuals
 * @param weight
 * @param stride
 * @param dilation
 * @param bias
 * @param output_size
 * @return torch::Tensor
 */
torch::Tensor sparse_convolution_input_based_optimized(
    const GenotypeData &input, const vector<vector<double>> &weight,
    const int &stride_row, const int &stride_col, const int &dilation_row,
    const int &dilation_col, const double &bias, const int &result_row_size,
    const int &result_col_size, const int &kernel_rows, const int &kernel_cols);

// colwise long vector, 2d weight, 1d result
torch::Tensor sparse_convolution_input_based(
    const vector<vector<int>> &homo_snps,
    const vector<vector<int>> &hetero_snps, const int &num_snps,
    const int &num_individuals, const vector<vector<int>> &weight,
    const std::tuple<int, int> &stride = std::make_tuple(1, 1),
    const std::tuple<int, int> &dilation = std::make_tuple(1, 1),
    const int &bias = 0,
    const std::tuple<int, int> &output_size = std::make_tuple(0, 0));

torch::Tensor sparse_convolution_input_based(
    const vector<vector<int>> &homo_snps,
    const vector<vector<int>> &hetero_snps, const int &num_snps,
    const int &num_individuals, const vector<vector<int>> &weight,
    const int &stride = 1,
    const std::tuple<int, int> &dilation = std::make_tuple(1, 1),
    const int &bias = 0,
    const std::tuple<int, int> &output_size = std::make_tuple(0, 0));

torch::Tensor sparse_convolution_input_based(
    const vector<vector<int>> &homo_snps,
    const vector<vector<int>> &hetero_snps, const int &num_snps,
    const int &num_individuals, const vector<vector<int>> &weight,
    const std::tuple<int, int> &stride = std::make_tuple(1, 1),
    const int &dilation = 1, const int &bias = 0,
    const std::tuple<int, int> &output_size = std::make_tuple(0, 0));

torch::Tensor sparse_convolution_input_based(
    const vector<vector<int>> &homo_snps,
    const vector<vector<int>> &hetero_snps, const int &num_snps,
    const int &num_individuals, const vector<vector<int>> &weight,
    const int &stride = 1, const int &dilation = 1, const int &bias = 0,
    const std::tuple<int, int> &output_size = std::make_tuple(0, 0));

int handle_element(const int &i, const int &j, const int &stride_row,
                   const int &stride_col, const int &dilation_row,
                   const int &dilation_col, const int &k_row, const int &k_col,
                   vector<int> &starting_indices_homo,
                   vector<int> &starting_indices_hetero,
                   const vector<vector<int>> &homo_snps,
                   const vector<vector<int>> &hetero_snps,
                   const vector<vector<int>> &weight,
                   const vector<vector<int>> &double_weight);

/**
 * @brief Second version of sparse convolution. Iterates through each element of
 * the result only once. Better for longer filters.
 *
 * @param homo_snps
 * @param hetero_snps
 * @param num_snps
 * @param num_individuals
 * @param weight column-wise 2d vector
 * @param stride tuple of stride for (row, col)
 * @param dilation tuple of dilation for (row, col)
 * @param bias
 * @param output_size tuple of output size. Used for backpropagation
 * @return torch::Tensor
 */
torch::Tensor sparse_convolution_result_based(
    const vector<vector<int>> &homo_snps,
    const vector<vector<int>> &hetero_snps, const int &num_snps,
    const int &num_individuals, const vector<vector<int>> &weight,
    const std::tuple<int, int> &stride = std::make_tuple(1, 1),
    const std::tuple<int, int> &dilation = std::make_tuple(1, 1),
    const int &bias = 0,
    const std::tuple<int, int> &output_size = std::make_tuple(0, 0));

torch::Tensor sparse_convolution_result_based(
    const vector<vector<int>> &homo_snps,
    const vector<vector<int>> &hetero_snps, const int &num_snps,
    const int &num_individuals, const vector<vector<int>> &weight,
    const std::tuple<int, int> &stride = std::make_tuple(1, 1),
    const int &dilation = 1, const int &bias = 0,
    const std::tuple<int, int> &output_size = std::make_tuple(0, 0));

torch::Tensor sparse_convolution_result_based(
    const vector<vector<int>> &homo_snps,
    const vector<vector<int>> &hetero_snps, const int &num_snps,
    const int &num_individuals, const vector<vector<int>> &weight,
    const int &stride = 1,
    const std::tuple<int, int> &dilation = std::make_tuple(1, 1),
    const int &bias = 0,
    const std::tuple<int, int> &output_size = std::make_tuple(0, 0));

torch::Tensor sparse_convolution_result_based(
    const vector<vector<int>> &homo_snps,
    const vector<vector<int>> &hetero_snps, const int &num_snps,
    const int &num_individuals, const vector<vector<int>> &weight,
    const int &stride = 1, const int &dilation = 1, const int &bias = 0,
    const std::tuple<int, int> &output_size = std::make_tuple(0, 0));