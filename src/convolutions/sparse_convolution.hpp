#include "../data_handling/genotype_data.hpp"
#include <torch/torch.h>
#include <tuple>
#include <vector>

using std::vector;

/**
 * @brief Sparse convolution forward pass with support for minibatch and
 * channels
 *
 * @param input (minibatch, in_channels, GenotypeData)
 * @param weight (out_channels, in_channels, width, height)
 * @param bias optional bias
 * @param stride
 * @param dilation
 * @return torch::Tensor with shape (minibatch, out_channels, result_row_size,
 * result_col_size)
 */
torch::Tensor sparse_convolution(
    const vector<vector<GenotypeData>> &input,
    const vector<vector<vector<vector<double>>>> &weight,
    const vector<double> &bias,
    const std::tuple<int, int> &stride = std::make_tuple(1, 1),
    const std::tuple<int, int> &dilation = std::make_tuple(1, 1));

torch::Tensor sparse_convolution_blocked(
    const vector<vector<GenotypeData>> &input,
    const vector<vector<vector<vector<double>>>> &weight,
    const vector<double> &bias,
    const std::tuple<int, int> &stride = std::make_tuple(1, 1),
    const std::tuple<int, int> &dilation = std::make_tuple(1, 1),
    const int &block_size = 1000);

torch::Tensor
sparse_convolution(const vector<vector<GenotypeData>> &input,
                   const vector<vector<vector<vector<double>>>> &weight,
                   const vector<double> &bias,
                   const std::tuple<int, int> &stride = std::make_tuple(1, 1),
                   const int &dilation = 1);

torch::Tensor sparse_convolution(
    const vector<vector<GenotypeData>> &input,
    const vector<vector<vector<vector<double>>>> &weight,
    const vector<double> &bias, const int &stride = 1,
    const std::tuple<int, int> &dilation = std::make_tuple(1, 1));

torch::Tensor
sparse_convolution(const vector<vector<GenotypeData>> &input,
                   const vector<vector<vector<vector<double>>>> &weight,
                   const vector<double> &bias, const int &stride = 1,
                   const int &dilation = 1);

/**
 * @brief Sparse convolution for the backward pass.
 *
 * @param input original input (mini_batch, in_channels, GenotypeData)
 * @param grad_output (mini_batch, out_channels, result_cols, result_rows)
 * @param stride original dilation from forward
 * @param dilation original stride from forward
 * @param result_rows original kernel_rows
 * @param result_cols original kernel_cols
 * @return torch::Tensor (out_channels, in_channels, kernel_rows, kernel_cols)
 */
torch::Tensor sparse_convolution_backward(
    const vector<vector<GenotypeData>> &input,
    const vector<vector<vector<vector<double>>>> &grad_output,
    const std::tuple<int, int> &stride, const std::tuple<int, int> &dilation,
    const int &result_rows, const int &result_cols);

torch::Tensor sparse_convolution_backward(
    const vector<vector<GenotypeData>> &input,
    const vector<vector<vector<vector<double>>>> &grad_output,
    const int &stride, const std::tuple<int, int> &dilation,
    const int &result_rows, const int &result_cols);

torch::Tensor sparse_convolution_backward(
    const vector<vector<GenotypeData>> &input,
    const vector<vector<vector<vector<double>>>> &grad_output,
    const std::tuple<int, int> &stride, const int &dilation,
    const int &result_rows, const int &result_cols);

torch::Tensor sparse_convolution_backward(
    const vector<vector<GenotypeData>> &input,
    const vector<vector<vector<vector<double>>>> &grad_output,
    const int &stride, const int &dilation, const int &result_rows,
    const int &result_cols);

/**
 * @brief Performs single convolution operation for one 2d input and one 2d
 * weight. colwise long vector, 2d weight, 1d result, colwise kernel
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

torch::Tensor sparse_convolution_input_based_blocked(
    const GenotypeData &input, const vector<vector<double>> &weight,
    const int &stride_row, const int &stride_col, const int &dilation_row,
    const int &dilation_col, const double &bias, const int &result_row_size,
    const int &result_col_size, const int &kernel_rows, const int &kernel_cols,
    const int &block_size);

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

torch::Tensor sparse_convolution_optimized(
    const vector<vector<GenotypeData>> &input,
    const vector<vector<vector<vector<double>>>> &weight,
    const vector<double> &bias, const std::tuple<int, int> &stride,
    const std::tuple<int, int> &dilation);

void handle_element(
    const int &batch, const int &in_channel, const int &i, const int &j,
    const int &kernel_rows, const int &kernel_cols, const int &stride_row,
    const int &stride_col, const int &dilation_row, const int &dilation_col,
    const int &num_snps, const int &num_individuals, const int &result_row_size,
    const int &result_col_size, const int &result_size, const int &out_channels,
    const vector<vector<vector<vector<double>>>> &weight,
    vector<double> &result, const int &starting_point, const int &j_in_bounds);