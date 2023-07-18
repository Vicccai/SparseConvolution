#include <torch/torch.h>
#include <tuple>
#include <vector>

using std::vector;

void do_convolution(
    const int &stride_dilation_check_row, const int &stride_dilation_check_col,
    const int &kernel_rows, const int &kernel_cols, const int &stride_row,
    const int &stride_col, const int &dilation_row, const int &dilation_col,
    const int &num_rows, const int &num_cols, const int &result_row_size,
    const int &result_col_size, const vector<vector<int>> &indices,
    const vector<vector<double>> &values, const vector<vector<double>> &weight,
    vector<double> &result);

// row-wise
torch::Tensor general_sparse_convolution(
    const vector<vector<int>> &indices, const vector<vector<double>> &values,
    const int &num_rows, const int &num_cols,
    const vector<vector<double>> &weight,
    const std::tuple<int, int> &stride = std::make_tuple(1, 1),
    const std::tuple<int, int> &dilation = std::make_tuple(1, 1),
    const int &bias = 0,
    const std::tuple<int, int> &output_size = std::make_tuple(0, 0));

torch::Tensor general_sparse_convolution(
    const vector<vector<int>> &indices, const vector<vector<double>> &values,
    const int &num_rows, const int &num_cols,
    const vector<vector<double>> &weight, const int &stride = 1,
    const std::tuple<int, int> &dilation = std::make_tuple(1, 1),
    const int &bias = 0,
    const std::tuple<int, int> &output_size = std::make_tuple(0, 0));

torch::Tensor general_sparse_convolution(
    const vector<vector<int>> &indices, const vector<vector<double>> &values,
    const int &num_rows, const int &num_cols,
    const vector<vector<double>> &weight,
    const std::tuple<int, int> &stride = std::make_tuple(1, 1),
    const int &dilation = 1, const int &bias = 0,
    const std::tuple<int, int> &output_size = std::make_tuple(0, 0));

torch::Tensor general_sparse_convolution(
    const vector<vector<int>> &indices, const vector<vector<double>> &values,
    const int &num_rows, const int &num_cols,
    const vector<vector<double>> &weight, const int &stride = 1,
    const int &dilation = 1, const int &bias = 0,
    const std::tuple<int, int> &output_size = std::make_tuple(0, 0));