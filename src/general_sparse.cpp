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
    vector<double> &result) {
  for (int i = 0; i < num_rows; i++) {
    if (stride_dilation_check_row &&
        i % std::min(dilation_row, stride_row) != 0) {
      continue;
    }
    int iToUse = std::min(i, num_rows - kernel_rows);
    for (int j : indices.at(i)) {
      if (stride_dilation_check_col &&
          j % std::min(dilation_col, stride_col) != 0) {
        continue;
      }
      int jToUse = std::min(j, num_cols - kernel_cols);
      for (int r = i - iToUse / stride_row * stride_row; r < kernel_rows;
           r += stride_row) {
        int result_row = (i - r) / stride_row;
        if (result_row < 0)
          break;
        if (r % dilation_row != 0 || result_row >= result_row_size)
          continue;
        int row_index = r / dilation_row;
        for (int c = j - jToUse / stride_col * stride_col; c < kernel_cols;
             c += stride_col) {
          int result_col = (j - c) / stride_col;
          if (result_col < 0)
            break;
          if (c % dilation_col != 0 || result_col >= result_col_size)
            continue;
          result[result_row * result_col_size + result_col] +=
              weight[row_index][c / dilation_col] * values[i][j];
        }
      }
    }
  }
}

// row-wise
torch::Tensor general_sparse_convolution(
    const vector<vector<int>> &indices, const vector<vector<double>> &values,
    const int &num_rows, const int &num_cols,
    const vector<vector<double>> &weight,
    const std::tuple<int, int> &stride = std::make_tuple(1, 1),
    const std::tuple<int, int> &dilation = std::make_tuple(1, 1),
    const int &bias = 0,
    const std::tuple<int, int> &output_size = std::make_tuple(0, 0)) {

  int k_row = weight.size();
  int k_col = weight.at(0).size();
  int stride_row = std::get<0>(stride);
  int stride_col = std::get<1>(stride);
  int dilation_row = std::get<0>(dilation);
  int dilation_col = std::get<1>(dilation);
  int kernel_rows = (k_row - 1) * dilation_row + 1;
  int kernel_cols = (k_col - 1) * dilation_col + 1;
  int result_row_size = 0;
  int result_col_size = 0;
  if (std::get<0>(output_size) == 0 && std::get<1>(output_size) == 0) {
    result_row_size =
        (num_rows - ((k_row - 1) * dilation_row + 1)) / stride_row + 1;
    result_col_size =
        (num_cols - ((k_col - 1) * dilation_col + 1)) / stride_col + 1;
  } else {
    result_row_size = std::get<0>(output_size);
    result_col_size = std::get<1>(output_size);
  }
  bool stride_dilation_check_row =
      dilation_row > 1 && stride_row > 1 &&
      (stride_row % dilation_row == 0 || dilation_row % stride_row == 0);
  bool stride_dilation_check_col =
      dilation_col > 1 && stride_col > 1 &&
      (stride_col % dilation_col == 0 || dilation_col % stride_col == 0);
  vector<double> result(result_row_size * result_col_size, bias);
  do_convolution(stride_dilation_check_row, stride_dilation_check_col,
                 kernel_rows, kernel_cols, stride_row, stride_col, dilation_row,
                 dilation_col, num_rows, num_cols, result_row_size,
                 result_col_size, indices, values, weight, result);
  torch::Tensor torch_result =
      torch::from_blob(result.data(), {result_row_size, result_col_size},
                       torch::TensorOptions().dtype(torch::kDouble));
  return torch_result;
}

torch::Tensor general_sparse_convolution(
    const vector<vector<int>> &indices, const vector<vector<double>> &values,
    const int &num_rows, const int &num_cols,
    const vector<vector<double>> &weight, const int &stride = 1,
    const std::tuple<int, int> &dilation = std::make_tuple(1, 1),
    const int &bias = 0,
    const std::tuple<int, int> &output_size = std::make_tuple(0, 0)) {
  return general_sparse_convolution(indices, values, num_rows, num_cols, weight,
                                    std::make_tuple(stride, stride), dilation,
                                    bias, output_size);
}

torch::Tensor general_sparse_convolution(
    const vector<vector<int>> &indices, const vector<vector<double>> &values,
    const int &num_rows, const int &num_cols,
    const vector<vector<double>> &weight,
    const std::tuple<int, int> &stride = std::make_tuple(1, 1),
    const int &dilation = 1, const int &bias = 0,
    const std::tuple<int, int> &output_size = std::make_tuple(0, 0)) {
  return general_sparse_convolution(indices, values, num_rows, num_cols, weight,
                                    stride, std::make_tuple(dilation, dilation),
                                    bias, output_size);
}

torch::Tensor general_sparse_convolution(
    const vector<vector<int>> &indices, const vector<vector<double>> &values,
    const int &num_rows, const int &num_cols,
    const vector<vector<double>> &weight, const int &stride = 1,
    const int &dilation = 1, const int &bias = 0,
    const std::tuple<int, int> &output_size = std::make_tuple(0, 0)) {
  return general_sparse_convolution(indices, values, num_rows, num_cols, weight,
                                    std::make_tuple(stride, stride),
                                    std::make_tuple(dilation, dilation), bias,
                                    output_size);
}