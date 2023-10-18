#include "sparse_convolution.hpp"
#include "/usr/local/opt/libomp/include/omp.h"

torch::Tensor
sparse_convolution(const vector<vector<GenotypeData>> &input,
                   const vector<vector<vector<vector<double>>>> &weight,
                   const vector<double> &bias,
                   const std::tuple<int, int> &stride,
                   const std::tuple<int, int> &dilation) {
  if (input[0].size() != weight[0].size()) {
    throw std::invalid_argument(
        "Input in_channels does not match weight in_channels");
  }
  int batch_size = input.size();
  int in_channels = input[0].size();
  int out_channels = weight.size();
  int num_snps = input[0][0].num_snps;
  int num_individuals = input[0][0].num_individuals;
  int k_col = weight[0][0].size();
  int k_row = weight[0][0][0].size();
  int stride_row = std::get<0>(stride);
  int stride_col = std::get<1>(stride);
  int dilation_row = std::get<0>(dilation);
  int dilation_col = std::get<1>(dilation);
  int kernel_rows = (k_row - 1) * dilation_row + 1;
  int kernel_cols = (k_col - 1) * dilation_col + 1;
  int result_row_size =
      (num_snps - ((k_row - 1) * dilation_row + 1)) / stride_row + 1;
  int result_col_size =
      (num_individuals - ((k_col - 1) * dilation_col + 1)) / stride_col + 1;
  torch::Tensor result = torch::zeros(
      {batch_size, out_channels, result_row_size, result_col_size});
  // #pragma omp parallel for collapse(2)
  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < out_channels; j++) {
      for (int k = 0; k < in_channels; k++) {
        result[i][j] += sparse_convolution_input_based_optimized(
            input[i][k], weight[j][k], stride_row, stride_col, dilation_row,
            dilation_col, bias[j], result_row_size, result_col_size,
            kernel_rows, kernel_cols);
      }
    }
  }
  return result;
}

torch::Tensor sparse_convolution_blocked(
    const vector<vector<GenotypeData>> &input,
    const vector<vector<vector<vector<double>>>> &weight,
    const vector<double> &bias, const std::tuple<int, int> &stride,
    const std::tuple<int, int> &dilation, const int &block_size) {
  if (input[0].size() != weight[0].size()) {
    throw std::invalid_argument(
        "Input in_channels does not match weight in_channels");
  }
  int batch_size = input.size();
  int in_channels = input[0].size();
  int out_channels = weight.size();
  int num_snps = input[0][0].num_snps;
  int num_individuals = input[0][0].num_individuals;
  int k_col = weight[0][0].size();
  int k_row = weight[0][0][0].size();
  int stride_row = std::get<0>(stride);
  int stride_col = std::get<1>(stride);
  int dilation_row = std::get<0>(dilation);
  int dilation_col = std::get<1>(dilation);
  int kernel_rows = (k_row - 1) * dilation_row + 1;
  int kernel_cols = (k_col - 1) * dilation_col + 1;
  int result_row_size =
      (num_snps - ((k_row - 1) * dilation_row + 1)) / stride_row + 1;
  int result_col_size =
      (num_individuals - ((k_col - 1) * dilation_col + 1)) / stride_col + 1;
  torch::Tensor result = torch::zeros(
      {batch_size, out_channels, result_row_size, result_col_size});
  // #pragma omp parallel for
  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < out_channels; j++) {
      for (int k = 0; k < in_channels; k++) {
        result[i][j] += sparse_convolution_input_based_blocked(
            input[i][k], weight[j][k], stride_row, stride_col, dilation_row,
            dilation_col, bias[j], result_row_size, result_col_size,
            kernel_rows, kernel_cols, block_size);
      }
    }
  }
  return result;
}

torch::Tensor
sparse_convolution(const vector<vector<GenotypeData>> &input,
                   const vector<vector<vector<vector<double>>>> &weight,
                   const vector<double> &bias, const int &stride,
                   const std::tuple<int, int> &dilation) {
  return sparse_convolution(input, weight, bias,
                            std::make_tuple(stride, stride), dilation);
}

torch::Tensor
sparse_convolution(const vector<vector<GenotypeData>> &input,
                   const vector<vector<vector<vector<double>>>> &weight,
                   const vector<double> &bias,
                   const std::tuple<int, int> &stride, const int &dilation) {
  return sparse_convolution(input, weight, bias, stride,
                            std::make_tuple(dilation, dilation));
}

torch::Tensor
sparse_convolution(const vector<vector<GenotypeData>> &input,
                   const vector<vector<vector<vector<double>>>> &weight,
                   const vector<double> &bias, const int &stride,
                   const int &dilation) {
  return sparse_convolution(input, weight, bias,
                            std::make_tuple(stride, stride),
                            std::make_tuple(dilation, dilation));
}

torch::Tensor sparse_convolution_backward(
    const vector<vector<GenotypeData>> &input,
    const vector<vector<vector<vector<double>>>> &grad_output,
    const std::tuple<int, int> &stride, const std::tuple<int, int> &dilation,
    const int &result_rows, const int &result_cols) {
  int batch_size = input.size();
  int in_channels = input[0].size();
  int out_channels = grad_output[0].size();
  int k_col = grad_output[0][0].size();
  int k_row = grad_output[0][0][0].size();
  int stride_row = std::get<0>(stride);
  int stride_col = std::get<1>(stride);
  int dilation_row = std::get<0>(dilation);
  int dilation_col = std::get<1>(dilation);
  int kernel_rows = (k_row - 1) * dilation_row + 1;
  int kernel_cols = (k_col - 1) * dilation_col + 1;
  torch::Tensor result =
      torch::empty({out_channels, in_channels, result_rows, result_cols});
  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < out_channels; j++) {
      for (int k = 0; k < in_channels; k++) {
        result[j][k] += sparse_convolution_input_based_optimized(
            input[i][k], grad_output[i][j], stride_row, stride_col,
            dilation_row, dilation_col, 0, result_rows, result_cols,
            kernel_rows, kernel_cols);
      }
    }
  }
  result /= batch_size;
  return result;
}

torch::Tensor sparse_convolution_backward(
    const vector<vector<GenotypeData>> &input,
    const vector<vector<vector<vector<double>>>> &grad_output,
    const int &stride, const std::tuple<int, int> &dilation,
    const int &result_rows, const int &result_cols) {
  return sparse_convolution_backward(input, grad_output,
                                     std::make_tuple(stride, stride), dilation,
                                     result_rows, result_cols);
}

torch::Tensor sparse_convolution_backward(
    const vector<vector<GenotypeData>> &input,
    const vector<vector<vector<vector<double>>>> &grad_output,
    const std::tuple<int, int> &stride, const int &dilation,
    const int &result_rows, const int &result_cols) {
  return sparse_convolution_backward(input, grad_output, stride,
                                     std::make_tuple(dilation, dilation),
                                     result_rows, result_cols);
}

torch::Tensor sparse_convolution_backward(
    const vector<vector<GenotypeData>> &input,
    const vector<vector<vector<vector<double>>>> &grad_output,
    const int &stride, const int &dilation, const int &result_rows,
    const int &result_cols) {
  return sparse_convolution_backward(
      input, grad_output, std::make_tuple(stride, stride),
      std::make_tuple(dilation, dilation), result_rows, result_cols);
}

void handle_row_1d_result_colwise_kernel(
    const int &i, const int &j, const int &kernel_rows, const int &kernel_cols,
    const int &stride_row, const int &stride_col, const int &dilation_row,
    const int &dilation_col, const int &num_snps, const int &num_individuals,
    const int &result_row_size, const int &result_col_size,
    const vector<vector<double>> &weight, vector<double> &result) {
  int j_in_bounds = std::min(j, num_individuals - kernel_cols);
  int i_in_bounds = std::min(i, num_snps - kernel_rows);
  int c_start = j - j_in_bounds / stride_col * stride_col;
  int r_start = i - i_in_bounds / stride_row * stride_row;
  int result_col = (j - c_start) / stride_col;
  int result_row = (i - r_start) / stride_row;
  int result_index = result_col * result_row_size + result_row;
  int og_result_row = result_row;
  for (int c = c_start; c < kernel_cols; c += stride_col) {
    if (result_col < 0)
      break;
    if (c % dilation_col != 0 || result_col >= result_col_size) {
      result_col--;
      result_index -= result_row_size;
      continue;
    }
    int col_index = c / dilation_col;
    int og_result_index = result_index;
    for (int r = r_start; r < kernel_rows; r += stride_row) {
      if (result_row < 0) {
        break;
      }
      if (r % dilation_row != 0 || result_row >= result_row_size) {
        result_row--;
        result_index--;
        continue;
      }
      int row_index = r / dilation_row;
      result[result_index] += weight[col_index][row_index];
      result_row--;
      result_index--;
    }
    result_row = og_result_row;
    result_index = og_result_index;
    result_col--;
    result_index -= result_row_size;
  }
}

void handle_row_1d_result_colwise_kernel_optimized(
    const int &i, const int &j, const int &kernel_rows, const int &kernel_cols,
    const int &stride_row, const int &stride_col, const int &dilation_row,
    const int &dilation_col, const int &num_snps, const int &num_individuals,
    const int &result_row_size, const int &result_col_size,
    const vector<vector<double>> &weight, vector<double> &result) {
  int c_start =
      j - j / stride_col * stride_col; // the first index in the kernel
  int r_start = i - i / stride_row * stride_row;
  // how many strides left in the kernel vs number of strides still in bounds
  // (top left boundary)
  int col_strides_to_end =
      std::min((kernel_cols - 1 - c_start), j) / stride_col;
  int row_strides_to_end =
      std::min((kernel_rows - 1 - r_start), i) / stride_row;
  int c_end = c_start + col_strides_to_end * stride_col;
  int r_end = r_start + row_strides_to_end * stride_row;
  int result_col =
      (j - c_end) / stride_col; // map index of input and kernel to result
  int result_row = (i - r_end) / stride_row;
  int result_index = result_col * result_row_size + result_row;
  int og_result_row = result_row;
  for (int c = c_end; c >= 0; c -= stride_col) {
    if (result_col >= result_col_size)
      break;
    if (c % dilation_col != 0) {
      result_col++;
      result_index += result_row_size;
      continue;
    }
    int col_index = c / dilation_col;
    int og_result_index = result_index;
    for (int r = r_end; r >= 0; r -= stride_row) {
      if (result_row >= result_row_size) {
        break;
      }
      if (r % dilation_row != 0) {
        result_row++;
        result_index++;
        continue;
      }
      int row_index = r / dilation_row;
      result[result_index] += weight[col_index][row_index];
      result_row++;
      result_index++;
    }
    result_row = og_result_row;
    result_index = og_result_index;
    result_col++;
    result_index += result_row_size;
  }
}

torch::Tensor sparse_convolution_input_based_optimized(
    const GenotypeData &input, const vector<vector<double>> &weight,
    const int &stride_row, const int &stride_col, const int &dilation_row,
    const int &dilation_col, const double &bias, const int &result_row_size,
    const int &result_col_size, const int &kernel_rows,
    const int &kernel_cols) {
  int num_snps = input.num_snps;
  int num_individuals = input.num_individuals;
  vector<vector<int>> homo_snps = input.homo_snps;
  vector<vector<int>> hetero_snps = input.hetero_snps;
  int k_col = weight.size();
  int k_row = weight.at(0).size();
  vector<vector<double>> double_weight(k_col, vector<double>(k_row, 0));
  for (int i = 0; i < k_col; i++) {
    for (int j = 0; j < k_row; j++) {
      double_weight[i][j] = weight[i][j] * 2;
    }
  }
  bool stride_dilation_check_row =
      dilation_row > 1 && stride_row > 1 &&
      (stride_row % dilation_row == 0 || dilation_row % stride_row == 0);
  bool stride_dilation_check_col =
      dilation_col > 1 && stride_col > 1 &&
      (stride_col % dilation_col == 0 || dilation_col % stride_col == 0);
  vector<double> result(result_col_size * result_row_size, bias);
  for (int j = 0; j < num_individuals; j++) {
    if (stride_dilation_check_col &&
        j % std::min(dilation_col, stride_col) != 0) {
      continue;
    }
    for (int i : homo_snps.at(j)) {
      if (stride_dilation_check_row &&
          i % std::min(dilation_row, stride_row) != 0) {
        continue;
      }
      handle_row_1d_result_colwise_kernel(
          i, j, kernel_rows, kernel_cols, stride_row, stride_col, dilation_row,
          dilation_col, num_snps, num_individuals, result_row_size,
          result_col_size, double_weight, result);
    }
    for (int i : hetero_snps.at(j)) {
      if (stride_dilation_check_row &&
          i % std::min(dilation_row, stride_row) != 0) {
        continue;
      }
      handle_row_1d_result_colwise_kernel(
          i, j, kernel_rows, kernel_cols, stride_row, stride_col, dilation_row,
          dilation_col, num_snps, num_individuals, result_row_size,
          result_col_size, weight, result);
    }
  }
  torch::Tensor torch_result =
      torch::from_blob(result.data(), {result_col_size, result_row_size},
                       torch::TensorOptions().dtype(torch::kFloat64))
          .to(torch::kFloat32)
          .clone()
          .transpose(0, 1);
  return torch_result;
}

torch::Tensor sparse_convolution_input_based_blocked(
    const GenotypeData &input, const vector<vector<double>> &weight,
    const int &stride_row, const int &stride_col, const int &dilation_row,
    const int &dilation_col, const double &bias, const int &result_row_size,
    const int &result_col_size, const int &kernel_rows, const int &kernel_cols,
    const int &block_size) {
  int num_snps = input.num_snps;
  int num_individuals = input.num_individuals;
  vector<vector<int>> homo_snps = input.homo_snps;
  vector<vector<int>> hetero_snps = input.hetero_snps;
  int k_col = weight.size();
  int k_row = weight.at(0).size();
  vector<vector<double>> double_weight(k_col, vector<double>(k_row, 0));
  for (int i = 0; i < k_col; i++) {
    for (int j = 0; j < k_row; j++) {
      double_weight[i][j] = weight[i][j] * 2;
    }
  }
  bool stride_dilation_check_row =
      dilation_row > 1 && stride_row > 1 &&
      (stride_row % dilation_row == 0 || dilation_row % stride_row == 0);
  bool stride_dilation_check_col =
      dilation_col > 1 && stride_col > 1 &&
      (stride_col % dilation_col == 0 || dilation_col % stride_col == 0);
  vector<double> result(result_col_size * result_row_size, bias);
  for (int j = 0; j < num_individuals; j++) {
    if (stride_dilation_check_col &&
        j % std::min(dilation_col, stride_col) != 0) {
      continue;
    }
    for (int i : homo_snps.at(j)) {
      if (stride_dilation_check_row &&
          i % std::min(dilation_row, stride_row) != 0) {
        continue;
      }
      handle_row_1d_result_colwise_kernel_optimized(
          i, j, kernel_rows, kernel_cols, stride_row, stride_col, dilation_row,
          dilation_col, num_snps, num_individuals, result_row_size,
          result_col_size, double_weight, result);
    }
    for (int i : hetero_snps.at(j)) {
      if (stride_dilation_check_row &&
          i % std::min(dilation_row, stride_row) != 0) {
        continue;
      }
      handle_row_1d_result_colwise_kernel_optimized(
          i, j, kernel_rows, kernel_cols, stride_row, stride_col, dilation_row,
          dilation_col, num_snps, num_individuals, result_row_size,
          result_col_size, weight, result);
    }
  }
  torch::Tensor torch_result =
      torch::from_blob(result.data(), {result_col_size, result_row_size},
                       torch::TensorOptions().dtype(torch::kFloat64))
          .to(torch::kFloat32)
          .clone()
          .transpose(0, 1);
  return torch_result;
}

// torch::Tensor sparse_convolution_input_based_blocked(
//     const GenotypeData &input, const vector<vector<double>> &weight,
//     const int &stride_row, const int &stride_col, const int &dilation_row,
//     const int &dilation_col, const double &bias, const int &result_row_size,
//     const int &result_col_size, const int &kernel_rows, const int
//     &kernel_cols, const int &block_size) {
//   int num_snps = input.num_snps;
//   int num_individuals = input.num_individuals;
//   vector<vector<int>> homo_snps = input.homo_snps;
//   vector<vector<int>> hetero_snps = input.hetero_snps;
//   int k_col = weight.size();
//   int k_row = weight.at(0).size();
//   vector<vector<double>> double_weight(k_col, vector<double>(k_row, 0));
//   for (int i = 0; i < k_col; i++) {
//     for (int j = 0; j < k_row; j++) {
//       double_weight[i][j] = weight[i][j] * 2;
//     }
//   }
//   bool stride_dilation_check_row =
//       dilation_row > 1 && stride_row > 1 &&
//       (stride_row % dilation_row == 0 || dilation_row % stride_row == 0);
//   bool stride_dilation_check_col =
//       dilation_col > 1 && stride_col > 1 &&
//       (stride_col % dilation_col == 0 || dilation_col % stride_col == 0);
//   vector<double> result(result_col_size * result_row_size, bias);
//   // try blocking
//   vector<int> curr_homo_inds(num_individuals, 0);
//   vector<int> curr_hetero_inds(num_individuals, 0);
// #pragma omp parallel for
//   for (int cutoff = block_size; cutoff < num_snps + block_size;
//        cutoff += block_size) {
//     for (int j = 0; j < num_individuals; j++) {
//       if (stride_dilation_check_col &&
//           j % std::min(dilation_col, stride_col) != 0) {
//         continue;
//       }
//       vector<int> homo_col = homo_snps.at(j);
//       int homo_len = homo_col.size();
//       int curr_homo_ind = curr_homo_inds[j];
//       int i = homo_col[curr_homo_ind];
//       while (curr_homo_ind < homo_len and i < cutoff) {
//         if (stride_dilation_check_row &&
//             i % std::min(dilation_row, stride_row) != 0) {
//           continue;
//         }
//         handle_row_1d_result_colwise_kernel(
//             i, j, kernel_rows, kernel_cols, stride_row, stride_col,
//             dilation_row, dilation_col, num_snps, num_individuals,
//             result_row_size, result_col_size, double_weight, result);
//         curr_homo_ind++;
//         i = homo_col[curr_homo_ind];
//       }
//       curr_homo_inds[j] = curr_homo_ind;

//       vector<int> hetero_col = hetero_snps.at(j);
//       int hetero_len = hetero_col.size();
//       int curr_hetero_ind = curr_hetero_inds[j];
//       i = hetero_col[curr_hetero_ind];
//       while (curr_hetero_ind < hetero_len and i < cutoff) {
//         if (stride_dilation_check_row &&
//             i % std::min(dilation_row, stride_row) != 0) {
//           continue;
//         }
//         handle_row_1d_result_colwise_kernel(
//             i, j, kernel_rows, kernel_cols, stride_row, stride_col,
//             dilation_row, dilation_col, num_snps, num_individuals,
//             result_row_size, result_col_size, weight, result);
//         curr_hetero_ind++;
//         i = hetero_col[curr_hetero_ind];
//       }
//       curr_hetero_inds[j] = curr_hetero_ind;
//     }
//   }

//   torch::Tensor torch_result =
//       torch::from_blob(result.data(), {result_col_size, result_row_size},
//                        torch::TensorOptions().dtype(torch::kFloat64))
//           .to(torch::kFloat32)
//           .clone()
//           .transpose(0, 1);
//   return torch_result;
// }

void handle_row_1d_result_colwise(
    const int &i, const int &j, const int &kernel_rows, const int &kernel_cols,
    const int &stride_row, const int &stride_col, const int &dilation_row,
    const int &dilation_col, const int &num_snps, const int &num_individuals,
    const int &result_row_size, const int &result_col_size,
    const vector<vector<int>> &weight, vector<int> &result) {
  int i_to_use = std::min(i, num_snps - kernel_rows);
  int j_to_use = std::min(j, num_individuals - kernel_cols);
  for (int r = i - i_to_use / stride_row * stride_row; r < kernel_rows;
       r += stride_row) {
    int result_row = (i - r) / stride_row;
    if (result_row < 0)
      break;
    if (r % dilation_row != 0 || result_row >= result_row_size)
      continue;
    int row_index = r / dilation_row;
    for (int c = j - j_to_use / stride_col * stride_col; c < kernel_cols;
         c += stride_col) {
      int result_col = (j - c) / stride_col;
      if (result_col < 0)
        break;
      if (c % dilation_col != 0 || result_col >= result_col_size)
        continue;
      result[result_col * result_row_size + result_row] +=
          weight[row_index][c / dilation_col];
    }
  }
}

torch::Tensor sparse_convolution_input_based(
    const vector<vector<int>> &homo_snps,
    const vector<vector<int>> &hetero_snps, const int &num_snps,
    const int &num_individuals, const vector<vector<int>> &weight,
    const std::tuple<int, int> &stride, const std::tuple<int, int> &dilation,
    const int &bias, const std::tuple<int, int> &output_size) {

  int k_row = weight.size();
  int k_col = weight.at(0).size();
  vector<vector<int>> double_weight(k_row, vector<int>(k_col, 0));
  for (int i = 0; i < k_row; i++) {
    for (int j = 0; j < k_col; j++) {
      double_weight[i][j] = weight[i][j] * 2;
    }
  }
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
        (num_snps - ((k_row - 1) * dilation_row + 1)) / stride_row + 1;
    result_col_size =
        (num_individuals - ((k_col - 1) * dilation_col + 1)) / stride_col + 1;
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
  vector<int> result(result_col_size * result_row_size, bias);
  for (int j = 0; j < num_individuals; j++) {
    if (stride_dilation_check_col &&
        j % std::min(dilation_col, stride_col) != 0) {
      continue;
    }
    for (int i : homo_snps.at(j)) {
      if (stride_dilation_check_row &&
          i % std::min(dilation_row, stride_row) != 0) {
        continue;
      }
      handle_row_1d_result_colwise(i, j, kernel_rows, kernel_cols, stride_row,
                                   stride_col, dilation_row, dilation_col,
                                   num_snps, num_individuals, result_row_size,
                                   result_col_size, double_weight, result);
    }
    for (int i : hetero_snps.at(j)) {
      if (stride_dilation_check_row &&
          i % std::min(dilation_row, stride_row) != 0) {
        continue;
      }
      handle_row_1d_result_colwise(i, j, kernel_rows, kernel_cols, stride_row,
                                   stride_col, dilation_row, dilation_col,
                                   num_snps, num_individuals, result_row_size,
                                   result_col_size, weight, result);
    }
  }
  torch::Tensor torch_result =
      torch::from_blob(result.data(), {result_col_size, result_row_size},
                       torch::TensorOptions().dtype(torch::kInt32))
          .clone()
          .transpose(0, 1);
  return torch_result;
}

torch::Tensor sparse_convolution_input_based(
    const vector<vector<int>> &homo_snps,
    const vector<vector<int>> &hetero_snps, const int &num_snps,
    const int &num_individuals, const vector<vector<int>> &weight,
    const int &stride, const std::tuple<int, int> &dilation, const int &bias,
    const std::tuple<int, int> &output_size) {
  return sparse_convolution_input_based(
      homo_snps, hetero_snps, num_snps, num_individuals, weight,
      std::make_tuple(stride, stride), dilation, bias, output_size);
}

torch::Tensor sparse_convolution_input_based(
    const vector<vector<int>> &homo_snps,
    const vector<vector<int>> &hetero_snps, const int &num_snps,
    const int &num_individuals, const vector<vector<int>> &weight,
    const std::tuple<int, int> &stride, const int &dilation, const int &bias,
    const std::tuple<int, int> &output_size) {
  return sparse_convolution_input_based(
      homo_snps, hetero_snps, num_snps, num_individuals, weight, stride,
      std::make_tuple(dilation, dilation), bias, output_size);
}

torch::Tensor sparse_convolution_input_based(
    const vector<vector<int>> &homo_snps,
    const vector<vector<int>> &hetero_snps, const int &num_snps,
    const int &num_individuals, const vector<vector<int>> &weight,
    const int &stride, const int &dilation, const int &bias,
    const std::tuple<int, int> &output_size) {
  return sparse_convolution_input_based(
      homo_snps, hetero_snps, num_snps, num_individuals, weight,
      std::make_tuple(stride, stride), std::make_tuple(dilation, dilation),
      bias, output_size);
}

torch::Tensor sparse_convolution_optimized(
    const vector<vector<GenotypeData>> &input,
    const vector<vector<vector<vector<double>>>> &weight,
    const vector<double> &bias, const std::tuple<int, int> &stride,
    const std::tuple<int, int> &dilation) {
  if (input[0].size() != weight[0].size()) {
    throw std::invalid_argument(
        "Input in_channels does not match weight in_channels");
  }
  int batch_size = input.size();
  int in_channels = input[0].size();
  int out_channels = weight.size();
  int num_snps = input[0][0].num_snps;
  int num_individuals = input[0][0].num_individuals;
  int k_col = weight[0][0].size();
  int k_row = weight[0][0][0].size();
  int stride_row = std::get<0>(stride);
  int stride_col = std::get<1>(stride);
  int dilation_row = std::get<0>(dilation);
  int dilation_col = std::get<1>(dilation);
  int kernel_rows = (k_row - 1) * dilation_row + 1;
  int kernel_cols = (k_col - 1) * dilation_col + 1;
  int result_row_size =
      (num_snps - ((k_row - 1) * dilation_row + 1)) / stride_row + 1;
  int result_col_size =
      (num_individuals - ((k_col - 1) * dilation_col + 1)) / stride_col + 1;
  vector<vector<vector<vector<double>>>> double_weight(
      out_channels, vector<vector<vector<double>>>(
                        in_channels, vector<vector<double>>(
                                         k_col, vector<double>(k_row, 0))));
  for (int i = 0; i < out_channels; i++) {
    for (int j = 0; j < in_channels; j++) {
      for (int k = 0; k < k_col; k++) {
        for (int l = 0; l < k_row; l++) {
          double_weight[i][j][k][l] = weight[i][j][k][l] * 2;
        }
      }
    }
  }
  bool stride_dilation_check_row =
      dilation_row > 1 && stride_row > 1 &&
      (stride_row % dilation_row == 0 || dilation_row % stride_row == 0);
  bool stride_dilation_check_col =
      dilation_col > 1 && stride_col > 1 &&
      (stride_col % dilation_col == 0 || dilation_col % stride_col == 0);
  vector<double> result(
      batch_size * out_channels * result_col_size * result_row_size, 0);
  int result_size = result_col_size * result_row_size;
  int single_output_size = out_channels * result_size;
  // initialize with biases
  int index = 0;
  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < out_channels; j++) {
      std::fill_n(result.begin() + index, result_size, bias[j]);
    }
    index += result_size;
  }
  int starting_point = 0;
  // actual sparse convolution
  for (int batch = 0; batch < batch_size; batch++) {
    for (int in_channel = 0; in_channel < in_channels; in_channel++) {
      GenotypeData genotype_data = input[batch][in_channel];
      vector<vector<int>> homo_snps = genotype_data.homo_snps;
      vector<vector<int>> hetero_snps = genotype_data.hetero_snps;
      for (int j = 0; j < num_individuals; j++) {
        if (stride_dilation_check_col &&
            j % std::min(dilation_col, stride_col) != 0) {
          continue;
        }
        int j_in_bounds = std::min(j, num_individuals - kernel_cols);
        for (int i : homo_snps.at(j)) {
          if (stride_dilation_check_row &&
              i % std::min(dilation_row, stride_row) != 0) {
            continue;
          }
          handle_element(batch, in_channel, i, j, kernel_rows, kernel_cols,
                         stride_row, stride_col, dilation_row, dilation_col,
                         num_snps, num_individuals, result_row_size,
                         result_col_size, result_size, out_channels,
                         double_weight, result, starting_point, j_in_bounds);
        }
        for (int i : hetero_snps.at(j)) {
          if (stride_dilation_check_row &&
              i % std::min(dilation_row, stride_row) != 0) {
            continue;
          }
          handle_element(batch, in_channel, i, j, kernel_rows, kernel_cols,
                         stride_row, stride_col, dilation_row, dilation_col,
                         num_snps, num_individuals, result_row_size,
                         result_col_size, result_size, out_channels, weight,
                         result, starting_point, j_in_bounds);
        }
      }
    }
    starting_point += single_output_size;
  }
  torch::Tensor torch_result =
      torch::from_blob(
          result.data(),
          {batch_size, out_channels, result_col_size, result_row_size},
          torch::TensorOptions().dtype(torch::kFloat64))
          .to(torch::kFloat32)
          .clone()
          .transpose(2, 3);
  return torch_result;
}

void handle_element(
    const int &batch, const int &in_channel, const int &i, const int &j,
    const int &kernel_rows, const int &kernel_cols, const int &stride_row,
    const int &stride_col, const int &dilation_row, const int &dilation_col,
    const int &num_snps, const int &num_individuals, const int &result_row_size,
    const int &result_col_size, const int &result_size, const int &out_channels,
    const vector<vector<vector<vector<double>>>> &weight,
    vector<double> &result, const int &starting_point, const int &j_in_bounds) {
  int i_in_bounds = std::min(i, num_snps - kernel_rows);
  int c_start = j - j_in_bounds / stride_col * stride_col;
  int r_start = i - i_in_bounds / stride_row * stride_row;
  int result_col = (j - c_start) / stride_col;
  int result_row = (i - r_start) / stride_row;
  int result_index = starting_point + result_col * result_row_size + result_row;
  int og_result_row = result_row;
  for (int out_channel = 0; out_channel < out_channels; out_channel++) {
    for (int c = c_start; c < kernel_cols; c += stride_col) {
      if (result_col < 0)
        break;
      if (c % dilation_col != 0 || result_col >= result_col_size) {
        result_col--;
        result_index -= result_row_size;
        continue;
      }
      int col_index = c / dilation_col;
      int og_result_index = result_index;
      for (int r = r_start; r < kernel_rows; r += stride_row) {
        if (result_row < 0) {
          break;
        }
        if (r % dilation_row != 0 || result_row >= result_row_size) {
          result_row--;
          result_index--;
          continue;
        }
        int row_index = r / dilation_row;
        result[result_index] +=
            weight[out_channel][in_channel][col_index][row_index];
        result_row--;
        result_index--;
      }
      result_row = og_result_row;
      result_index = og_result_index;
      result_col--;
      result_index -= result_row_size;
    }
    result_index += result_size;
  }
}