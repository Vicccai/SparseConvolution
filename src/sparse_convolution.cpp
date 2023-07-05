// #include <torch/extension.h>
#include <torch/torch.h>
#include <tuple>
#include <vector>

using std::vector;

void handle_row(const int &i, const int &j, const int &kernel_size,
                const int &stride, const int &dilation,
                const int &result_row_size, const int &result_col_size,
                const vector<vector<int>> &weight,
                vector<vector<int>> &result) {
  for (int r = i - i / stride * stride; r < kernel_size; r += stride) {
    int result_row = (i - r) / stride;
    if (result_row < 0)
      break;
    if (r % dilation != 0 || result_row >= result_row_size)
      continue;
    for (int c = j - j / stride * stride; c < kernel_size; c += stride) {
      int result_col = (j - c) / stride;
      if (result_col < 0)
        break;
      if (c % dilation != 0 || result_col >= result_col_size)
        continue;
      result[result_row][result_col] += weight[r][c];
    }
  }
}

torch::Tensor sparse_convolution(
    const vector<vector<int>> &homo_snps,
    const vector<vector<int>> &hetero_snps, const int &num_snps,
    const int &num_individuals, torch::Tensor &weight,
    const std::tuple<int, int> &output_size = std::make_tuple(0, 0),
    const int &bias = 0, const int &stride = 1, const int &dilation = 1) {
  torch::Tensor double_weight = weight * 2;
  int k = weight.sizes().at(0);
  int result_row_size = 0;
  int result_col_size = 0;
  if (std::get<0>(output_size) == 0 && std::get<1>(output_size) == 0) {
    result_row_size = (num_snps - ((k - 1) * dilation + 1)) / stride + 1;
    result_col_size = (num_individuals - ((k - 1) * dilation + 1)) / stride + 1;
  } else {
    result_row_size = std::get<0>(output_size);
    result_col_size = std::get<1>(output_size);
  }
  bool stride_dilation_check =
      dilation > 1 && stride > 1 &&
      (stride % dilation == 0 || dilation % stride == 0);
  torch::Tensor result =
      torch::zeros({result_row_size, result_col_size}) + bias;
  for (int i = 0; i < num_snps; i++) {
    // if (stride_dilation_check && i % std::min(dilation, stride) != 0) {
    //   continue;
    // }
    for (int j : homo_snps.at(i)) {
      // if (stride_dilation_check && j % std::min(dilation, stride) != 0) {
      //   continue;
      // }
      for (int r = i - i / stride * stride; r < k; r++) {
        int result_row = (i - r) / stride;
        if (result_row < 0)
          break;
        if (r % dilation != 0 || result_row >= result_row_size)
          continue;
        for (int c = j - j / stride * stride; c < k; c++) {
          int result_col = (j - c) / stride;
          if (result_col < 0)
            break;
          if (c % dilation != 0 || result_col >= result_col_size)
            continue;
          result[result_row][result_col] += double_weight[r][c];
        }
      }
    }
    for (int j : hetero_snps.at(i)) {
      // if (stride_dilation_check && j % std::min(dilation, stride) != 0) {
      //   continue;
      // }
      for (int r = i - i / stride * stride; r < k; r++) {
        int result_row = (i - r) / stride;
        if (result_row < 0)
          break;
        if (r % dilation != 0 || result_row >= result_row_size)
          continue;
        for (int c = j - j / stride * stride; c < k; c++) {
          int result_col = (j - c) / stride;
          if (result_col < 0)
            break;
          if (c % dilation != 0 || result_col >= result_col_size)
            continue;
          result[result_row][result_col] += weight[r][c];
        }
      }
    }
  }
  return result;
}

vector<vector<int>> sparse_convolution_2(
    const vector<vector<int>> &homo_snps,
    const vector<vector<int>> &hetero_snps, const int &num_snps,
    const int &num_individuals, const vector<vector<int>> &weight,
    const std::tuple<int, int> &output_size = std::make_tuple(0, 0),
    const int &bias = 0, const int &stride = 1, const int &dilation = 1) {
  int k = weight.size();
  int kernel_size = (k - 1) * dilation + 1;
  vector<vector<int>> double_weight(k, vector<int>(k, 0));
  for (int i = 0; i < k; i++) {
    for (int j = 0; j < k; j++) {
      double_weight[i][j] = weight[i][j] * 2;
    }
  }
  int result_row_size = 0;
  int result_col_size = 0;
  if (std::get<0>(output_size) == 0 && std::get<1>(output_size) == 0) {
    result_row_size = (num_snps - ((k - 1) * dilation + 1)) / stride + 1;
    result_col_size = (num_individuals - ((k - 1) * dilation + 1)) / stride + 1;
  } else {
    result_row_size = std::get<0>(output_size);
    result_col_size = std::get<1>(output_size);
  }
  bool stride_dilation_check =
      dilation > 1 && stride > 1 &&
      (stride % dilation == 0 || dilation % stride == 0);
  vector<vector<int>> result(result_row_size,
                             vector<int>(result_col_size, bias));
  for (int i = 0; i < num_snps; i++) {
    if (stride_dilation_check && i % std::min(dilation, stride) != 0) {
      continue;
    }
    for (int j : homo_snps.at(i)) {
      if (stride_dilation_check && j % std::min(dilation, stride) != 0) {
        continue;
      }
      int r_start = i - i / stride * stride;
      handle_row(i, j, kernel_size, stride, dilation, result_row_size,
                 result_col_size, double_weight, result);
    }
    for (int j : hetero_snps.at(i)) {
      if (stride_dilation_check && j % std::min(dilation, stride) != 0) {
        continue;
      }
      int r_start = i - i / stride * stride;
      handle_row(i, j, kernel_size, stride, dilation, result_row_size,
                 result_col_size, weight, result);
    }
  }
  return result;
}

vector<vector<int>> sparse_convolution_3(
    const vector<vector<int>> &homo_snps,
    const vector<vector<int>> &hetero_snps, const int &num_snps,
    const int &num_individuals, const vector<vector<int>> &weight,
    const std::tuple<int, int> &output_size = std::make_tuple(0, 0),
    const int &bias = 0, const int &stride = 1, const int &dilation = 1) {

  int k = weight.size();
  vector<vector<int>> double_weight(k, vector<int>(k, 0));
  for (int i = 0; i < k; i++) {
    for (int j = 0; j < k; j++) {
      double_weight[i][j] = weight[i][j] * 2;
    }
  }
  int result_row_size = 0;
  int result_col_size = 0;
  int kernel_size = (k - 1) * dilation + 1;
  if (std::get<0>(output_size) == 0 && std::get<1>(output_size) == 0) {
    result_row_size = (num_snps - ((k - 1) * dilation + 1)) / stride + 1;
    result_col_size = (num_individuals - ((k - 1) * dilation + 1)) / stride + 1;
  } else {
    result_row_size = std::get<0>(output_size);
    result_col_size = std::get<1>(output_size);
  }
  bool stride_dilation_check =
      dilation > 1 && stride > 1 &&
      (stride % dilation == 0 || dilation % stride == 0);
  vector<vector<int>> result(result_row_size,
                             vector<int>(result_col_size, bias));
  for (int j = 0; j < num_individuals; j++) {
    if (stride_dilation_check && j % std::min(dilation, stride) != 0) {
      continue;
    }
    for (int i : homo_snps.at(j)) {
      if (stride_dilation_check && i % std::min(dilation, stride) != 0) {
        continue;
      }
      handle_row(i, j, kernel_size, stride, dilation, result_row_size,
                 result_col_size, double_weight, result);
    }
    for (int i : hetero_snps.at(j)) {
      if (stride_dilation_check && i % std::min(dilation, stride) != 0) {
        continue;
      }
      handle_row(i, j, kernel_size, stride, dilation, result_row_size,
                 result_col_size, weight, result);
    }
  }
  return result;
}