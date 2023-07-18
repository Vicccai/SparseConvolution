#include "dense_convolution.hpp"

torch::Tensor dense_convolution(const vector<vector<int>> &data,
                                const vector<vector<int>> &weight,
                                const std::tuple<int, int> &stride,
                                const std::tuple<int, int> &dilation,
                                const int &bias) {
  int num_snps = data.size();
  int num_individuals = data.at(0).size();
  int k_row = weight.size();
  int k_col = weight.at(0).size();
  int stride_row = std::get<0>(stride);
  int stride_col = std::get<1>(stride);
  int dilation_row = std::get<0>(dilation);
  int dilation_col = std::get<1>(dilation);
  int result_row_size =
      (num_snps - ((k_row - 1) * dilation_row + 1)) / stride_row + 1;
  int result_col_size =
      (num_individuals - ((k_col - 1) * dilation_col + 1)) / stride_col + 1;
  vector<int> result(result_row_size * result_col_size, bias);
  for (int i = 0; i < result_row_size; i++) {
    for (int j = 0; j < result_col_size; j++) {
      for (int a = 0; a < k_row; a++) {
        for (int b = 0; b < k_col; b++) {
          result[i * result_col_size + j] +=
              weight[a][b] * data[i * stride_row + a * dilation_row]
                                 [j * stride_col + b * dilation_col];
        }
      }
    }
  }
  torch::Tensor torch_result =
      torch::from_blob(result.data(), {result_row_size, result_col_size},
                       torch::TensorOptions().dtype(torch::kInt32))
          .clone();
  return torch_result;
}

torch::Tensor dense_convolution(const vector<vector<int>> &data,
                                const vector<vector<int>> &weight,
                                const int &stride,
                                const std::tuple<int, int> &dilation,
                                const int &bias) {
  return dense_convolution(data, weight, std::make_tuple(stride, stride),
                           dilation, bias);
}

torch::Tensor dense_convolution(const vector<vector<int>> &data,
                                const vector<vector<int>> &weight,
                                const std::tuple<int, int> &stride,
                                const int &dilation, const int &bias) {
  return dense_convolution(data, weight, stride,
                           std::make_tuple(dilation, dilation), bias);
}

torch::Tensor dense_convolution(const vector<vector<int>> &data,
                                const vector<vector<int>> &weight,
                                const int &stride, const int &dilation,
                                const int &bias) {
  return dense_convolution(data, weight, std::make_tuple(stride, stride),
                           std::make_tuple(dilation, dilation), bias);
}