#include <vector>

using std::vector;

torch::Tensor dense_convolution(const vector<vector<int>> &data,
                                const vector<vector<int>> &weight,
                                const int &stride = 1, const int &dilation = 1,
                                const int &bias = 0) {
  int num_snps = data.size();
  int num_individuals = data.at(0).size();
  int k_row = weight.size();
  int k_col = weight.at(0).size();
  int result_row_size = (num_snps - ((k_row - 1) * dilation + 1)) / stride + 1;
  int result_col_size =
      (num_individuals - ((k_col - 1) * dilation + 1)) / stride + 1;
  vector<int> result(result_row_size * result_col_size, bias);
  for (int i = 0; i < result_row_size; i++) {
    for (int j = 0; j < result_col_size; j++) {
      for (int a = 0; a < k_row; a++) {
        for (int b = 0; b < k_col; b++) {
          result[i * result_col_size + j] +=
              weight[a][b] *
              data[i * stride + a * dilation][j * stride + b * dilation];
        }
      }
    }
  }
  torch::Tensor torch_result =
      torch::from_blob(result.data(), {result_row_size, result_col_size},
                       torch::TensorOptions().dtype(torch::kInt32))
          .to(torch::kInt64);
  return torch_result;
}