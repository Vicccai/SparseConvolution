// #include <torch/extension.h>
#include <torch/torch.h>
#include <tuple>
#include <vector>

using std::vector;

void handle_row_1d_result_colwise(
    const int &i, const int &j, const int &kernel_rows, const int &kernel_cols,
    const int &stride, const int &dilation, const int &num_snps,
    const int &num_individuals, const int &result_row_size,
    const int &result_col_size, const vector<vector<int>> &weight,
    vector<int> &result) {
  int iToUse = std::min(i, num_snps - kernel_rows);
  int jToUse = std::min(j, num_individuals - kernel_cols);
  for (int r = i - iToUse / stride * stride; r < kernel_rows; r += stride) {
    int result_row = (i - r) / stride;
    if (result_row < 0)
      break;
    if (r % dilation != 0 || result_row >= result_row_size)
      continue;
    int row_index = r / dilation;
    for (int c = j - jToUse / stride * stride; c < kernel_cols; c += stride) {
      int result_col = (j - c) / stride;
      if (result_col < 0)
        break;
      if (c % dilation != 0 || result_col >= result_col_size)
        continue;
      result[result_col * result_row_size + result_row] +=
          weight[row_index][c / dilation];
    }
  }
}

// colwise long vector, 2d weight, 1d result
torch::Tensor sparse_convolution_input_based(
    const vector<vector<int>> &homo_snps,
    const vector<vector<int>> &hetero_snps, const int &num_snps,
    const int &num_individuals, const vector<vector<int>> &weight,
    const int &stride = 1, const int &dilation = 1, const int &bias = 0,
    const std::tuple<int, int> &output_size = std::make_tuple(0, 0)) {

  int k_row = weight.size();
  int k_col = weight.at(0).size();
  vector<vector<int>> double_weight(k_row, vector<int>(k_col, 0));
  for (int i = 0; i < k_row; i++) {
    for (int j = 0; j < k_col; j++) {
      double_weight[i][j] = weight[i][j] * 2;
    }
  }
  int result_row_size = 0;
  int result_col_size = 0;
  int kernel_rows = (k_row - 1) * dilation + 1;
  int kernel_cols = (k_col - 1) * dilation + 1;
  if (std::get<0>(output_size) == 0 && std::get<1>(output_size) == 0) {
    result_row_size = (num_snps - ((k_row - 1) * dilation + 1)) / stride + 1;
    result_col_size =
        (num_individuals - ((k_col - 1) * dilation + 1)) / stride + 1;
  } else {
    result_row_size = std::get<0>(output_size);
    result_col_size = std::get<1>(output_size);
  }
  bool stride_dilation_check =
      dilation > 1 && stride > 1 &&
      (stride % dilation == 0 || dilation % stride == 0);
  vector<int> result(result_col_size * result_row_size, bias);
  for (int j = 0; j < num_individuals; j++) {
    if (stride_dilation_check && j % std::min(dilation, stride) != 0) {
      continue;
    }
    for (int i : homo_snps.at(j)) {
      // std::cout << i << std::endl;
      if (stride_dilation_check && i % std::min(dilation, stride) != 0) {
        continue;
      }
      handle_row_1d_result_colwise(i, j, kernel_rows, kernel_cols, stride,
                                   dilation, num_snps, num_individuals,
                                   result_row_size, result_col_size,
                                   double_weight, result);
    }
    for (int i : hetero_snps.at(j)) {
      if (stride_dilation_check && i % std::min(dilation, stride) != 0) {
        continue;
      }
      handle_row_1d_result_colwise(
          i, j, kernel_rows, kernel_cols, stride, dilation, num_snps,
          num_individuals, result_row_size, result_col_size, weight, result);
    }
  }
  torch::Tensor torch_result =
      torch::from_blob(result.data(), {result_col_size, result_row_size},
                       torch::TensorOptions().dtype(torch::kInt32))
          .to(torch::kInt64)
          .transpose(0, 1);
  return torch_result;
}

int handle_element(const int &i, const int &j, const int &stride,
                   const int &dilation, const int &k_row, const int &k_col,
                   vector<int> &starting_indices_homo,
                   vector<int> &starting_indices_hetero,
                   const vector<vector<int>> &homo_snps,
                   const vector<vector<int>> &hetero_snps,
                   const vector<vector<int>> &weight,
                   const vector<vector<int>> &double_weight) {
  int next_min = (i + 1) * stride;
  int current_max = i * stride + (k_row - 1) * dilation;
  int sum = 0;
  for (int b = 0; b < k_col; b++) {
    // std::cout << "b: " << b << std::endl;
    int col = j * stride + b * dilation;
    // homozygous
    int starting_index = starting_indices_homo[b];
    bool looking_for_next = true;
    int row = homo_snps[col][starting_index];
    while (row <= current_max && starting_index < homo_snps[col].size()) {
      if (looking_for_next && row >= next_min) {
        looking_for_next = false;
        starting_indices_homo[b] = starting_index;
      }
      if ((row - i * stride) % dilation != 0) {
        row = homo_snps[col][++starting_index];
        continue;
      }
      int a = (row - i * stride) / dilation;
      sum += double_weight[b][a];
      row = homo_snps[col][++starting_index];
    }
    if (looking_for_next) {
      while (starting_index < homo_snps[col].size() &&
             homo_snps[col][starting_index] < next_min) {
        starting_index++;
      }
      starting_indices_homo[b] = starting_index;
    }
    // heterozygous
    starting_index = starting_indices_hetero[b];
    looking_for_next = true;
    row = hetero_snps[col][starting_index];
    while (row <= current_max && starting_index < hetero_snps[col].size()) {
      if (looking_for_next && row >= next_min) {
        looking_for_next = false;
        starting_indices_hetero[b] = starting_index;
      }
      if ((row - i * stride) % dilation != 0) {
        row = hetero_snps[col][++starting_index];
        continue;
      }
      int a = (row - i * stride) / dilation;
      sum += weight[b][a];
      row = hetero_snps[col][++starting_index];
    }
    if (looking_for_next) {
      while (starting_index < hetero_snps[col].size() &&
             hetero_snps[col][starting_index] < next_min) {
        starting_index++;
      }
      starting_indices_hetero[b] = starting_index;
    }
  }
  return sum;
}

// updated version, more tuned for long filters. assume weight is colwise
torch::Tensor sparse_convolution_result_based(
    const vector<vector<int>> &homo_snps,
    const vector<vector<int>> &hetero_snps, const int &num_snps,
    const int &num_individuals, const vector<vector<int>> &weight,
    const int &stride = 1, const int &dilation = 1, const int &bias = 0,
    const std::tuple<int, int> &output_size = std::make_tuple(0, 0)) {

  int k_col = weight.size();
  int k_row = weight.at(0).size(); // long way
  vector<vector<int>> double_weight(k_col, vector<int>(k_row, 0));
  for (int i = 0; i < k_col; i++) {
    for (int j = 0; j < k_row; j++) {
      double_weight[i][j] = weight[i][j] * 2;
    }
  }
  int result_row_size = 0;
  int result_col_size = 0;
  int kernel_rows = (k_row - 1) * dilation + 1;
  int kernel_cols = (k_col - 1) * dilation + 1;
  if (std::get<0>(output_size) == 0 && std::get<1>(output_size) == 0) {
    result_row_size = (num_snps - ((k_row - 1) * dilation + 1)) / stride + 1;
    result_col_size =
        (num_individuals - ((k_col - 1) * dilation + 1)) / stride + 1;
  } else {
    result_row_size = std::get<0>(output_size);
    result_col_size = std::get<1>(output_size);
  }
  vector<int> result(result_row_size * result_col_size, bias);

  // i and j are result indices, a and b are filter indices, row and col are
  // input indices
  for (int j = 0; j < result_col_size; j++) {
    vector<int> starting_indices_homo(k_row, 0);
    vector<int> starting_indices_hetero(k_row, 0);
    for (int i = 0; i < result_row_size; i++) {
      int sum = handle_element(i, j, stride, dilation, k_row, k_col,
                               starting_indices_homo, starting_indices_hetero,
                               homo_snps, hetero_snps, weight, double_weight);
      result[j * result_row_size + i] += sum;
    }
  }
  torch::Tensor torch_result =
      torch::from_blob(result.data(), {result_col_size, result_row_size},
                       torch::TensorOptions().dtype(torch::kInt32))
          .to(torch::kInt64)
          .transpose(0, 1);
  return torch_result;
}
