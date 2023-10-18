// #include "src/testing/benchmark.hpp"
// #include "src/testing/test.hpp"
#include "src/convolutions/sparse_convolution.hpp"
#include "src/data_handling/file_input.hpp"
// #include "src/data_handling/generate_data.hpp"
#include <chrono>
#include <iostream>
#include <string>
#include <torch/torch.h>
#include <vector>

using namespace std::chrono;
using std::vector;
namespace F = torch::nn::functional;
using namespace torch::indexing;

void test_new() {
  int batch_size = 10;
  int out_channels = 4;
  int in_channels = 1;
  int result_row_size = 5000;
  int result_col_size = 50;
  auto start = steady_clock::now();
  vector<double> result(batch_size * out_channels * result_row_size *
                        result_col_size);
  int unit = result_col_size * result_row_size;
  int start_index = 0;
  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < out_channels; j++) {
      for (int k = 0; k < in_channels; k++) {
        // result[start_index] += 0;
      }
      start_index += unit;
    }
  }
  torch::Tensor torch_result =
      torch::from_blob(
          result.data(),
          {batch_size, out_channels, result_col_size, result_row_size},
          torch::TensorOptions().dtype(torch::kFloat64))
          .to(torch::kFloat32)
          .clone()
          .transpose(2, 3);
  auto end = steady_clock::now();
  std::cout << "Time if optimized: "
            << duration_cast<milliseconds>(end - start).count() << std::endl;
}

void test_old() {
  int batch_size = 10;
  int out_channels = 4;
  int in_channels = 1;
  int result_row_size = 5000;
  int result_col_size = 50;
  auto start = steady_clock::now();
  torch::Tensor result = torch::zeros(
      {batch_size, out_channels, result_row_size, result_col_size});
  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < out_channels; j++) {
      for (int k = 0; k < in_channels; k++) {
        vector<double> result_vect(result_col_size * result_row_size, 0);
        torch::Tensor torch_result =
            torch::from_blob(result_vect.data(),
                             {result_col_size, result_row_size},
                             torch::TensorOptions().dtype(torch::kFloat64))
                .to(torch::kFloat32)
                .clone()
                .transpose(0, 1);
        // result[i][j] += torch_result;
      }
    }
  }
  auto end = steady_clock::now();
  std::cout << "Time with current: "
            << duration_cast<milliseconds>(end - start).count() << std::endl;
}

int main() {
  int output_channels = 1;
  fileinput::FileInput input = fileinput::FileInput();
  GenotypeData sparse = input.TxtToSparseTensor("../data/data_trans_01.txt");
  vector<vector<GenotypeData>> sparse_input(1, vector<GenotypeData>(1, sparse));
  std::cout << sparse.num_individuals << " " << sparse.num_snps << std::endl;
  vector<vector<vector<vector<double>>>> weight(
      output_channels,
      vector<vector<vector<double>>>(
          1, vector<vector<double>>(2, vector<double>(20, 2))));
  vector<double> bias(output_channels, 0);

  // vector<int> block_sizes{8000};
  auto start = steady_clock::now();
  torch::Tensor result_blocked = sparse_convolution_blocked(
      sparse_input, weight, bias, std::make_tuple(1, 1), std::make_tuple(1, 1),
      0);
  auto end = steady_clock::now();
  std::cout << result_blocked.sizes() << std::endl;
  std::cout << "Time for Blocked Sparse: "
            << duration_cast<milliseconds>(end - start).count() << std::endl;
  // std::cout << result[0][0][0][0] << std::endl;
  // std::cout << result_blocked[0][0][0][0] << std::endl;

  start = steady_clock::now();
  torch::Tensor result = sparse_convolution(
      sparse_input, weight, bias, std::make_tuple(1, 1), std::make_tuple(1, 1));
  end = steady_clock::now();
  std::cout << result.sizes() << std::endl;
  std::cout << "Time for Sparse: "
            << duration_cast<milliseconds>(end - start).count() << std::endl;
  std::cout << torch::equal(result, result_blocked) << std::endl;

  // benchmark::benchmark_general();
  // benchmark::benchmark_sparse();
  // torch::Tensor test = torch::ones({3, 4}).to(torch::kInt32);
  // std::vector<int> v(test.data_ptr<int>(), test.data_ptr<int>() +
  // test.numel()); std::cout << v << std::endl;
  // test::test_same(std::make_tuple(100, 2), 1, 1);
  // test::test_optimized_sparse_same(
  //     std::make_tuple(20, 2), std::make_tuple(1, 1), std::make_tuple(1,
  //     1));
  // benchmark::benchmark_density();
  // test::test_stride_dilation(50);
  // test::test_same(1000, 5, 2);
  return 0;
}