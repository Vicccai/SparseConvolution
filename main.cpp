#include "src/compare_result.cpp"
#include "src/generate_data.cpp"
#include "test.cpp"
#include <chrono>
#include <iostream>
#include <torch/torch.h>
#include <vector>

using namespace std::chrono;
using std::vector;
namespace F = torch::nn::functional;

void benchmark() {
  FileInput input = FileInput();

  vector<vector<int>> naive_dense_data =
      input.TxtToDenseVector("../data/data_01.txt");
  GenotypeData sparse_data =
      input.TxtToSparseTensor("../data/data_trans_01.txt");
  int stride = 1;
  int dilation = 1;
  vector<int> sizes = {2, 5, 10, 20};
  for (int size : sizes) {
    std::cout << "Size: " << size << std::endl;
    for (int i = 0; i < 5; i++) {
      std::cout << "Run: " << (i + 1) << std::endl;
      // naive dense
      // test::test_naive_dense(naive_dense_data, size, stride, dilation);
      // torch dense
      torch::Tensor torch_dense_data =
          input.TxtToDenseTensor("../data/data_01.txt");
      test::test_torch_dense(torch_dense_data, size, stride, dilation);
      // sparse 1
      // test::test_sparse_input_based(sparse_data, size, stride, dilation);
      // sparse 2
      // test::test_sparse_result_based(sparse_data, size, stride, dilation);
      std::this_thread::sleep_for(std::chrono::seconds(60));
    }
  }
}

void benchmark_density() {
  int stride = 1;
  int dilation = 1;
  FileInput input = FileInput();
  vector<string> densities = {"001"};
  for (string density : densities) {
    std::cout << "Density: " << density << std::endl;
    for (int i = 0; i < 5; i++) {
      std::cout << "Run: " << (i + 1) << std::endl;
      // sparse 1
      GenotypeData sparse_data =
          input.TxtToSparseTensor("../data/data_trans_" + density + ".txt");
      test::test_sparse_input_based(sparse_data, 20, stride, dilation);
      // sparse 2
      test::test_sparse_result_based(sparse_data, 20, stride, dilation);
    }
  }
}

void generate_density_data() {
  vector<double> densities = {0.01, 0.05, 0.1, 0.2};
  vector<string> density_strings = {"0.01", "0.05", "0.1", "0.2"};
  for (int i = 0; i < densities.size(); i++) {
    vector<vector<int>> data = generate::generate_data(densities[i], 63663, 50);
    string output_path = "../data/data_" + density_strings[i] + ".txt";
    generate::write_data_to_file(data, output_path);
    vector<vector<int>> data_trans = generate::transpose_data(data);
    output_path = "../data/data_trans_" + density_strings[i] + ".txt";
    generate::write_data_to_file(data_trans, output_path);
  }
}

int main() {
  // FileInput input = FileInput();
  // torch::Tensor torch_dense_data =
  //     input.TxtToDenseTensor("../data/data_01.txt");
  // test::test_torch_dense(torch_dense_data, 50, 1, 1);
  // generate_density_data();
  // std::cout << test::get_density() << std::endl;
  // torch::Tensor test = torch::ones({3, 4}).to(torch::kInt32);
  // std::vector<int> v(test.data_ptr<int>(), test.data_ptr<int>() +
  // test.numel()); std::cout << v << std::endl;
  benchmark_density();
  // test::test_torch_dense(2000, 1, 1);
  // test::test_conv2d();
  // test::test_same(1000, 5, 2);
  // test::test_same(100, 1, 1);
  // test::test_same(100, 1, 2);
  // test::test_same(100, 150, 1);

  return 0;
}
