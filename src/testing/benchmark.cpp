#include "benchmark.hpp"

using std::string;

namespace benchmark {

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
      test::test_torch_dense(torch_dense_data, std::make_tuple(size, 2), stride,
                             dilation);
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
  vector<string> densities = {"001", "005", "01", "02"};
  for (string density : densities) {
    std::cout << "Density: " << density << std::endl;
    GenotypeData sparse_data =
        input.TxtToSparseTensor("../data/data_trans_" + density + ".txt");
    for (int i = 0; i < 5; i++) {
      std::cout << "Run: " << (i + 1) << std::endl;
      // sparse 1
      test::test_sparse_input_based(sparse_data, std::make_tuple(20, 2),
                                    std::make_tuple(stride, stride),
                                    std::make_tuple(dilation, dilation));
      // sparse 2
      // test::test_sparse_result_based(sparse_data, 20, stride, dilation);
    }
  }
}

void benchmark_general() {
  FileInput input = FileInput();
  torch::Tensor torch_dense_data =
      input.TxtToDenseTensor("../data/data_01.txt");
  test::test_torch_dense(torch_dense_data, std::make_tuple(7, 7), 1, 1);
  test::test_torch_dense(torch_dense_data, std::make_tuple(11, 11), 1, 1);
  test::test_torch_dense(torch_dense_data, std::make_tuple(17, 17), 1, 1);

  std::tuple stride = std::make_tuple(1, 1);
  std::tuple dilation = std::make_tuple(1, 1);
  vector<int> sizes = {7, 11, 17};
  GeneralData sparse_data = input.VcfToGeneral("../data/cleaned_test.vcf");
  for (int size : sizes) {
    std::cout << "Size: " << size << std::endl;
    for (int i = 0; i < 3; i++) {
      std::cout << "Run: " << (i + 1) << std::endl;
      test::test_general_sparse(sparse_data, std::make_tuple(size, size),
                                stride, dilation);
    }
  }
}

void benchmark_sparse() {
  FileInput input = FileInput();
  GenotypeData sparse_data =
      input.TxtToSparseTensor("../data/data_trans_01.txt");
  std::tuple stride = std::make_tuple(1, 1);
  std::tuple dilation = std::make_tuple(1, 1);

  vector<string> densities = {"001", "005", "01", "02"};
  vector<int> sizes = {50};
  for (string density : densities) {
    std::cout << "Density: " << density << std::endl;
    GenotypeData sparse_data =
        input.TxtToSparseTensor("../data/data_trans_" + density + ".txt");
    for (int size : sizes) {
      std::cout << "Size: " << size << std::endl;
      for (int i = 0; i < 3; i++) {
        std::cout << "Run: " << (i + 1) << std::endl;
        test::test_sparse_input_based(sparse_data, std::make_tuple(size, 2),
                                      stride, dilation);
        test::test_sparse_optimized(sparse_data, std::make_tuple(size, 2),
                                    stride, dilation);
      }
    }
  }
}

} // namespace benchmark