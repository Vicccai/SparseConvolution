#include "src/dense_convolution.cpp"
#include "src/sparse_convolution.cpp"
#include "src/vcf_input.cpp"
#include <chrono>
#include <torch/torch.h>
using fileinput::FileInput;
using namespace std::chrono;
namespace F = torch::nn::functional;

namespace test {

torch::Tensor test_dense_conv() {
  FileInput input = FileInput();
  vector<vector<int>> data = input.VcfToDenseVector("../data/cleaned_test.vcf");
  vector<vector<int>> weight(1000, vector<int>(2, 2));
  auto start = steady_clock::now();
  vector<int> result = dense_convolution(data, weight);
  auto end = steady_clock::now();
  std::cout << "Time for naive dense: "
            << duration_cast<milliseconds>(end - start).count() << std::endl;
  torch::Tensor torch_result =
      torch::from_blob(result.data(), {62664, 49},
                       torch::TensorOptions().dtype(torch::kInt32))
          .to(torch::kInt64);
  return torch_result;
}

torch::Tensor test_dense_tensor() {
  FileInput input = FileInput();
  torch::Tensor data = input.VcfToDenseTensor("../data/cleaned_test.vcf");
  torch::Tensor data_reshape =
      data.reshape({1, 1, data.sizes().at(0), data.sizes().at(1)});
  // torch::Tensor weight = torch::tensor({1, 2, 3, 4});
  torch::Tensor weight = 2 * torch::ones({1000, 2});
  torch::Tensor weight_reshaped =
      weight.reshape({1, 1, 1000, 2}).to(torch::kInt32);
  auto start = steady_clock::now();
  torch::Tensor result = F::conv2d(data_reshape, weight_reshaped);
  auto end = steady_clock::now();
  std::cout << "Time for dense: "
            << duration_cast<milliseconds>(end - start).count() << std::endl;
  return result;
}

GenotypeData test_vcf_to_individual() {
  FileInput input = FileInput();
  GenotypeData data =
      input.VcfToSparseTensorIndividuals("../data/cleaned_test.vcf");
  std::cout << data.hetero_snps.size() << std::endl;
  std::cout << data.homo_snps.size() << std::endl;
  std::cout << data.num_individuals << std::endl;
  std::cout << data.num_snps << std::endl;
  std::cout << data.hetero_snps.at(0) << std::endl;
  std::cout << data.hetero_snps.at(0).size() << std::endl;
  return data;
}

GenotypeData test_txt_to_individual() {
  FileInput input = FileInput();
  GenotypeData data = input.TxtToSparseTensor("../data/genotype_data.txt");
  std::cout << data.hetero_snps.size() << std::endl;
  std::cout << data.homo_snps.size() << std::endl;
  std::cout << data.num_individuals << std::endl;
  std::cout << data.num_snps << std::endl;
  std::cout << data.hetero_snps.at(0) << std::endl;
  std::cout << data.hetero_snps.at(0).size() << std::endl;
  return data;
}

void test() {

  static int arr[63663][50] = {};
  auto start = steady_clock::now();
  for (int i = 0; i < 63663; i++) {
    for (int j = 0; j < 50; j++) {
      arr[i][j] = 1;
    }
  }
  auto end = steady_clock::now();
  std::cout << "Time for arr: "
            << duration_cast<milliseconds>(end - start).count() << std::endl;

  static torch::Tensor tensor = torch::zeros({63663, 50});
  start = steady_clock::now();
  for (int i = 0; i < 63663; i++) {
    for (int j = 0; j < 50; j++) {
      tensor[i][j] = 1;
    }
  }
  end = steady_clock::now();
  std::cout << "Time for tensor: "
            << duration_cast<milliseconds>(end - start).count() << std::endl;
}

void test_sparse_2() {
  FileInput input = FileInput();
  GenotypeData data = input.VcfToSparseTensor("../data/cleaned_test.vcf");
  // vector<vector<int>> weight{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  vector<vector<int>> weight{{1, 2}, {3, 4}};

  auto start = steady_clock::now();
  vector<vector<int>> result =
      sparse_convolution_2(data.homo_snps, data.hetero_snps, data.num_snps,
                           data.num_individuals, weight);
  auto end = steady_clock::now();
  std::cout << "Time 2: " << duration_cast<milliseconds>(end - start).count()
            << std::endl;
}

void test_sparse_4() {
  FileInput input = FileInput();
  GenotypeData data = input.VcfToSparseTensor("../data/cleaned_test.vcf");
  // vector<vector<int>> weight{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  vector<int> weight{1, 2, 3, 4};

  auto start = steady_clock::now();
  vector<vector<int>> result =
      sparse_convolution_4(data.homo_snps, data.hetero_snps, data.num_snps,
                           data.num_individuals, weight);
  auto end = steady_clock::now();
  std::cout << "Time 4: " << duration_cast<milliseconds>(end - start).count()
            << std::endl;
}

void test_sparse_5() {
  FileInput input = FileInput();
  GenotypeData data = input.VcfToSparseTensor("../data/cleaned_test.vcf");
  // vector<vector<int>> weight{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  vector<int> weight{1, 2, 3, 4};

  auto start = steady_clock::now();
  vector<int> result =
      sparse_convolution_5(data.homo_snps, data.hetero_snps, data.num_snps,
                           data.num_individuals, weight);
  auto end = steady_clock::now();
  std::cout << "Time 5: " << duration_cast<milliseconds>(end - start).count()
            << std::endl;
}

void test_sparse_6() {
  FileInput input = FileInput();
  GenotypeData data = input.VcfToSparseTensor("../data/cleaned_test.vcf");
  vector<vector<int>> weight{{1, 2}, {3, 4}};

  auto start = steady_clock::now();
  vector<int> result =
      sparse_convolution_6(data.homo_snps, data.hetero_snps, data.num_snps,
                           data.num_individuals, weight);
  auto end = steady_clock::now();
  std::cout << "Time 6: " << duration_cast<milliseconds>(end - start).count()
            << std::endl;
}

void test_sparse_3() {
  FileInput input = FileInput();
  GenotypeData data = input.TxtToSparseTensor("../data/genotype_data.txt");
  // GenotypeData data =
  //     input.VcfToSparseTensorIndividuals("../data/cleaned_test.vcf");
  // vector<vector<int>> weight{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  vector<vector<int>> weight{{1, 2}, {3, 4}};

  auto start = steady_clock::now();
  vector<vector<int>> result3 =
      sparse_convolution_3(data.homo_snps, data.hetero_snps, data.num_snps,
                           data.num_individuals, weight);
  auto end = steady_clock::now();
  std::cout << "Time 3: " << duration_cast<milliseconds>(end - start).count()
            << std::endl;
}

vector<int> test_sparse_7() {
  FileInput input = FileInput();
  GenotypeData data = input.TxtToSparseTensor("../data/genotype_data.txt");
  // vector<vector<int>> weight{{1, 2}, {3, 4}};
  vector<vector<int>> weight(1000, vector<int>(2, 2));
  auto start = steady_clock::now();
  vector<int> result =
      sparse_convolution_7(data.homo_snps, data.hetero_snps, data.num_snps,
                           data.num_individuals, weight);
  auto end = steady_clock::now();
  std::cout << "Time 7: " << duration_cast<milliseconds>(end - start).count()
            << std::endl;
  return result;
}

vector<vector<int>> test_sparse_8() {
  FileInput input = FileInput();
  GenotypeData data = input.TxtToSparseTensor("../data/genotype_data.txt");
  // vector<vector<int>> weight{{1, 2}, {3, 4}};
  vector<vector<int>> weight(1000, vector<int>(2, 2));
  auto start = steady_clock::now();
  vector<vector<int>> result =
      sparse_convolution_8(data.homo_snps, data.hetero_snps, data.num_snps,
                           data.num_individuals, weight);
  auto end = steady_clock::now();
  std::cout << "Time 8: " << duration_cast<milliseconds>(end - start).count()
            << std::endl;
  return result;
}

void test_loop() {
  auto start = steady_clock::now();
  for (int i = 0; i < 3183150; i++) {
    for (int j = 0; j < 16; j++) {
    }
  }
  auto end = steady_clock::now();
  std::cout << "Time double for loop: "
            << duration_cast<milliseconds>(end - start).count() << std::endl;
  start = steady_clock::now();
  for (int i = 0; i < 3183150; i++) {
    for (int j = 0; j < 4; j++) {
      for (int k = 0; k < 4; k++) {
      }
    }
  }
  end = steady_clock::now();
  std::cout << "Time triple for loop: "
            << duration_cast<milliseconds>(end - start).count() << std::endl;
}

} // namespace test