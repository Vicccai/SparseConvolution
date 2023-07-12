#include "src/dense_convolution.cpp"
#include "src/sparse_convolution.cpp"
#include "src/vcf_input.cpp"
#include <chrono>
#include <torch/torch.h>
using fileinput::FileInput;
using namespace std::chrono;
namespace F = torch::nn::functional;
using namespace torch::indexing;

namespace test {

torch::Tensor test_naive_dense(const int &kernel_length) {
  FileInput input = FileInput();
  vector<vector<int>> data = input.VcfToDenseVector("../data/cleaned_test.vcf");
  vector<vector<int>> weight(kernel_length, vector<int>(2, 2));
  auto start = steady_clock::now();
  torch::Tensor result = dense_convolution(data, weight);
  auto end = steady_clock::now();
  std::cout << "Time for naive dense: "
            << duration_cast<milliseconds>(end - start).count() << std::endl;
  return result;
}

torch::Tensor test_torch_dense(const int &kernel_length) {
  FileInput input = FileInput();
  torch::Tensor data = input.VcfToDenseTensor("../data/cleaned_test.vcf");
  torch::Tensor data_reshape =
      data.reshape({1, 1, data.sizes().at(0), data.sizes().at(1)});
  torch::Tensor weight = 2 * torch::ones({kernel_length, 2});
  torch::Tensor weight_reshaped =
      weight.reshape({1, 1, kernel_length, 2}).to(torch::kInt32);
  auto start = steady_clock::now();
  torch::Tensor result = F::conv2d(data_reshape, weight_reshaped);
  auto end = steady_clock::now();
  std::cout << "Time for torch dense: "
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

void test_general() {
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

void test_sparse_2d() {
  FileInput input = FileInput();
  GenotypeData data = input.VcfToSparseTensor("../data/cleaned_test.vcf");
  // vector<vector<int>> weight{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  vector<vector<int>> weight{{1, 2}, {3, 4}};

  auto start = steady_clock::now();
  vector<vector<int>> result =
      sparse_convolution_2d(data.homo_snps, data.hetero_snps, data.num_snps,
                            data.num_individuals, weight);
  auto end = steady_clock::now();
  std::cout << "Time 2: " << duration_cast<milliseconds>(end - start).count()
            << std::endl;
}

void test_sparse_1d() {
  FileInput input = FileInput();
  GenotypeData data = input.VcfToSparseTensor("../data/cleaned_test.vcf");
  vector<vector<int>> weight{{1, 2}, {3, 4}};

  auto start = steady_clock::now();
  vector<int> result =
      sparse_convolution_1d(data.homo_snps, data.hetero_snps, data.num_snps,
                            data.num_individuals, weight);
  auto end = steady_clock::now();
  std::cout << "Time 6: " << duration_cast<milliseconds>(end - start).count()
            << std::endl;
}

torch::Tensor test_sparse_1d_colwise(const int &kernel_length) {
  FileInput input = FileInput();
  GenotypeData data = input.TxtToSparseTensor("../data/genotype_data.txt");
  vector<vector<int>> weight(kernel_length, vector<int>(2, 2));
  auto start = steady_clock::now();
  torch::Tensor result = sparse_convolution_1d_colwise(
      data.homo_snps, data.hetero_snps, data.num_snps, data.num_individuals,
      weight);
  auto end = steady_clock::now();
  std::cout << "Time for Sparse 1: "
            << duration_cast<milliseconds>(end - start).count() << std::endl;
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

torch::Tensor test_sparse_convolution(const int &kernel_length) {
  FileInput input = FileInput();
  GenotypeData data = input.TxtToSparseTensor("../data/genotype_data.txt");
  vector<vector<int>> weight(2, vector<int>(kernel_length, 2));
  auto start = steady_clock::now();
  torch::Tensor result =
      sparse_convolution(data.homo_snps, data.hetero_snps, data.num_snps,
                         data.num_individuals, weight);
  auto end = steady_clock::now();
  std::cout << "Time for Sparse 2: "
            << duration_cast<milliseconds>(end - start).count() << std::endl;
  return result;
}

void test_same() {
  torch::Tensor sparse_result = test::test_sparse_1d_colwise(1000);
  torch::Tensor sparse_result_2 = test::test_sparse_convolution(1000);
  torch::Tensor dense_result = test::test_naive_dense(1000);

  // vector<int> result = compare::TextToResult("../data/result.txt");
  // // std::cout << result.size() << std::endl;
  // torch::Tensor torch_result2 =
  //     torch::from_blob(result.data(), {49, 62664},
  //                      torch::TensorOptions().dtype(torch::kInt32))
  //         .to(torch::kInt64)
  //         .transpose(0, 1);
  // std::cout << torch_result.index({Slice(0, 1)}) << std::endl;
  // // torch::Tensor dense_result = test::test_dense_conv();
  // std::cout << torch_result2.index({Slice(0, 1)}) << std::endl;
  // std::cout << torch::equal(torch_result, torch_result2) << std::endl;

  // vector<int> result_new = test::test_sparse_7();
  // torch::Tensor t_new =
  //     torch::from_blob(result_new.data(), {49, 62664},
  //                      torch::TensorOptions().dtype(torch::kInt32))
  //         .to(torch::kInt64)
  //         .transpose(0, 1);
  std::cout << torch::equal(sparse_result, sparse_result_2) << std::endl;
  std::cout << torch::equal(sparse_result, dense_result) << std::endl;
}

void test_conv2d() {
  torch::globalContext().setDeterministicCuDNN(true);
  torch::globalContext().setBenchmarkCuDNN(false);
  FileInput input = FileInput();
  torch::Tensor data = input.VcfToDenseTensor("../data/cleaned_test.vcf");
  torch::Tensor data_reshape =
      data.reshape({1, 1, data.sizes().at(0), data.sizes().at(1)});
  torch::Tensor weight = 2 * torch::ones({1000, 2});
  torch::Tensor weight_reshaped =
      weight.reshape({1, 1, 1000, 2}).to(torch::kInt32);
  std::cout << data_reshape[0][0].index({Slice(0, 1)}) << std::endl;
  std::cout << weight_reshaped[0][0].index({Slice(0, 1)}) << std::endl;

  torch::Tensor result = F::conv2d(data_reshape, weight_reshaped);
  std::cout << "done" << std::endl;

  std::cout << data_reshape[0][0].index({Slice(0, 1)}) << std::endl;
  std::cout << weight_reshaped[0][0].index({Slice(0, 1)}) << std::endl;

  FileInput input2 = FileInput();
  torch::Tensor data2 = input2.VcfToDenseTensor("../data/cleaned_test.vcf");
  torch::Tensor data_reshape2 =
      data2.reshape({1, 1, data2.sizes().at(0), data2.sizes().at(1)});
  torch::Tensor weight2 = 2 * torch::ones({1000, 2});
  torch::Tensor weight_reshaped2 =
      weight2.reshape({1, 1, 1000, 2}).to(torch::kInt32);

  std::cout << data_reshape[0][0].index({Slice(0, 1)}) << std::endl;
  std::cout << weight_reshaped[0][0].index({Slice(0, 1)}) << std::endl;
  std::cout << data_reshape2[0][0].index({Slice(0, 1)}) << std::endl;
  std::cout << weight_reshaped2[0][0].index({Slice(0, 1)}) << std::endl;

  torch::Tensor result2 = F::conv2d(data_reshape2, weight_reshaped2);
  std::cout << torch::equal(result, result2) << std::endl;
  std::cout << data_reshape[0][0].index({Slice(0, 1)}) << std::endl;
  std::cout << weight_reshaped[0][0].index({Slice(0, 1)}) << std::endl;
  std::cout << data_reshape2[0][0].index({Slice(0, 1)}) << std::endl;
  std::cout << weight_reshaped2[0][0].index({Slice(0, 1)}) << std::endl;

  for (int i = 0; i < 62664; i++) {
    torch::Tensor result3 = result[0][0].index({Slice(i, i + 1)});
    torch::Tensor result4 = result2[0][0].index({Slice(i, i + 1)});
    if (!torch::equal(result3, result4)) {
      std::cout << i << std::endl;
      std::cout << result3 << std::endl;
      std::cout << result4 << std::endl;
      break;
    }
  }
}

} // namespace test