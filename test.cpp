#include "src/dense_convolution.cpp"
#include "src/file_input.cpp"
#include "src/sparse_convolution.cpp"
#include <chrono>
#include <torch/torch.h>
using fileinput::FileInput;
using namespace std::chrono;
namespace F = torch::nn::functional;
using namespace torch::indexing;

namespace test {

double get_density() {
  FileInput input = FileInput();
  GenotypeData data =
      input.VcfToSparseTensorIndividuals("../data/cleaned_test.vcf");
  double count = 0;
  for (vector<int> snp : data.hetero_snps) {
    count += snp.size();
  }
  for (vector<int> snp : data.homo_snps) {
    count += snp.size();
  }
  int total = data.num_snps * data.num_individuals;
  return count / total;
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

torch::Tensor test_naive_dense(const int &kernel_length, const int &stride,
                               const int &dilation) {
  FileInput input = FileInput();
  vector<vector<int>> data = input.VcfToDenseVector("../data/cleaned_test.vcf");
  vector<vector<int>> weight(kernel_length, vector<int>(2, 2));
  auto start = steady_clock::now();
  torch::Tensor result = dense_convolution(data, weight, stride, dilation);
  auto end = steady_clock::now();
  std::cout << "Time for naive dense: "
            << duration_cast<milliseconds>(end - start).count() << std::endl;
  return result;
}

torch::Tensor test_torch_dense(const int &kernel_length, const int &stride,
                               const int &dilation) {
  FileInput input = FileInput();
  torch::Tensor data = input.VcfToDenseTensor("../data/cleaned_test.vcf");
  torch::Tensor data_reshape =
      data.reshape({1, 1, data.sizes().at(0), data.sizes().at(1)});
  torch::Tensor weight = 2 * torch::ones({kernel_length, 2});
  torch::Tensor weight_reshaped =
      weight.reshape({1, 1, kernel_length, 2}).to(torch::kInt32);
  auto start = steady_clock::now();
  torch::Tensor result =
      F::conv2d(data_reshape, weight_reshaped,
                F::Conv2dFuncOptions().stride(stride).dilation(dilation));
  auto end = steady_clock::now();
  std::cout << "Time for torch dense: "
            << duration_cast<milliseconds>(end - start).count() << std::endl;
  return result;
}

torch::Tensor test_sparse_input_based(const int &kernel_length,
                                      const int &stride, const int &dilation) {
  FileInput input = FileInput();
  GenotypeData data = input.TxtToSparseTensor("../data/genotype_data.txt");
  vector<vector<int>> weight(kernel_length, vector<int>(2, 2));
  auto start = steady_clock::now();
  torch::Tensor result = sparse_convolution_input_based(
      data.homo_snps, data.hetero_snps, data.num_snps, data.num_individuals,
      weight, stride, dilation);
  auto end = steady_clock::now();
  std::cout << "Time for Sparse 1: "
            << duration_cast<milliseconds>(end - start).count() << std::endl;
  return result;
}

torch::Tensor test_sparse_result_based(const int &kernel_length,
                                       const int &stride, const int &dilation) {
  FileInput input = FileInput();
  GenotypeData data = input.TxtToSparseTensor("../data/genotype_data.txt");
  vector<vector<int>> weight(2, vector<int>(kernel_length, 2));
  auto start = steady_clock::now();
  torch::Tensor result = sparse_convolution_result_based(
      data.homo_snps, data.hetero_snps, data.num_snps, data.num_individuals,
      weight, stride, dilation);
  auto end = steady_clock::now();
  std::cout << "Time for Sparse 2: "
            << duration_cast<milliseconds>(end - start).count() << std::endl;
  return result;
}

void test_same(const int &kernel_length, const int &stride,
               const int &dilation) {
  torch::Tensor sparse_result =
      test::test_sparse_input_based(kernel_length, stride, dilation);
  torch::Tensor sparse_result_2 =
      test::test_sparse_result_based(kernel_length, stride, dilation);
  torch::Tensor dense_result =
      test::test_naive_dense(kernel_length, stride, dilation);
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