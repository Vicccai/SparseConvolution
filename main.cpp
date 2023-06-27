#include "src/vcf_input.cpp"
#include <iostream>
#include <torch/torch.h>
#include <vector>

using fileinput::FileInput;
using namespace torch::indexing;
namespace F = torch::nn::functional;

void test_dense_tensor() {
  FileInput input = FileInput();
  torch::Tensor data = input.VcfToDenseTensor("../data/cleaned_test.vcf");
  std::cout << data.sizes() << std::endl;
  torch::Tensor matrix =
      data.index({Slice(0, 2), Slice(0, 9)}).reshape({1, 1, 2, 9});
  std::cout << matrix << std::endl;
  std::cout << data.sizes() << std::endl;

  torch::Tensor weight = torch::tensor({1, 2, 3, 4});
  torch::Tensor weight_reshaped =
      weight.reshape({1, 1, 2, 2}).to(torch::kInt32);
  std::cout << weight_reshaped << std::endl;
  std::cout << weight_reshaped.sizes() << std::endl;
  torch::Tensor result = F::conv2d(matrix, weight_reshaped);
  std::cout << result << std::endl;
}

void test_sparse_tensor() {
  FileInput input = FileInput();
  GenotypeData data = input.VcfToSparseTensor("../data/cleaned_test.vcf");
  std::cout << data.hetero_snps.size() << std::endl;
  std::cout << data.homo_snps.size() << std::endl;
  std::cout << data.num_individuals << std::endl;
  std::cout << data.num_snps << std::endl;
  std::cout << data.hetero_snps.at(0) << std::endl;
  std::cout << data.homo_snps.at(0).size() << std::endl;
}

int main() {
  // test_dense_tensor();
  // test_sparse_tensor();
  // torch::Tensor rand = torch::zeros({2, 3}) + 2;
  std::cout << 5 / 3 * 3 << std::endl;
  // torch::Tensor test_one = torch::tensor({1, 2, 3, 4});
  // torch::Tensor test_two = torch::tensor({1.0, 2.0, 3.0, 4.0});
  // std::cout << torch::equal(test_one, test_two) << std::endl;
  // std::cout << rand[0][0] << std::endl;

  return 0;
}
