#include "test.cpp"
#include <chrono>
#include <iostream>
#include <torch/torch.h>
#include <vector>

using namespace torch::indexing;
using namespace std::chrono;
using std::vector;
namespace F = torch::nn::functional;

int main() {
  // GenotypeData data1 = test::test_txt_to_individual();
  // GenotypeData data2 = test::test_vcf_to_individual();
  // for (int i = 0; i < data1.num_individuals; i++) {
  //   vector<int> data1hetero = data1.hetero_snps.at(i);
  //   vector<int> data2hetero = data2.hetero_snps.at(i);
  //   for (int j = 0; j < data1hetero.size(); j++) {
  //     if (data1hetero.at(j) != data2hetero.at(j)) {
  //       std::cout << "hetero " << i << j << std::endl;
  //     }
  //   }

  //   vector<int> data1homo = data1.homo_snps.at(i);
  //   vector<int> data2homo = data2.homo_snps.at(i);
  //   for (int j = 0; j < data1homo.size(); j++) {
  //     if (data1homo.at(j) != data2homo.at(j)) {
  //       std::cout << "homo " << i << j << std::endl;
  //     }
  //   }

  //   // if (data1.hetero_snps.at(i).size() != data2.hetero_snps.at(i).size())
  //   {
  //   //   std::cout << "hetero " << i << std::endl;
  //   // }
  //   // if (data1.homo_snps.at(i).size() != data2.homo_snps.at(i).size()) {
  //   //   std::cout << "homo " << i << std::endl;
  //   // }
  // }
  // std::cout << b << std::endl;
  // test::test_dense_tensor();
  // test::test_sparse_2();
  // test::test_sparse_3();
  // test::test_sparse_4();
  // test::test_sparse_5();
  // test::test_sparse_6();
  test::test_sparse_7();
  test::test_sparse_8();
  // test_loop();
  // torch::Tensor dense_result = test_dense_tensor()[0][0];
  // torch::Tensor sparse_result = test_sparse_tensor();
  // std::cout << dense_result.sizes() << std::endl;
  // std::cout << sparse_result.sizes() << std::endl;
  // // torch::Tensor test_one = torch::tensor({1, 2, 3, 4});
  // // torch::Tensor test_two = torch::tensor({1.0, 2.0, 3.0, 4.0});
  // std::cout << torch::equal(dense_result, sparse_result) << std::endl;

  return 0;
}
