#include "src/compare_result.cpp"
#include "test.cpp"
#include <chrono>
#include <iostream>
#include <torch/torch.h>
#include <vector>

using namespace std::chrono;
using std::vector;
namespace F = torch::nn::functional;

int main() {
  // test::test_conv2d();
  test::test_same();
  // vector<int> dense_result =
  // compare::TextToResult("../data/dense_result.txt"); std::cout <<
  // dense_result.size() << std::endl; torch::Tensor t_new =
  //     torch::from_blob(dense_result.data(), {62664, 49},
  //                      torch::TensorOptions().dtype(torch::kInt32))
  //         .to(torch::kInt64);
  // torch::Tensor result = test::test_dense_tensor()[0][0];
  // std::cout << torch::equal(t_new, result) << std::endl;
  // std::cout << t_new.index({Slice(0, 1), Slice()}) << std::endl;
  // std::cout << result.index({Slice(0, 1), Slice()}) << std::endl;
  // compare::TensorToText(result);
  // torch::Tensor matrix = result.index({Slice(0, 10), Slice(0, 1)});
  // std::cout << matrix << std::endl;
  // std::cout << result.index({Slice(0, 1)}) << std::endl;
  // std::cout << result.index({Slice(0, 1), Slice()}) << std::endl;
  // std::cout << result.sizes() << std::endl;
  // test::test_sparse_2();
  // test::test_sparse_3();
  // test::test_sparse_4();
  // test::test_sparse_5();
  // test::test_sparse_6();
  // vector<int> result3 = test::test_sparse_7();
  // std::cout << result3.size() << std::endl;
  // torch::Tensor t =
  //     torch::from_blob(result3.data(), {49, 62664},
  //                      torch::TensorOptions().dtype(torch::kInt32))
  //         .to(torch::kInt64);
  // torch::Tensor t_trans = t.transpose(0, 1);
  // std::cout << t_trans.sizes() << std::endl;
  // vector<vector<int>> result2 = test::test_sparse_8();
  // std::cout << result2.size() << std::endl;
  // std::cout << result2.at(0).size() << std::endl;
  // int equal = 0;
  // int notEqual = 0;
  // for (int i = 0; i < 62664; i++) {
  //   torch::Tensor result3 = result1.index({Slice(i, i + 1)});
  //   torch::Tensor result4 = result2.index({Slice(i, i + 1)});
  //   if (!torch::equal(result3, result4)) {
  //     std::cout << i << std::endl;
  //     std::cout << result3 << std::endl;
  //     std::cout << result4 << std::endl;
  //     break;
  //   }
  // }
  // std::cout << equal << std::endl;
  // std::cout << notEqual << std::endl;
  // torch::Tensor result1 = result.index({Slice(62663, 62664)});
  // std::cout << result1 << std::endl;
  // std::cout << torch::equal(result, t_trans) << std::endl;

  return 0;
}
