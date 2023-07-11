#include "src/compare_result.cpp"
#include "test.cpp"
#include <chrono>
#include <iostream>
#include <torch/torch.h>
#include <vector>

using namespace torch::indexing;
using namespace std::chrono;
using std::vector;
namespace F = torch::nn::functional;

void test_same() {
  vector<int> sparse_result = test::test_sparse_7();
  torch::Tensor torch_result =
      torch::from_blob(sparse_result.data(), {49, 62664},
                       torch::TensorOptions().dtype(torch::kInt32))
          .to(torch::kInt64)
          .transpose(0, 1);
  vector<int> result = compare::TextToResult("../data/result.txt");
  // std::cout << result.size() << std::endl;
  torch::Tensor torch_result2 =
      torch::from_blob(result.data(), {49, 62664},
                       torch::TensorOptions().dtype(torch::kInt32))
          .to(torch::kInt64)
          .transpose(0, 1);
  std::cout << torch_result.index({Slice(0, 1)}) << std::endl;
  // torch::Tensor dense_result = test::test_dense_conv();
  std::cout << torch_result2.index({Slice(0, 1)}) << std::endl;
  std::cout << torch::equal(torch_result, torch_result2) << std::endl;

  // vector<int> result_new = test::test_sparse_7();
  // torch::Tensor t_new =
  //     torch::from_blob(result_new.data(), {49, 62664},
  //                      torch::TensorOptions().dtype(torch::kInt32))
  //         .to(torch::kInt64)
  //         .transpose(0, 1);
  // std::cout << torch::equal(t_file, t_new) << std::endl;
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

int main() {
  // test_conv2d();
  test_same();
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
