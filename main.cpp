// #include "src/testing/benchmark.hpp"
#include "src/testing/test.hpp"
#include <chrono>
#include <iostream>
#include <string>
#include <torch/torch.h>
#include <vector>

using namespace std::chrono;
using std::vector;
namespace F = torch::nn::functional;
using namespace torch::indexing;

int main() {
  // benchmark::benchmark_general();
  test::test_optimized_sparse_same(
      std::make_tuple(20, 2), std::make_tuple(1, 1), std::make_tuple(1, 1));
  // benchmark::benchmark_sparse();
  // torch::Tensor test = torch::ones({3, 4}).to(torch::kInt32);
  // std::vector<int> v(test.data_ptr<int>(), test.data_ptr<int>() +
  // test.numel()); std::cout << v << std::endl;
  // test::test_same(std::make_tuple(100, 2), 1, 1);
  // benchmark::benchmark_density();
  // test::test_stride_dilation(50);
  // test::test_same(1000, 5, 2);
  return 0;
}