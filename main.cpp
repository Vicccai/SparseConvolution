#include "src/test.cpp"
#include <chrono>
#include <iostream>
#include <torch/torch.h>
#include <vector>

using namespace std::chrono;
using std::vector;
namespace F = torch::nn::functional;

int main() {
  test::test_general_sparse_same(2, std::make_tuple(1, 1),
                                 std::make_tuple(1, 1));
  // FileInput input = FileInput();
  // torch::Tensor torch_dense_data =
  //     input.TxtToDenseTensor("../data/data_01.txt");
  // test::test_torch_dense(torch_dense_data, 50, 1, 1);
  // generate_density_data();
  // std::cout << test::get_density() << std::endl;
  // torch::Tensor test = torch::ones({3, 4}).to(torch::kInt32);
  // std::vector<int> v(test.data_ptr<int>(), test.data_ptr<int>() +
  // test.numel()); std::cout << v << std::endl;
  // test::test_same(100, 1, 1);
  // benchmark::benchmark_density();
  // test::test_stride_dilation(50);
  // test::test_torch_dense(2000, 1, 1);
  // test::test_conv2d();
  // test::test_same(1000, 5, 2);
  // test::test_same(100, 1, 2);
  // test::test_same(100, 150, 1);

  return 0;
}
