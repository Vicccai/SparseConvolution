#include "../convolutions/dense_convolution.hpp"
#include "../convolutions/general_sparse.hpp"
#include "../convolutions/sparse_convolution.hpp"
#include "../data_handling/file_input.hpp"
#include <chrono>
#include <torch/torch.h>
#include <tuple>
using fileinput::FileInput;
using namespace std::chrono;
namespace F = torch::nn::functional;
using namespace torch::indexing;

namespace test {

double get_density();

GenotypeData test_vcf_to_individual();

GenotypeData test_txt_to_individual();

torch::Tensor test_naive_dense(vector<vector<int>> data,
                               const std::tuple<int, int> &kernel_size,
                               const std::tuple<int, int> &stride,
                               const std::tuple<int, int> &dilation);

torch::Tensor test_torch_dense(torch::Tensor data,
                               const std::tuple<int, int> &kernel_size,
                               const int &stride, const int &dilation);

torch::Tensor test_sparse_input_based(GenotypeData data,
                                      const std::tuple<int, int> &kernel_size,
                                      const std::tuple<int, int> &stride,
                                      const std::tuple<int, int> &dilation);

torch::Tensor test_general_sparse(GeneralData data,
                                  const std::tuple<int, int> &kernel_size,
                                  const std::tuple<int, int> &stride,
                                  const std::tuple<int, int> &dilation);

torch::Tensor test_sparse_optimized(GenotypeData data,
                                    const std::tuple<int, int> &kernel_size,
                                    const std::tuple<int, int> &stride,
                                    const std::tuple<int, int> &dilation);

void test_same(const std::tuple<int, int> &kernel_size,
               const std::tuple<int, int> &stride,
               const std::tuple<int, int> &dilation);

void test_same(const std::tuple<int, int> &kernel_size,
               const std::tuple<int, int> &stride, const int &dilation);

void test_same(const std::tuple<int, int> &kernel_size, const int &stride,
               const std::tuple<int, int> &dilation);

void test_same(const std::tuple<int, int> &kernel_size, const int &stride,
               const int &dilation);

void test_stride_dilation(const std::tuple<int, int> &kernel_size);

void test_conv2d();

void test_general_sparse_same(const std::tuple<int, int> &kernel_size,
                              const std::tuple<int, int> &stride,
                              const std::tuple<int, int> &dilation);

void test_optimized_sparse_same(const std::tuple<int, int> &kernel_size,
                                const std::tuple<int, int> &stride,
                                const std::tuple<int, int> &dilation);
} // namespace test