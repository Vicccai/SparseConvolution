// #include <torch/extension.h>
#include <torch/torch.h>
#include <tuple>
#include <vector>

using std::vector;

torch::Tensor sparse_convolution(
    const vector<torch::Tensor> &homo_snps,
    const vector<torch::Tensor> &hetero_snps, const int &num_snps,
    const int &num_individuals,
    const std::tuple<int, int> &output_size = std::make_tuple(0, 0),
    const torch::Tensor &weight, const torch::Tensor &bias, const int &stride,
    const int &dilation) {

  int k = weight.sizes().at(0);
  int resultRowSize = 0;
  int resultColSize = 0;
  if (std::get<0>(output_size) == 0 && std::get<1>(output_size) == 0) {
    resultRowSize = (num_snps - ((k - 1) * dilation + 1)) / stride + 1;
    resultColSize = (num_individuals - ((k - 1) * dilation + 1)) / stride + 1;
  } else {
    resultRowSize = std::get<0>(output_size);
    resultColSize = std::get<1>(output_size);
  }

  torch::Tensor result;
  return result;
}