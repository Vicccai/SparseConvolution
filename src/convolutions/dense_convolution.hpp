#include <torch/torch.h>
#include <tuple>
#include <vector>

using std::vector;

torch::Tensor
dense_convolution(const vector<vector<int>> &data,
                  const vector<vector<int>> &weight,
                  const std::tuple<int, int> &stride = std::make_tuple(1, 1),
                  const std::tuple<int, int> &dilation = std::make_tuple(1, 1),
                  const int &bias = 0);

torch::Tensor
dense_convolution(const vector<vector<int>> &data,
                  const vector<vector<int>> &weight, const int &stride = 1,
                  const std::tuple<int, int> &dilation = std::make_tuple(1, 1),
                  const int &bias = 0);

torch::Tensor
dense_convolution(const vector<vector<int>> &data,
                  const vector<vector<int>> &weight,
                  const std::tuple<int, int> &stride = std::make_tuple(1, 1),
                  const int &dilation = 1, const int &bias = 0);

torch::Tensor dense_convolution(const vector<vector<int>> &data,
                                const vector<vector<int>> &weight,
                                const int &stride = 1, const int &dilation = 1,
                                const int &bias = 0);