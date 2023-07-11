#include <fstream>
#include <iterator>
#include <string>
#include <torch/torch.h>
#include <vector>

using std::vector;

namespace compare {
void ResultToText(vector<int> result) {
  std::ofstream output_file("../data/result.txt");

  std::ostream_iterator<int> output_iterator(output_file, "\n");
  std::copy(result.begin(), result.end(), output_iterator);
}

vector<int> TextToResult(const std::string &file) {
  std::ifstream input_file(file);
  std::string line;
  std::vector<int> result;
  while (std::getline(input_file, line)) {
    result.push_back(std::stoi(line));
  }
  return result;
}

void TensorToText(torch::Tensor result) {
  result = result.contiguous();
  vector<int> v(result.data_ptr<int>(),
                result.data_ptr<int>() + result.numel());
  std::ofstream output_file("../data/dense_result.txt");

  std::ostream_iterator<int> output_iterator(output_file, "\n");
  std::copy(v.begin(), v.end(), output_iterator);
}

} // namespace compare