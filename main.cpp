#include "src/vcf_input.cpp"
#include <iostream>
#include <torch/torch.h>
#include <vector>

using fileinput::FileInput;
using namespace torch::indexing;

int main() {
  FileInput input = FileInput();
  torch::Tensor tensor = input.VcfToDenseTensor("../data/cleaned_test.vcf");
  std::cout << tensor.sizes() << std::endl;
  std::cout << tensor.index({Slice(0, 1), Slice()}) << std::endl;
  // std::cout << tensor.index({Slice(0, 1), Slice()}) << std::endl;
  // std::cout << tensor << std::endl;
  return 0;
}