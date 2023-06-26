#include <iostream>
#include <torch/torch.h>
#include <vector>

int main() {
  torch::Tensor tensor = torch::rand({2, 3});
  std::vector<int> vect;
  vect.push_back(10);
  std::cout << vect << std::endl;
  std::cout << tensor << std::endl;
  return 0;
}