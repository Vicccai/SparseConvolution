#include <torch/torch.h>
#include <vector>

using std::vector;

struct GenotypeData {
  int num_snps;
  int num_individuals;
  vector<torch::Tensor> homo_snps;
  vector<torch::Tensor> hetero_snps;

  GenotypeData(const int &num_snps, const int &num_individuals,
               const vector<torch::Tensor> &homo_snps,
               const vector<torch::Tensor> &hetero_snps) {
    this->num_snps = num_snps;
    this->num_individuals = num_individuals;
    this->homo_snps = homo_snps;
    this->hetero_snps = hetero_snps;
  }
};