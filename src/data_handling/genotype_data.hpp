#ifndef GENOTYPE_DATA
#define GENOTYPE_DATA
#include <vector>

using std::vector;

struct GenotypeData {
  int num_snps;
  int num_individuals;
  vector<vector<int>> homo_snps;
  vector<vector<int>> hetero_snps;

  GenotypeData(const int &num_snps, const int &num_individuals,
               const vector<vector<int>> &homo_snps,
               const vector<vector<int>> &hetero_snps) {
    this->num_snps = num_snps;
    this->num_individuals = num_individuals;
    this->homo_snps = homo_snps;
    this->hetero_snps = hetero_snps;
  }
};

#endif