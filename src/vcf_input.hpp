#include "genotype_data.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <torch/torch.h>
#include <vector>

namespace fileinput {

class FileInput {

private:
  int num_individuals;
  int num_snps;
  int num_empty_fields;
  void PreProcess(const std::string &file_path);
  void GetIndividualStrings(const std::string &file_path,
                            const int &num_individuals,
                            vector<vector<int>> &homo_snps,
                            vector<vector<int>> &hetero_snps, const bool &count_snps);

public:
  /**
   * @brief Construct a new File Input object
   *
   */
  FileInput();

  /**
   * @brief Destroy the File Input object
   *
   */
  ~FileInput() = default;

  torch::Tensor VcfToDenseTensor(const std::string &file_path);

  GenotypeData VcfToSparseTensor(const std::string &file_path);
  GenotypeData VcfToSparseTensorIndividuals(const std::string &file_path);
};

} // namespace fileinput