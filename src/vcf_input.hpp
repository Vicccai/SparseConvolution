#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <torch/torch.h>
#include <vector>
#include "genotype_data.hpp"

namespace fileinput {

class FileInput {

private:
  int num_individuals;
  int num_snps;
  int num_empty_fields;

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

  void PreProcess(const std::string &file_path);

  torch::Tensor VcfToDenseTensor(const std::string &file_path);

  GenotypeData VcfToSparseTensor(const std::string &file_path);
};

} // namespace fileinput