#include "genotype_data.hpp"
#include "general_data.hpp"
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
  /**
   * @brief sets num_individuals, num_snps, and num_empty_fields of vcf file
   *
   * @param file_path path to vcf file
   */
  void PreProcess(const std::string &file_path);
  void GetIndividuals(const std::string &file_path, const int &num_individuals,
                      vector<vector<int>> &homo_snps,
                      vector<vector<int>> &hetero_snps, const bool &count_snps);
  void GetIndividualStrings(const std::string &file_path, const int &individual,
                            std::ofstream &genotype_data);
  /**
   * @brief sets num_individuals and num_snps of txt file
   *
   * @param file_path path to txt file that has num_individuals lines
   */
  void PreProcessTxt(const std::string &file_path);

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
  torch::Tensor TxtToDenseTensor(const std::string &file_path);

  GenotypeData VcfToSparseTensor(const std::string &file_path);
  GenotypeData VcfToSparseTensorIndividuals(const std::string &file_path);
  /**
   * @brief If using for sparse convolution, make sure txt file is column-wise.
   * In other words, each line of the txt file is an individual.
   *
   * @param file_path path to txt file containing 0s, 1s and 2s representing a
   * genotype matrix
   * @return GenotypeData
   */
  GenotypeData TxtToSparseTensor(const std::string &file_path);

  vector<vector<int>> VcfToDenseVector(const std::string &file_path);
  vector<vector<int>> TxtToDenseVector(const std::string &file_path);

  void VcfConvert(const std::string &file_path, const std::string &output_path);

  GeneralData VcfToGeneral(const std::string &file_path);
};

} // namespace fileinput