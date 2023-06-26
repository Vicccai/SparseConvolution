#include "vcf_input.hpp"

using std::ifstream;
using std::ofstream;
using std::string;
using std::stringstream;
using std::vector;

namespace fileinput {

FileInput::FileInput() {
  num_individuals = 0;
  num_snps = 0;
  num_empty_fields = 0;
}

void FileInput::PreProcess(const string &file_path) {
  ifstream vcf_file(file_path);

  if (!vcf_file.is_open()) {
    std::cout << "Could not open file: " << file_path << std::endl;
    exit(EXIT_FAILURE);
  }

  string next_line = "";

  while (getline(vcf_file, next_line)) {
    if (next_line.at(1) != '#') {
      stringstream buffer(next_line);
      string item = "";
      bool count_fields = true;
      while (getline(buffer, item, '\t')) {
        if (count_fields && item.at(0) == 'i')
          count_fields = false;
        if (!count_fields) {
          num_individuals++;
        } else {
          num_empty_fields++;
        }
      }
      break;
    }
  }
}

torch::Tensor FileInput::VcfToDenseTensor(const string &file_path) {

  PreProcess(file_path);

  ifstream vcf_file(file_path);

  if (!vcf_file.is_open()) {
    std::cout << "Could not open file: " << file_path << std::endl;
    exit(EXIT_FAILURE);
  }

  string next_line = "";
  vector<int> snps;
  int item_index = 0;

  while (getline(vcf_file, next_line)) {
    if (next_line.at(0) != '#') {
      num_snps++;
      stringstream buffer(next_line);
      string item = "";
      while (getline(buffer, item, '\t')) {
        item_index++;
        if (item_index <= num_empty_fields)
          continue;
        if (item == "1|1") {
          snps.push_back(2);
        } else if (item == "0|1" || item == "1|0") {
          snps.push_back(1);
        } else {
          snps.push_back(0);
        }
      }
    }
  }

  vcf_file.close();
  torch::Tensor genotype_matrix =
      torch::from_blob(snps.data(), {num_snps, num_individuals},
                       torch::TensorOptions().dtype(torch::kInt32))
          .to(torch::kInt64);
  return genotype_matrix;
}
} // namespace fileinput