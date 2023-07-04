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

GenotypeData FileInput::VcfToSparseTensor(const string &file_path) {
  PreProcess(file_path);
  ifstream vcf_file(file_path);

  if (!vcf_file.is_open()) {
    std::cout << "Could not open file: " << file_path << std::endl;
    exit(EXIT_FAILURE);
  }

  string next_line = "";
  vector<vector<int>> homo_snps;
  vector<vector<int>> hetero_snps;

  while (getline(vcf_file, next_line)) {
    if (next_line.at(0) != '#') {
      num_snps++;
      stringstream buffer(next_line);
      string item = "";
      vector<int> homo_inds;
      vector<int> hetero_inds;
      int item_index = 0;
      while (getline(buffer, item, '\t')) {
        if (item_index++ < num_empty_fields)
          continue;
        if (item == "1|1") {
          homo_inds.push_back(item_index - num_empty_fields - 1);
        } else if (item == "0|1" || item == "1|0") {
          hetero_inds.push_back(item_index - num_empty_fields - 1);
        }
      }
      homo_snps.push_back(homo_inds);
      torch::Tensor hetero_inds_tensor =
          torch::from_blob(hetero_inds.data(), {(int)hetero_inds.size()},
                           torch::TensorOptions().dtype(torch::kInt32))
              .to(torch::kInt64);
      hetero_snps.push_back(hetero_inds);
    }
  }
  return GenotypeData(num_snps, num_individuals, homo_snps, hetero_snps);
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

  while (getline(vcf_file, next_line)) {
    if (next_line.at(0) != '#') {
      num_snps++;
      stringstream buffer(next_line);
      string item = "";
      int item_index = 0;
      while (getline(buffer, item, '\t')) {
        if (item_index++ < num_empty_fields)
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
                       torch::TensorOptions().dtype(torch::kInt32));
  return genotype_matrix;
}

void FileInput::GetIndividualStrings(const string &file_path,
                                     const int &individual,
                                     vector<vector<int>> &homo_snps,
                                     vector<vector<int>> &hetero_snps,
                                     const bool &count_snps) {
  ifstream vcf_file(file_path);

  if (!vcf_file.is_open()) {
    std::cout << "Could not open file: " << file_path << std::endl;
    exit(EXIT_FAILURE);
  }

  string next_line = "";

  vector<int> homo_inds;
  vector<int> hetero_inds;
  int item_index = 0;
  while (getline(vcf_file, next_line)) {
    if (next_line.at(0) != '#') {
      if (count_snps) {
        num_snps++;
      }
      stringstream buffer(next_line);
      string item = "";
      bool count_fields = true;
      int start_index = 0;
      for (int i = 0; i < num_empty_fields; i++) {
        getline(buffer, item, '\t');
        start_index += item.length() + 1;
      }
      int ind_index = start_index + 4 * individual;
      item = next_line.substr(ind_index, 3);
      if (item == "1|1") {
        homo_inds.push_back(item_index);
      } else if (item == "0|1" || item == "1|0") {
        hetero_inds.push_back(item_index);
      }
      item_index++;
    }
  }
  homo_snps.push_back(homo_inds);
  hetero_snps.push_back(hetero_inds);
}

GenotypeData FileInput::VcfToSparseTensorIndividuals(const string &file_path) {
  PreProcess(file_path);
  vector<vector<int>> homo_snps;
  vector<vector<int>> hetero_snps;
  bool count_snps = true;
  for (int i = 0; i < num_individuals; i++) {
    GetIndividualStrings(file_path, i, homo_snps, hetero_snps, count_snps);
    count_snps = false;
  }
  return GenotypeData(num_snps, num_individuals, homo_snps, hetero_snps);
}
} // namespace fileinput