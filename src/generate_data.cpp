#include <fstream>
#include <random>
#include <vector>

using std::string;
using std::vector;

namespace generate {
vector<vector<int>> generate_data(const double &density, const int &rows,
                                  const int &cols) {
  vector<vector<int>> data(rows, vector<int>(cols, 0));
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_real_distribution<> dis(0, 1);
      double prob_populate = dis(gen);
      if (prob_populate <= density) {
        double prob_one_or_two = dis(gen);
        if (prob_one_or_two <= 0.5) {
          data[i][j] = 1;
        } else {
          data[i][j] = 2;
        }
      }
    }
  }
  return data;
}

vector<vector<int>> transpose_data(vector<vector<int>> data) {
  int rows = data.size();
  int cols = data[0].size();
  vector<vector<int>> trans(cols, vector<int>(rows, 0));
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      trans[j][i] = data[i][j];
    }
  }
  return trans;
}

void write_data_to_file(const vector<vector<int>> &data,
                        const string &output_path) {
  std::ofstream data_file(output_path);
  for (vector<int> row : data) {
    string row_string = "";
    for (int el : row) {
      row_string += std::to_string(el);
    }
    data_file << row_string << std::endl;
  }
}
} // namespace generate