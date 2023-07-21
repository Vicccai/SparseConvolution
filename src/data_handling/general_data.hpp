#include <vector>

using std::vector;

struct GeneralData {
  int num_rows;
  int num_cols;
  vector<vector<int>> indices;
  vector<vector<double>> values;

  GeneralData(const int &num_rows, const int &num_cols,
              const vector<vector<int>> &indices,
              const vector<vector<double>> &values) {
    this->num_rows = num_rows;
    this->num_cols = num_cols;
    this->indices = indices;
    this->values = values;
  }
};