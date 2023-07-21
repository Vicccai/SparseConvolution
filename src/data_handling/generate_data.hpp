#include <fstream>
#include <random>
#include <vector>

using std::string;
using std::vector;

namespace generate {

/**
 * @brief Generate random rows-by-cols 2d vector with density density
 *
 * @param density
 * @param rows
 * @param cols
 * @return vector<vector<int>>
 */
vector<vector<int>> generate_data(const double &density, const int &rows,
                                  const int &cols);

/**
 * @brief Transposes a 2d vector
 *
 * @param data
 * @return vector<vector<int>>
 */
vector<vector<int>> transpose_data(vector<vector<int>> data);

/**
 * @brief Wrties a 2d vector to a file
 *
 * @param data
 * @param output_path
 */
void write_data_to_file(const vector<vector<int>> &data,
                        const string &output_path);
} // namespace generate