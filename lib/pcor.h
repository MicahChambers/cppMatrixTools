#include <Eigen/Dense>

namespace npl
{

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * @brief Performs shooting estimate of partial correlation matrix from a
 * correlation matrix.
 *
 * @param corr Correlation matrix
 *
 * @return Estimated Partial Correlation Matrix
 */
MatrixXd shootingEstimate(const MatrixXd& corr);

};
