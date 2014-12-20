#include "pcor.h"

namespace npl {

/**
 * @brief Performs shooting estimate of partial correlation matrix from a
 * correlation matrix.
 *
 * @param corr Correlation matrix
 *
 * @return Estimated Partial Correlation Matrix
 */
VectorXd shootingEstimate(const MatrixXd& X, const VectorXd& Y)
{
	VectorXd beta(X.cols());
	
	for(size_t jj=0; jj<X.cols(); jj++) {
		double YtXj = Y.transpose()*X.col(jj);
		if((fabs(YtXj)-gamma) > 0)
			beta[jj] = sign(YtXj)*(fabs(YtXj)-gamma)/X.col(jj).normSquared();
		else
			beta[jj] = 0;
	}
	return corr;
}


}

