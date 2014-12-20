/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *	http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @file matrix_decomps.h Basic matrix operations, the pinacle of which is a 
 * Band Lanczos Method of matrix reduction.
 *
 *****************************************************************************/

#include <Eigen/Dense>

namespace npl 
{

class BandLanczosEigenSolver 
{
public:
	Eigen::VectorXd& eigenvalues() { return evals; };
	Eigen::MatrixXd& eigenvectors() { return evecs; };
	
	BandLanczosEigenSolver() {};
	BandLanczosEigenSolver(const Eigen::MatrixXd& A);

	void solve(const Eigen::MatrixXd& A, size_t estbase = 0);
	void solve(const Eigen::MatrixXd& A, Eigen::MatrixXd& V);
private:

	Eigen::VectorXd evals;
	Eigen::MatrixXd evecs;
};
}
