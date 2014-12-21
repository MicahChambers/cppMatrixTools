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
 * @file lanczos_test1.cpp Test the band lanczos algorithm of eigenvector
 * computation.
 *
 *****************************************************************************/

#include <iostream>
#include <vector>

#include <Eigen/Dense>

#include "matrix_decomps.h"

using namespace std;
using namespace npl;

using Eigen::VectorXd;
using Eigen::MatrixXd;

/**
 * @brief Fills a matrix with random (but symmetric, positive definite) values
 */
void createRandom(MatrixXd& tgt, size_t rank)
{
	assert(tgt.rows() == tgt.cols());
	VectorXd tmp(tgt.rows());
	tgt.setZero();

	for(size_t ii=0; ii<rank; ii++) {
		tmp.setRandom();
		tmp.normalize();
		tgt += tmp*tmp.transpose();
	}
}

int main(int argc, char** argv)
{
	// Size of Matrix to Compute Eigenvalues of 
	size_t matsize = 10;

	// Number of orthogonal vectors to start with
	size_t nbasis = 10;

	// Rank of matrix to construct
	size_t nrank = 10;
	if(argc == 2) {
		matsize = atoi(argv[1]);
	} else if(argc == 3) {
		matsize = atoi(argv[1]);
		nbasis = atoi(argv[2]);
	} else if(argc == 4) {
		matsize = atoi(argv[1]);
		nbasis = atoi(argv[2]);
		nrank = atoi(argv[3]);
	} else {
		cerr << "Using default matsize, nbasis, rank (set with: " << argv[0] 
			<< " [matsize] [nbasis] [rank]" << endl;
	}

	MatrixXd A(matsize, matsize);
	createRandom(A, nrank);
	cerr << "A: " << endl << A << endl << endl;

	Eigen::SelfAdjointEigenSolver<MatrixXd> solver(A);
	cerr << "Eigen's Solution: " << endl << solver.eigenvectors() << endl 
		<< endl << solver.eigenvalues() << endl;

	MatrixXd evs = solver.eigenvectors();

	BandLanczosEigenSolver blsolver;
	blsolver.solve(A, evs);

	cerr << "My Solution: " << endl << blsolver.eigenvectors() << endl 
		<< endl << blsolver.eigenvalues() << endl;
}
