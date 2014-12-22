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
 * @file matrix_decomps.cpp Basic matrix operations, the pinacle of which is a 
 * Band Lanczos Method of matrix reduction.
 *
 *****************************************************************************/

#include <list>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <iostream>

#include "matrix_decomps.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::NoChange;

using namespace std;

//#define VERYDEBUG 

namespace npl 
{

/**
 * @brief Changes the eigenvalue corresponding to the provided eigenvector 
 * by dlambda. This could be used to ADD an eigenvalue/eigenvector pair, or 
 * to REMOVE (deflate) one for further analysis.
 *
 * @param A Matrix to modify
 * @param ev Eigenvector whose eigenvalue will be shifted
 * @param dlambda Change in eigenvalue
 */
void shiftEigenValue(MatrixXd& A, VectorXd& ev, double dlambda)
{
	(void)A;
	(void)ev;
	(void)dlambda;
}

/**
 * @brief Constructor for Band Lanczos Eigen Solver 
 *
 * @param A Input matrix to find the eigenvalues of
 */
BandLanczosEigenSolver::BandLanczosEigenSolver(const MatrixXd& A)
{
	solve(A);
}

/**
 * @brief Solves the matrix A with the band lanczos method with estbase random
 * starting vectors.
 *
 * @param A Vector to eigensolve
 * @param estbase Number of starting vectors to use (number of simultaneous
 * eigenvalues to estimate).
 */
void BandLanczosEigenSolver::solve(const MatrixXd& A, size_t estbase)
{
	if(estbase == 0)
		estbase = A.rows();

	// Create Random Matrix
	MatrixXd V(A.rows(), estbase);
	V.setRandom();

	// Normalize, Orthogonalize Each Column
	for(size_t cc=0; cc<V.cols(); cc++) {
		// Orthogonalize
		for(size_t jj=0; jj<cc; jj++) {
			double vc_vj = V.col(cc).dot(V.col(jj));
			V.col(cc) -= V.col(jj)*vc_vj;
		}
		// Normalize
		V.col(cc).normalize();
	}

	solve(A, V);
}

/**
 * @brief Band Lanczos Methof for Hessian Matrices
 *
 * p initial guesses (b_1...b_p)
 * set v_k = b_k for k = 1,2,...p
 * set p_c = p
 * set I = nullset
 * for j = 1,2, ..., until convergence or p_c = 0; do
 * (3) compute ||v_j||
 *     decide if v_j should be deflated, if yes, then
 *         if j - p_c > 0, set I = I union {j-p_c}
 *         set p_c = p_c-1. If p_c = 0, set j = j-1 and STOP
 *         for k = j, j+1, ..., j+p_c-1, set v_k = v_{k+1}
 *         return to step (3)
 *     set t(j,j-p_c) = ||v_j|| and normalize v_j = v_j/t(j,j-p_c)
 *     for k = j+1, j+2, ..., j+p_c-1, set
 *         t(j,k-p_c) = v_j^*v_k and v_k = v_k - v_j t(j,k-p_c)
 *     compute v(j+p_c) = Av_j
 *     set k_0 = max{1,j-p_c}. For k = k_0, k_0+1,...,j-1, set
 *         t(k,j) = conjugate(t(j,k)) and v_{j+p_c} = v_{j+p_c}-v_k t(k,j)
 *     for k in (I union {j}) (in ascending order), set
 *         t(k,j) = v^*_k v_{j+p_c} and v_{j+p_c} = v_{j+p_c} - v_k t(k,j)
 *     for k in I, set s(j,k) = conjugate(t(k,j))
 *     set T_j^(pr) = T_j + S_j = [t(i,k)] + [s(i,k)] for (i,k=1,2,3...j)
 *     test for convergence
 * end for
 *
 * @param V input/output the initial and final vectors
 */
void BandLanczosEigenSolver::solve(const MatrixXd& A, MatrixXd& V)
{
	const double dtol = 1e-12;

	// I in text, the iterators to nonzero rows of T(d) as well as the index
	// of them in nonzero_i
	std::list<int64_t> nonzero;
	int64_t pc = V.cols(); 

	// We are going to continuously grow these as more Lanczos Vectors are
	// computed
	size_t csize = V.cols()*2;
	MatrixXd approx(csize, csize);
	V.conservativeResize(NoChange, V.cols()*2);
	Eigen::SelfAdjointEigenSolver<MatrixXd> solver;

	// V is the list of candidates
	VectorXd band(pc); // store values in the band T[jj,jj-pc] to T[jj, jj-1]
	int64_t jj=0;

	while(pc > 0) {
#ifdef VERYDEBUG
		cerr << "j=" << jj << ", pc=" << pc << endl;
#endif //VERYDEBUG
		if(jj+pc >= csize) {
			// Need to Grow
			csize *= 2;
			approx.conservativeResize(csize, csize);
			V.conservativeResize(NoChange, csize);
		}

		// (3) compute ||v_j||
		double Vjnorm = V.col(jj).norm();
#ifdef VERYDEBUG
		cerr << "Norm: " << Vjnorm << endl;
#endif //VERYDEBUG

		// decide if vj should be deflated
		if(Vjnorm < dtol) {
#ifdef VERYDEBUG
			cerr << "Deflating" << endl << V.col(jj).transpose() << endl << endl;
#endif //VERYDEBUG

			// if j-pc > 0 (switch to 0 based indexing), I = I U {j-pc}
			if(jj-pc>= 0)
				nonzero.push_back(jj-pc);

			// set pc = pc - 1
			if(--pc == 0) {
				// if pc==0 set j = j-1 and stop
				jj--;
				break;
			}

			// for k = j , ... j+pc-1, set v_k = v_{k+1}
			// return to step 3
			// Erase Vj and leave jj the same
			// NOTE THAT THIS DOESN't HAPPEN MUCH SO WE DON'T WORRY AOBUT THE
			// LINEAR TIME NECESSARY
			for(int64_t cc = jj; cc<jj+pc; cc++)
				V.col(cc) = V.col(cc+1);
			continue;
		}

		// set t_{j,j-pc} = ||V_j||
		band[0] = Vjnorm;
		if(jj-pc >= 0)
			approx(jj, jj-pc) = Vjnorm;
		
		// normalize vj = vj/t{j,j-pc}
		V.col(jj) /= Vjnorm;

		/************************************************************
		 * Orthogonalize Candidate Vectors Against Vj 
		 * and make T(j,k-pc) = V(j).V(k) for k = j+1, ... jj+pc
		 * or say T(j,k) = V(j).V(k+pc) for k = j-pc, ... jj-1
		 ************************************************************/
		// for k = j+1, j+2, ... j+pc-1
#ifdef VERYDEBUG
		cerr << "Orthogonalized Candidates: " << endl;
#endif //VERYDEBUG
		for(int64_t kk=jj+1; kk<jj+pc; kk++) {
			// set t_{j,k-pc} = v^T_j v_k
			double vj_vk = V.col(jj).dot(V.col(kk));
			band[kk-pc-(jj-pc)] = vj_vk;
			
			if(kk-pc >= 0)
				approx(jj,kk-pc) = vj_vk;

			// v_k = v_k - v_j t_{j,k-p_c}
			V.col(kk) -= V.col(jj)*vj_vk;
		}
#ifdef VERYDEBUG
		cerr << V.block(0, jj+1, V.rows(), pc-1).transpose() << endl;
#endif //VERYDEBUG

		/************************************************************
		 * Create a New Candidate Vector by transforming current
		 ***********************************************************/
		// compute v_{j+pc} = A v_j
		V.col(jj+pc) = A*V.col(jj);
#ifdef VERYDEBUG
		cerr << "\nNew Candidate: " << endl;
		cerr << V.col(jj+pc).transpose() << endl << endl;
#endif //VERYDEBUG

		/*******************************************************
		 * Fill Off Diagonals with reflection T(k,j) = 
		 *******************************************************/
		// set k_0 = max{1,j-pc} for k = k_0,k_0+1, ... j-1 set
		for(int64_t kk = std::max(0l, jj-pc); kk < jj; kk++) {

			// t_kj = t_jk
			double t_kj = band[kk-(jj-pc)];
			approx(kk, jj) = t_kj;

			// v_{j+pc} = v_{j+pc} - v_k t_{k,j}
			V.col(jj+pc) -= V.col(kk)*t_kj;
		}

		// for k in I 
		for(auto kk: nonzero) {
			// t_{k,j} = v_k v_{j+pc}
			double vk_vjpc = V.col(kk).dot(V.col(jj+pc));
			approx(kk, jj) = vk_vjpc;

			// v_{j+pc} = v_{j+pc} - v_k t_{k,j}
			V.col(jj+pc) -= V.col(kk)*vk_vjpc;
		}
		// include jj 
		{
			// t_{k,j} = v_k v_{j+pc}
			double vk_vjpc = V.col(jj).dot(V.col(jj+pc));
			approx(jj, jj) = vk_vjpc;
			
			// v_{j+pc} = v_{j+pc} - v_k t_{k,j}
			V.col(jj+pc) -= V.col(jj)*vk_vjpc;
		}

		// for k in I, set s_{j,k} = t_{k,j}
		for(auto kk: nonzero)
			approx(jj, kk) = approx(kk, jj);
#ifdef VERYDEBUG
		cerr << "\nNew Candidate (Orth): " << endl;
		cerr << V.col(jj+pc).transpose() << endl << endl;
#endif //VERYDEBUG

//		/// CHECK CONVERGENCE
//		solver.compute(approx.topLeftCorner(jj+1, jj+1));
//		
//		// Project Eigenvectors through V
//		A*V.topLeftCorner(V.rows(), jj+1)*solver.eigenvectors();
		jj++;
	}
	approx.conservativeResize(jj+1, jj+1);
	V.conservativeResize(NoChange, jj+1);
	solver.compute(approx);
	evals = solver.eigenvalues();
	evecs = V*solver.eigenvectors();

#ifdef VERYDEBUG
	cerr << "T (Similar to A) " << endl << approx << endl;
	cerr << "A projected " << endl << V.transpose()*A*V << endl;
	cerr << "EigenValues: " << endl << evals << endl << endl;
	cerr << "EigenVectors: " << endl << evecs << endl << endl;
#endif //VERYDEBUG
}

}
