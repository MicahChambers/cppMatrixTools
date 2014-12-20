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

#include "matrix_decomps.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::NoChange;

namespace npl 
{

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
	MatrixXd V(A.rows(), estbase);
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
	const double dtol = 1e-20;

	// I in text, the iterators to nonzero rows of T(d) as well as the index
	// of them in nonzero_i
	std::list<int64_t> nonzero;

	// We are going to continuously grow these as more Lanczos Vectors are
	// computed
	size_t csize = V.cols()*2;
	MatrixXd approx(csize, csize);
	V.conservativeResize(NoChange, V.cols()*2);
	Eigen::SelfAdjointEigenSolver<MatrixXd> solver;

	// V is the list of candidates
	int64_t pc = V.size(); 
	VectorXd band(pc); // store values in the band T[jj,jj-pc] to T[jj, jj-1]
	int64_t jj=0;

	while(pc > 0) {
		if(jj+pc >= csize) {
			// Need to Grow
			csize *= 2;
			approx.conservativeResize(csize, csize);
			V.conservativeResize(NoChange, csize);
		}

		// (3) compute ||v_j||
		double Vjnorm = V.row(jj).norm();

		// decide if vj should be deflated
		if(Vjnorm < dtol) {

			// if j-pc > 0 (switch to 0 based indexing), I = I U {j-pc}
			if(jj-pc>= 0)
				nonzero.push_back(jj-pc);

			// set pc = pc - 1
			if(--pc== 0) {
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
				V.row(cc) = V.row(cc+1);
			continue;
		}

		// set t_{j,j-pc} = ||V_j||
		band[0] = Vjnorm;
		if(jj-pc >= 0)
			approx(jj, jj-pc) = Vjnorm;
		
		// normalize vj = vj/t{j,j-pc}
		V.row(jj) /= Vjnorm;

		// for k = j+1, j+2, ... j+pc-1
		for(int64_t kk=jj+1; kk<jj+pc; kk++) {
			// set t_{j,k-pc} = v^T_j v_k
			double vj_vk = V.row(jj).dot(V.row(kk));
			band[kk-pc-(jj-pc)] = vj_vk;
			
			if(kk-pc >= 0)
				approx(jj,kk-pc) = vj_vk;

			// v_k = v_k - v_j t_{j,k-p_c}
			V.row(kk) -= V.row(jj)*vj_vk;
		}

		// compute v_{j+pc} = A v_j
		V.row(jj+pc) = A*V.row(jj);

		// set k_0 = max{1,j-pc} for k = k_0,k_0+1, ... j-1 set
		for(int64_t kk = std::max(0l, jj-pc); kk < jj; kk++) {

			// t_kj = t_jk
			double t_kj = band[kk-(jj-pc)];
			approx(kk, jj) = t_kj;

			// v_{j+pc} = v_{j+pc} - v_k t_{k,j}
			V.row(jj+pc) -= V.row(kk)*t_kj;
		}

		// for k in I 
		for(auto kk: nonzero) {
			// t_{k,j} = v_k v_{j+pc}
			double vk_vjpc = V.row(kk).dot(V.row(jj+pc));

			// v_{j+pc} = v_{j+pc} - v_k t_{k,j}
			V.row(jj+pc) -= V.row(kk)*vk_vjpc;
		}
		// include jj 
		{
			// t_{k,j} = v_k v_{j+pc}
			double vk_vjpc = V.row(jj).dot(V.row(jj+pc));
			
			// v_{j+pc} = v_{j+pc} - v_k t_{k,j}
			V.row(jj+pc) -= V.row(jj)*vk_vjpc;
		}

		// for k in I, set s_{j,k} = t_{k,j}
		for(auto kk: nonzero)
			approx(jj, kk) = approx(kk, jj);

		/// CHECK CONVERGENCE
		solver.compute(approx.topLeftCorner(jj+1, jj+1));
		
		// Project Eigenvectors through V
		A*V.topLeftCorner(V.rows(), jj+1)*solver.eigenvectors();
	}
	V.conservativeResize(NoChange, jj+1);

}

}
