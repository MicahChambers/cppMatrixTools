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
#include <list>
#include <limits>

namespace npl 
{

class BandLanczosEigenSolver 
{
public:
	typedef Eigen::MatrixXd MatrixType;
	typedef Eigen::VectorXd VectorType;

	/**
	 * @brief Return vector of eigenvalues
	 *
	 * @return Eigenvalues as vector
	 */
	const VectorType& eigenvalues() { return m_evals; };
	
	/**
	 * @brief Return Matrix of eigenvectors (columns correspond to values in)
	 * matching row of eigenvalues().
	 *
	 * @return Eigenvectors as matrix, 1 vector per row
	 */
	const MatrixType& eigenvectors() { return m_evecs; };
	
	/**
	 * @brief Basic constructor 
	 */
	BandLanczosEigenSolver() : m_initbase(10), m_rank(UINT_MAX) {};
	
	/**
	 * @brief Constructor that solves A
	 */
	BandLanczosEigenSolver(const MatrixType& A) 
		: m_initbase(10), m_rank(UINT_MAX)
	{
		solve(A);
	};
	
	/**
	 * @brief Constructor that solves A with the initial projection (Krylov 
	 * basis vectors) set to the columns of V
	 */
	BandLanczosEigenSolver(const MatrixType& A, const MatrixType& V) 
		: m_initbase(10), m_rank(UINT_MAX)
	{
		m_proj = V;
		_solve(A);
	};

	/**
	 * @brief Solves A with the initial projection (Krylov 
	 * basis vectors) set to the random vectors of dimension set by
	 * setRandomBasis()
	 *
	 * @param A matrix to solve the eigensystem of
	 */
	void solve(const MatrixType& A)
	{
		if(m_initbase <= 0) {
			throw "Invalid initial basis size";
		}

		// Create Random Matrix
		m_proj.resize(A.rows(), m_initbase);
		m_proj.setRandom();

		// Normalize, Orthogonalize Each Column
		for(size_t cc=0; cc<m_proj.cols(); cc++) {
			// Orthogonalize
			for(size_t jj=0; jj<cc; jj++) {
				double vc_vj = m_proj.col(cc).dot(m_proj.col(jj));
				m_proj.col(cc) -= m_proj.col(jj)*vc_vj;
			}

			// Normalize
			m_proj.col(cc).normalize();
		}

		_solve(A);
	};

	/**
	 * @brief Solves A with the initial projection (Krylov 
	 * basis vectors) set to the random vectors of dimension set by
	 * setRandomBasis()
	 *
	 * @param A matrix to solve the eigensystem of
	 */
	void solve(const MatrixType& A, const MatrixType& V)
	{
		m_proj = V;
		_solve(A);
	};

	/**
	 * @brief Set dimensionality of random basis during optimization. The 
	 * effect of this is difficult to know. This should be the size of the
	 * largest cluster (closely spaced set) of eigenvalues. 10 is the default.
	 *
	 * @param Number of basis vectors to use initially
	 */
	void setRandomBasisSize(size_t nvec)
	{
		m_initbase = nvec;
	};

	/**
	 * @brief Set a hard limit on the number of eigenvectors to compute
	 *
	 * @param rank of eigenvector output
	 */
	void setRank(size_t rank)
	{
		m_rank = rank;
	};
private:

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
	 * @param A Matrix to decompose
	 */
	void _solve(const MatrixType& A)
	{
		const double dtol = sqrt(std::numeric_limits<double>::epsilon());
		MatrixType& V = m_proj;

		// I in text, the iterators to nonzero rows of T(d) as well as the index
		// of them in nonzero_i
		std::list<int64_t> nonzero;
		int64_t pc = V.cols(); 

		// We are going to continuously grow these as more Lanczos Vectors are
		// computed
		size_t csize = V.cols()*2;
		MatrixType approx(csize, csize);
		V.conservativeResize(Eigen::NoChange, V.cols()*2);

		// V is the list of candidates
		VectorType band(pc); // store values in the band T[jj,jj-pc] to T[jj, jj-1]
		int64_t jj=0;

		while(pc > 0 && jj < m_rank) {
#ifdef VERYDEBUG
			std::cerr << "j=" << jj << ", pc=" << pc << std::endl;
#endif //VERYDEBUG
			if(jj+pc >= csize) {
				// Need to Grow
				csize *= 2;
				approx.conservativeResize(csize, csize);
				V.conservativeResize(Eigen::NoChange, csize);
			}

			// (3) compute ||v_j||
			double Vjnorm = V.col(jj).norm();
#ifdef VERYDEBUG
			std::cerr << "Norm: " << Vjnorm << std::endl;
#endif //VERYDEBUG

			// decide if vj should be deflated
			if(Vjnorm < dtol) {
#ifdef VERYDEBUG
				std::cerr << "Deflating" << std::endl << V.col(jj).transpose() << std::endl << std::endl;
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
			std::cerr << "Orthogonalized Candidates: " << std::endl;
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
			std::cerr << V.block(0, jj+1, V.rows(), pc-1).transpose() << std::endl;
#endif //VERYDEBUG

			/************************************************************
			 * Create a New Candidate Vector by transforming current
			 ***********************************************************/
			// compute v_{j+pc} = A v_j
			V.col(jj+pc) = A*V.col(jj);
#ifdef VERYDEBUG
			std::cerr << "\nNew Candidate: " << std::endl;
			std::cerr << V.col(jj+pc).transpose() << std::endl << std::endl;
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
			std::cerr << "\nNew Candidate (Orth): " << std::endl;
			std::cerr << V.col(jj+pc).transpose() << std::endl << std::endl;
#endif //VERYDEBUG

			jj++;
		}

		// Set Approximate and Projection to Final Size
		approx.conservativeResize(jj+1, jj+1);
		V.conservativeResize(Eigen::NoChange, jj+1);
		
		// Compute Eigen Solution to Similar Matrix, then project through V
		Eigen::SelfAdjointEigenSolver<MatrixType> solver(approx);
		m_evals = solver.eigenvalues();
		m_evecs = V*solver.eigenvectors();

#ifdef VERYDEBUG
		std::cerr << "T (Similar to A) " << std::endl << approx << std::endl;
		std::cerr << "A projected " << std::endl << V.transpose()*A*V << std::endl;
		std::cerr << "EigenValues: " << std::endl << m_evals << std::endl << std::endl;
		std::cerr << "EigenVectors: " << std::endl << m_evecs << std::endl << std::endl;
#endif //VERYDEBUG
	}

	size_t m_initbase;
	size_t m_rank;
	VectorType m_evals;
	MatrixType m_evecs;
	MatrixType m_proj; // Computed Projection Matrix (V)
};
}
