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

#ifndef CHUNKPCA_H
#define CHUNKPCA_H

#include <Eigen/Dense>
#include <list>
#include <limits>
#include "matrix_decomps.h"

namespace npl
{

//class ChunkPCA
//{
//public:
//	typedef MatrixXd MatrixType;
//	typedef VectorXd VectorType;
//
//	/**
//	 * @brief Creates the chunk PCA. Each column of the X matrices should be a 
//	 * signal in 1 dimension. Each row should be a ND sample. This effectively 
//	 * concatinates the samples by first performing PCA analysis on each 
//	 * chunk then creating an approximate signal from the reduce-dimension
//	 * signals.
//	 *
//	 * @param nsample Number of samples (rows) in each input chunk
//	 */
//	ChunkPCA();
//
//	/**
//	 * @brief Adds a chunk by immediately performing an SVD on the input
//	 * and storing the whitened result, UE.
//	 *
//	 * @param X
//	 */
//	void addChunk(const MatrixType& X)
//	{
//		if(
//		// Perform SVD
//		Eigen::JacobiSVD<MatrixType> svd(X, 
//				Eigen::ComputeThinU|Eigen::ComputeThinV);
//
//		// TODO Need to further reduce matrixV!!! set fewer columns
//		vmats.push_back(svd.matrixV());
//		sigmas.push_back(svd.singularValues());
//	};
//
//	/**
//	 * @brief Merge the current chunks into a single PCA by computing the
//	 * eigenvalues/eigenvectors of the covariance matrix. 
//	 * Given that X has S samples (Rows) and D dimensions (cols), X^TX has
//	 * D^2 dimensions, which could be massive. However for a thinV computation
//	 * E will be length min(S,D), V will be (D x min(S,D)), 
//	 * C = X^TX, X = UEV^T
//	 * C = (VE^TU^T)(UEV^T)
//	 * C = VE^2V^T
//	 * and EigenVectors of Covariance Matrix.
//	 */
//	void compute(double keepvar = 0.95, bool keepchunks = false)
//	{
//		BandLanczosHermitianEigenSolver eigen;
//		// TODO set BandLanczos eigen properties
//		// rank
//		// base size 
//		
//		eigen.solve([&](const VectorType& in, VectorType& out) 
//		{
//			out.setZero();
//			for(size_t ii=0; ii<vmats.size(); ii++) {
//				assert(out.rows() == vmats[ii].rows());
//				out += vmats[ii]*sigmas[ii].asDiagonal()*sigmas.asDiagonal()*
//						vmats[ii].transpose();
//			}
//		});
//
//		if(!keepchunks) {
//			vmats.clear();
//			signas.clear();
//		}
//
//		// Create Basis
//		double totalvar = eigen.eigenvalues().sum();
//		double var = 0;
//		size_t keepdim;
//		for(keepdim = 0; keepdim<eigen.eigenvalues.rows(); keepdim++) {
//			var += eigen.eigenvalues()[keepdim];
//			if(var > totalvar*keepvar) {
//				break;
//			}
//		}
//
//		m_proj = eigen.eigenvecotrs().leftCols(keepdim);
//	};
//
//	void project(Xj
//private:
//	std::vector<MatrixType> vmats;
//	std::vector<VectorType> sigmas;
//	MatrixType m_proj;
//}

}

#endif //CHUNKPCA_H
