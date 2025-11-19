#include "gpu_brute_force.h"
#include "datatypes.h"
#include "cuda_util.h"


/**
* Unsupported matrix type
* iT: index type
* vT: value type
* sT: state type
* MatrixType: matrix type 
*/
template<typename iT, typename vT, typename sT, typename MatrixType>
std::vector<std::vector<sT>> GPUQUBOBruteForcer<iT, vT, sT, MatrixType>::brute_force_optima(MatrixType const & mat) {
	throw std::runtime_error("GPUQUBOBruteForcer not implemented for this matrix type.");
    return std::vector<std::vector<sT>>();
}


/**
* Partial specialization for dense matrices
* iT: index type
* vT: value type
* sT: state type
*/
template<typename iT, typename vT, typename sT>
struct GPUQUBOBruteForcer<iT, vT, sT, DenseMatrix<vT>> : public QUBOBruteForcer<iT, vT, sT, DenseMatrix<vT>>
{
	/// INPUT: CPU dense matrix
	/// OUTPUT: CPU optimal state vectors
    /// Processing shall be done on the GPU
	std::vector<std::vector<sT>> brute_force_optima(DenseMatrix<vT> const & mat) override {
		/****************************
		*
		* IMPLEMENTATION GOES HERE
		*
		****************************/
		return std::vector<std::vector<sT>>();
	}
};



/**
* Partial specialization for sparse matrices
* iT: index type
* vT: value type
* sT: state type
*/
template<typename iT, typename vT, typename sT>
struct GPUQUBOBruteForcer<iT, vT, sT, SparseMatrix<vT, iT>> : public QUBOBruteForcer<iT, vT, sT, SparseMatrix<vT, iT>>
{
    /// INPUT: CPU sparse matrix
    /// OUTPUT: CPU optimal state vectors
    /// Processing shall be done on the GPU
	std::vector<std::vector<sT>> brute_force_optima(SparseMatrix<vT, iT> const & mat) override {
		/****************************
        *
		* IMPLEMENTATION GOES HERE
        *
		****************************/
		return std::vector<std::vector<sT>>();
	}
};


template class GPUQUBOBruteForcer<IndexType, ValueType, StateType, DenseMatrix<ValueType>>;
template class GPUQUBOBruteForcer<IndexType, ValueType, StateType, SparseMatrix<ValueType, IndexType>>;