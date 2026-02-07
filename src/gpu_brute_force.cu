#include "gpu_brute_force.h"
#include "datatypes.h"
#include "cuda_util.h"
#include "state_vector.h"
#include <cmath>
#include <vector>
#include <cstdint>

// Standardize mapping to avoid fragile indexing
#define BIT_TO_ROW(bit, n) ((n) - 1 - (bit))

typedef uint64_t state_t;


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


template <typename iT, typename vT>
__global__ void compute_fixed_energies_kernel(
    const vT* values, const iT* columns, const iT* offsets,
    size_t n, size_t m, vT* initial_energies
) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t num_tasks = 1ULL << m;
    if (tid >= num_tasks) return;

    vT energy = 0.0;
    // Iterate through fixed bits (0 to m-1)
    for (int i = 0; i < m; i++) {
        if ((tid >> i) & 1ULL) {
            int row = BIT_TO_ROW(i, n);
            for (iT k = offsets[row]; k < offsets[row + 1]; k++) {
                int col_idx = BIT_TO_ROW(columns[k], n);
                // Interaction only counts if neighbor is also fixed and active
                if (col_idx < m && (tid >> col_idx) & 1ULL) {
                    // Upper Triangular: Only add if current bit index <= neighbor index
                    if (i <= col_idx) energy += values[k];
                }
            }
        }
    }
    initial_energies[tid] = energy;
}

template <typename iT, typename vT>
__global__ void sparse_qubo_kernel(
    const vT* values, const iT* columns, const iT* offsets, 
    size_t n, size_t m, size_t n_sub, 
    state_t* output_states, vT* output_energies,
    const vT* initial_energies 
) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (1ULL << m)) return;

    state_t current_state = tid; 
    vT current_energy = initial_energies[tid];
    
    vT min_energy = current_energy;
    state_t best_state = current_state;

    uint64_t limit = 1ULL << n_sub;
    for (uint64_t k = 1; k < limit; k++) {
        int local_flip = __ffsll(k) - 1; 
        int flip_bit = local_flip + m; 
        int row = BIT_TO_ROW(flip_bit, n);

        // Sign is +1 if 0->1, -1 if 1->0
        vT sign = ((current_state >> flip_bit) & 1ULL) ? -1.0 : 1.0;
        vT running_sum = 0.0;

        // Traverse row for interactions
        for (iT idx = offsets[row]; idx < offsets[row + 1]; idx++) {
            int col = columns[idx];
            int col_bit = n - 1 - col;
            
            // Diagonal/Linear or Active Neighbor
            if (col == row || (current_state >> col_bit) & 1ULL) {
                running_sum += values[idx];
            }
        }

        current_energy += sign * running_sum;
        current_state ^= (1ULL << flip_bit);

        if (current_energy < min_energy) {
            min_energy = current_energy;
            best_state = current_state;
        }
    }
    output_states[tid] = best_state;
    output_energies[tid] = min_energy;
}

template<typename iT, typename vT, typename sT>
struct GPUQUBOBruteForcer<iT, vT, sT, SparseMatrix<vT, iT>> : public QUBOBruteForcer<iT, vT, sT, SparseMatrix<vT, iT>>
{
    std::vector<std::vector<sT>> brute_force_optima(SparseMatrix<vT, iT> const & mat) override {
        size_t n = mat.rows;

        // 1. Logic for Subproblems (Capped at 2^20 tasks to avoid VRAM bloat)
        size_t m = 0;
        if (n > 14) m = n - 14;
        if (m > 24) m = 24;
        if (m < 0) m = 0;
        size_t remaining = n - m;
        
        uint64_t num_tasks = 1ULL << m;

        // 2. Allocate CSR Matrix on Device
        vT *d_values; iT *d_columns, *d_offsets;
        CUDA_CALL(cudaMalloc(&d_values, mat.nnz * sizeof(vT)));
        CUDA_CALL(cudaMalloc(&d_columns, mat.nnz * sizeof(iT)));
        CUDA_CALL(cudaMalloc(&d_offsets, (n + 1) * sizeof(iT)));
        CUDA_CALL(cudaMemcpy(d_values, mat.values, mat.nnz * sizeof(vT), cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(d_columns, mat.columns, mat.nnz * sizeof(iT), cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(d_offsets, mat.offsets, (n + 1) * sizeof(iT), cudaMemcpyHostToDevice));

        // 3. Allocate Results & Initial Energies (No more Dense SpMM matrices)
        state_t *d_packed_states; vT *d_output_energies, *d_initial_energies;
        CUDA_CALL(cudaMalloc(&d_packed_states, num_tasks * sizeof(state_t)));
        CUDA_CALL(cudaMalloc(&d_output_energies, num_tasks * sizeof(vT)));
        CUDA_CALL(cudaMalloc(&d_initial_energies, num_tasks * sizeof(vT)));

        int blockSize = 256;
        uint64_t numBlocks = (num_tasks + blockSize - 1) / blockSize;

        // 4. Step 1: Analytical Initialization
        compute_fixed_energies_kernel<<<numBlocks, blockSize>>>(d_values, d_columns, d_offsets, n, m, d_initial_energies);
        
        // 5. Step 2: Main Search
        sparse_qubo_kernel<<<numBlocks, blockSize>>>(
            d_values, d_columns, d_offsets, n, m, remaining,
            d_packed_states, d_output_energies, d_initial_energies
        );
        CUDA_CALL(cudaDeviceSynchronize());

        // 6. Reduce on Host (Replace with Hierarchical Reduction for N > 40)
        std::vector<state_t> host_states(num_tasks);
        std::vector<vT> host_energies(num_tasks);
        CUDA_CALL(cudaMemcpy(host_states.data(), d_packed_states, num_tasks * sizeof(state_t), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpy(host_energies.data(), d_output_energies, num_tasks * sizeof(vT), cudaMemcpyDeviceToHost));

        vT global_min = std::numeric_limits<vT>::max();
        for (vT e : host_energies) if (e < global_min) global_min = e;

        std::vector<std::vector<sT>> results;
        for (size_t i = 0; i < num_tasks; i++) {
            if (std::abs(host_energies[i] - global_min) < 1e-5) {
                results.push_back(binary_representation_to_state_vector<sT>(host_states[i], n));
            }
        }

        // Cleanup
        CUDA_CALL(cudaFree(d_values)); CUDA_CALL(cudaFree(d_columns)); CUDA_CALL(cudaFree(d_offsets));
        CUDA_CALL(cudaFree(d_packed_states)); CUDA_CALL(cudaFree(d_output_energies)); CUDA_CALL(cudaFree(d_initial_energies));

        return results;
    }
};

template class GPUQUBOBruteForcer<IndexType, ValueType, StateType, DenseMatrix<ValueType>>;
template class GPUQUBOBruteForcer<IndexType, ValueType, StateType, SparseMatrix<ValueType, IndexType>>;