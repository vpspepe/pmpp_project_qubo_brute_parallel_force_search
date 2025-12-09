#include "gpu_brute_force.h"
#include "datatypes.h"
#include "cuda_util.h"
#include "state_vector.h"
#include <cmath>
#include <vector>

// Forward declaration of the kernel
template <typename iT, typename vT, typename sT>
__global__ void sparse_qubo_kernel(
    const vT* values, const iT* columns, const iT* offsets, 
    size_t n, size_t m, size_t n_sub, 
    sT* output_states, vT* output_energies
) {
    // 1. Thread ID represents the fixed LOWER bits (LSB)
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long num_tasks = 1ULL << m;

    if (tid >= num_tasks) return;

    // 2. Initialize State
    // Fixed bits are at the bottom (0 to m-1). Iterating bits (m to n-1) start at 0.
    // So the initial state is just the Thread ID.
    unsigned long long current_state = tid; 
    
    // 3. Compute Initial Energy (Full Calculation)
    vT current_energy = 0.0;
    
    // Standard sparse multiplication: x^T * Q * x
    // Iterate over all rows to compute full energy from scratch
    for (size_t row = 0; row < n; row++) {
        // Map row index to bit position: 
        // state_vector.h implies state_vector[0] is MSB (bit n-1).
        // To check if row 'row' is active, we check bit (n - 1 - row).
        size_t row_bit = (current_state >> (n - 1 - row)) & 1ULL;
        
        if (row_bit == 0) continue; 

        iT start = offsets[row];
        iT end = offsets[row + 1];

        for (iT k = start; k < end; k++) {
            iT col = columns[k];
            // Check if column bit is active
            size_t col_bit = (current_state >> (n - 1 - col)) & 1ULL;
            
            // For upper triangular/symmetric matrix handled by reader:
            // Add term if both bits are 1
            if (col_bit) {
                current_energy += values[k];
            }
        }
    }

    // Initialize Best Local Result
    vT min_energy = current_energy;
    unsigned long long best_state = current_state;

    // 4. Gray Code Loop
    // Iterate over the UPPER n_sub bits (from 1 to 2^n_sub - 1)
    unsigned long long limit = 1ULL << n_sub;

    for (unsigned long long k = 1; k < limit; k++) {
        // Find which bit changes in the Gray code sequence of the UPPER part
        int local_flip = __ffsll(k) - 1; 
        
        // Shift this bit index up by m, because we are flipping the MSB part
        int flip_bit = local_flip + m; 
        
        // Map this bit index to the matrix row index.
        // Bit 0 is LSB, corresponds to matrix index n-1.
        // Bit n-1 is MSB, corresponds to matrix index 0.
        size_t matrix_row_idx = n - 1 - flip_bit; 

        // 5. Calculate Delta Energy
        size_t row = matrix_row_idx;
        iT start = offsets[row];
        iT end = offsets[row + 1];

        vT energy_diff = 0.0;
        
        // Determine if we are flipping 0->1 or 1->0
        size_t current_bit_val = (current_state >> flip_bit) & 1ULL;
        // If current is 0, we flip to 1 (sign +1). If 1, flip to 0 (sign -1).
        vT sign = (current_bit_val == 0) ? 1.0 : -1.0;

        // Iterate sparse neighbors of the flipped variable
        for (iT idx = start; idx < end; idx++) {
            iT col = columns[idx];
            if (col == row) {
                // Diagonal term (Q_ii * x_i)
                energy_diff += values[idx];
            } else {
                // Interaction term (Q_ij * x_i * x_j)
                // We need the state of the neighbor x_j
                size_t col_bit = (current_state >> (n - 1 - col)) & 1ULL;
                if (col_bit) {
                    energy_diff += values[idx];
                }
            }
        }

        // Apply sign to the delta
        energy_diff *= sign;

        // 6. Update State and Energy
        current_energy += energy_diff;
        current_state ^= (1ULL << flip_bit); // Toggle the bit

        // Track Minimum
        if (current_energy < min_energy) {
            min_energy = current_energy;
            best_state = current_state;
        }
    }

    // 7. Write Output
    output_states[tid] = best_state;
    output_energies[tid] = min_energy;
}

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



/** TODO partial
* Specialization for sparse matrices
* iT: index type
* vT: value type
* sT: state type
*/
template<typename iT, typename vT, typename sT>
struct GPUQUBOBruteForcer<iT, vT, sT, SparseMatrix<vT, iT>> : public QUBOBruteForcer<iT, vT, sT, SparseMatrix<vT, iT>>
{
    std::vector<std::vector<sT>> brute_force_optima(SparseMatrix<vT, iT> const & mat) override {
        size_t n = mat.rows;

        // --- 1. Calculate M (Fixed Bits) ---
        // Your logic: "m=0.2*n if n-m < 25, otherwise m must satisfy n-m = 25"
        
        size_t m = static_cast<size_t>(n * 0.2);
        size_t remaining = n - m;

        if (remaining >= 25) {
            remaining = 25;
            m = n - 25; 
        }

        // Number of threads (tasks) = 2^m
        unsigned long long num_tasks = 1ULL << m;

        // --- 2. Allocate & Copy Matrix to Device ---
        vT *d_values;
        iT *d_columns, *d_offsets;
        
        CUDA_CALL(cudaMalloc(&d_values, mat.nnz * sizeof(vT)));
        CUDA_CALL(cudaMalloc(&d_columns, mat.nnz * sizeof(iT)));
        CUDA_CALL(cudaMalloc(&d_offsets, (mat.rows + 1) * sizeof(iT)));

        CUDA_CALL(cudaMemcpy(d_values, mat.values, mat.nnz * sizeof(vT), cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(d_columns, mat.columns, mat.nnz * sizeof(iT), cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(d_offsets, mat.offsets, (mat.rows + 1) * sizeof(iT), cudaMemcpyHostToDevice));

        // --- 3. Allocate Output Arrays ---
        // We use size_t for packed states on GPU for efficiency
        size_t *d_packed_states;
        vT *d_output_energies;

        CUDA_CALL(cudaMalloc(&d_packed_states, num_tasks * sizeof(size_t)));
        CUDA_CALL(cudaMalloc(&d_output_energies, num_tasks * sizeof(vT)));

        // --- 4. Launch Kernel ---
        int blockSize = 256; 
        unsigned long long numBlocks = (num_tasks + blockSize - 1) / blockSize;

        sparse_qubo_kernel<iT, vT, size_t><<<dim3(numBlocks), dim3(blockSize)>>>(
            d_values, d_columns, d_offsets,
            n, m, remaining,
            d_packed_states, d_output_energies
        );
        CUDA_CALL(cudaGetLastError());
        CUDA_CALL(cudaDeviceSynchronize());

        // --- 5. Retrieve & Reduce Results ---
        std::vector<size_t> host_states(num_tasks);
        std::vector<vT> host_energies(num_tasks);

        CUDA_CALL(cudaMemcpy(host_states.data(), d_packed_states, num_tasks * sizeof(size_t), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpy(host_energies.data(), d_output_energies, num_tasks * sizeof(vT), cudaMemcpyDeviceToHost));

        // Find global minimum
        vT global_min = std::numeric_limits<vT>::max();
        for (vT e : host_energies) {
            if (e < global_min) global_min = e;
        }

        // Collect all states matching the minimum
        std::vector<std::vector<sT>> results;
        for (size_t i = 0; i < num_tasks; i++) {
            // Using small epsilon for float comparison
            if (std::abs(host_energies[i] - global_min) < 1e-5) {
                // Convert packed size_t back to vector<uint8_t>
                results.push_back(binary_representation_to_state_vector<sT>(host_states[i], n));
            }
        }

        // Cleanup
        CUDA_CALL(cudaFree(d_values));
        CUDA_CALL(cudaFree(d_columns));
        CUDA_CALL(cudaFree(d_offsets));
        CUDA_CALL(cudaFree(d_packed_states));
        CUDA_CALL(cudaFree(d_output_energies));

        return results;
    }
};


template class GPUQUBOBruteForcer<IndexType, ValueType, StateType, DenseMatrix<ValueType>>;
template class GPUQUBOBruteForcer<IndexType, ValueType, StateType, SparseMatrix<ValueType, IndexType>>;