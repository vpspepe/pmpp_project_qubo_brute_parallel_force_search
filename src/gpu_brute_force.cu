#include "cuda_util.h"
#include "datatypes.h"
#include "gpu_brute_force.h"
#include "datatypes.h"
#include "cuda_util.h"
#include "kernel.h"
#include "state_vector.h"

#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>
#include <cstdint>
#include "state_vector.h"

unsigned int compute_dim(std::uint64_t global_size, int block_size) {
  return static_cast<unsigned int>((global_size / block_size) +
                                   (global_size % block_size > 0 ? 1 : 0));
}

// Explicitly handle memory to avoid including extra headers
// Simple helper to free memory on scope exit
struct CudaFreer {
  void *p;
  CudaFreer(void *ptr) : p(ptr) {}
  ~CudaFreer() {
    if (p)
      cudaFree(p);
  }
};

// Standardize mapping to avoid fragile indexing
#define BIT_TO_ROW(bit, n) ((n) - 1 - (bit))

__device__ unsigned int g_match_count;

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
* Specialization for dense matrices
* iT: index type
* vT: value type
* sT: state type
*/
template <typename iT, typename vT, typename sT>
struct GPUQUBOBruteForcer<iT, vT, sT, DenseMatrix<vT>>
    : public QUBOBruteForcer<iT, vT, sT, DenseMatrix<vT>> {

  std::vector<std::vector<sT>>
  brute_force_optima(DenseMatrix<vT> const &mat) override {

    std::cout << "DEBUG: Starting GPU Two-Pass Optimization..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    size_t n = mat.rows;

    // 1. Calculate Grid Dimensions
    // We stick to the heuristic that worked well in your previous code
    // int m_fixed_bits = 0;
    // if (n > 14)
    //   m_fixed_bits = n - 14;
    // if (m_fixed_bits > 24)
    //   m_fixed_bits = 24;
    // if (m_fixed_bits < 0)
    //   m_fixed_bits = 0;

    int m_fixed_bits = 14;
    if (m_fixed_bits >= n)
      m_fixed_bits = n - 1;
    if (m_fixed_bits < 0)
      m_fixed_bits = 0;
    std::cout << "DEBUG: m is " << m_fixed_bits << std::endl;

    unsigned long long total_threads = 1ULL << m_fixed_bits;
    std::cout << "DEBUG: total_threads is " << total_threads << std::endl;
    int block_size = 32;
    int grid_size = compute_dim(total_threads, block_size);
    // int grid_size = (total_threads + block_size - 1) / block_size;
    std::cout << "DEBUG: grid_size is " << grid_size << std::endl;

    size_t smem_size = (n * n * sizeof(vT)) + (block_size * sizeof(vT));

    auto stop1 = std::chrono::high_resolution_clock::now();

    // 2. Allocate GPU Memory
    vT *d_Q = nullptr;
    vT *d_global_min = nullptr;
    unsigned int *d_counter = nullptr;
    unsigned long long *d_solutions = nullptr;
    unsigned int max_solutions = 10000000;

    CUDA_CALL(cudaMalloc(&d_Q, n * n * sizeof(vT)));
    CudaFreer freeQ(d_Q);

    CUDA_CALL(cudaMalloc(&d_global_min, sizeof(vT)));
    CudaFreer freeMin(d_global_min);

    CUDA_CALL(cudaMalloc(&d_counter, sizeof(unsigned int)));
    CudaFreer freeCounter(d_counter);

    CUDA_CALL(
        cudaMalloc(&d_solutions, max_solutions * sizeof(unsigned long long)));
    CudaFreer freeSol(d_solutions);

    auto stop2 = std::chrono::high_resolution_clock::now();

    // 3. Copy Data
    CUDA_CALL(
        cudaMemcpy(d_Q, mat.data, n * n * sizeof(vT), cudaMemcpyHostToDevice));

    // Initialize Min Energy
    vT initial_max = std::numeric_limits<vT>::max();
    CUDA_CALL(cudaMemcpy(d_global_min, &initial_max, sizeof(vT),
                         cudaMemcpyHostToDevice));

    auto stop3 = std::chrono::high_resolution_clock::now();

    // 4. PASS 1: Find Minimum Energy
    kernel_find_min_energy<vT><<<grid_size, block_size, smem_size>>>(
        d_Q, n, m_fixed_bits, d_global_min);
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize());

    vT h_min_energy;
    CUDA_CALL(cudaMemcpy(&h_min_energy, d_global_min, sizeof(vT),
                         cudaMemcpyDeviceToHost));

    auto stop4 = std::chrono::high_resolution_clock::now();

    // 5. PASS 2: Collect All Solutions
    CUDA_CALL(cudaMemset(d_counter, 0, sizeof(unsigned int)));
    auto stop41 = std::chrono::high_resolution_clock::now();

    kernel_collect_solutions<vT><<<grid_size, block_size, smem_size>>>(
        d_Q, n, m_fixed_bits, h_min_energy, d_solutions, d_counter,
        max_solutions);
    auto stop42 = std::chrono::high_resolution_clock::now();
    CUDA_CALL(cudaGetLastError());
    auto stop43 = std::chrono::high_resolution_clock::now();
    CUDA_CALL(cudaDeviceSynchronize());

    auto stop5 = std::chrono::high_resolution_clock::now();

    // Retrieve Count
    unsigned int h_count;
    CUDA_CALL(cudaMemcpy(&h_count, d_counter, sizeof(unsigned int),
                         cudaMemcpyDeviceToHost));

    auto stop6 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed;
    std::cout << "DEBUG: Solutions found: " << h_count << std::endl;
    elapsed = stop1 - start;
    // std::cout << "DEBUG: Time taken for parameter selection: "
    //           << elapsed.count() << " ms" << std::endl;
    // elapsed = stop2 - stop1;
    // std::cout << "DEBUG: Time taken for memory allocation: " <<
    // elapsed.count()
    //           << " ms" << std::endl;
    // elapsed = stop3 - stop2;
    // std::cout << "DEBUG: Time taken for data copying: " << elapsed.count()
    //           << " ms" << std::endl;
    // elapsed = stop4 - stop3;
    // std::cout << "DEBUG: Time taken for finding minimal energy: "
    //           << elapsed.count() << " ms" << std::endl;
    elapsed = stop41 - stop4;
    std::cout << "DEBUG: Time taken for memory initialization: "
              << elapsed.count() << " ms" << std::endl;
    elapsed = stop42 - stop41;
    std::cout << "DEBUG: Time taken for running kernel: " << elapsed.count()
              << " ms" << std::endl;
    elapsed = stop43 - stop42;
    std::cout << "DEBUG: Time taken for getting last error: " << elapsed.count()
              << " ms" << std::endl;
    elapsed = stop5 - stop43;
    std::cout << "DEBUG: Time taken for synchrnization: " << elapsed.count()
              << " ms" << std::endl;
    elapsed = stop5 - stop4;
    std::cout << "DEBUG: Time taken for finding all solutions: "
              << elapsed.count() << " ms" << std::endl;
    // elapsed = stop6 - stop5;
    // std::cout << "DEBUG: Time taken for count retrieval: " << elapsed.count()
    //           << " ms" << std::endl;

    if (h_count > max_solutions)
      h_count = max_solutions;

    // Retrieve Solutions
    std::vector<unsigned long long> raw_solutions(h_count);
    if (h_count > 0) {
      CUDA_CALL(cudaMemcpy(raw_solutions.data(), d_solutions,
                           h_count * sizeof(unsigned long long),
                           cudaMemcpyDeviceToHost));
    }

    // 6. Format Output
    std::vector<std::vector<sT>> formatted_solutions;
    formatted_solutions.reserve(h_count);

    for (unsigned long long state_mask : raw_solutions) {
      std::vector<sT> state_vec(n);
      for (size_t bit = 0; bit < n; ++bit) {
        state_vec[bit] = (state_mask >> bit) & 1;
      }
      formatted_solutions.push_back(state_vec);
    }

    return formatted_solutions;
  }
};



/**
* Specialization for sparse matrices
* iT: index type
* vT: value type
* sT: state type
*/
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

template <typename iT, typename vT>
__global__ void collect_all_optima_sparse_kernel(
    const vT* values, const iT* columns, const iT* offsets, 
    size_t n, size_t m, size_t n_sub, 
    vT global_min_energy, // The energy found in Pass 1
    state_t* all_best_states, // Pre-allocated buffer for results
    size_t max_results, // Size of all_best_states to prevent overflow
    const vT* initial_energies 
) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (1ULL << m)) return;

    state_t current_state = tid; 
    vT current_energy = initial_energies[tid];
    
    // Lambda or helper to check and save matches
    auto check_and_save = [&](vT energy, state_t state) {
        if (std::abs(energy - global_min_energy) < 1e-6) {
            unsigned int idx = atomicAdd(&g_match_count, 1);
            if (idx < max_results) {
                all_best_states[idx] = state;
            }
        }
    };

    // Check the starting state of the thread
    check_and_save(current_energy, current_state);

    uint64_t limit = 1ULL << n_sub;
    for (uint64_t k = 1; k < limit; k++) {
        int local_flip = __ffsll(k) - 1; 
        int flip_bit = local_flip + m; 
        int row = BIT_TO_ROW(flip_bit, n);

        vT sign = ((current_state >> flip_bit) & 1ULL) ? -1.0 : 1.0;
        vT running_sum = 0.0;

        for (iT idx = offsets[row]; idx < offsets[row + 1]; idx++) {
            int col = columns[idx];
            int col_bit = n - 1 - col;
            if (col == row || (current_state >> col_bit) & 1ULL) {
                running_sum += values[idx];
            }
        }

        current_energy += sign * running_sum;
        current_state ^= (1ULL << flip_bit);

        // Check every single state in the Gray code sequence
        check_and_save(current_energy, current_state);
    }
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

        int blockSize = 32;
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

        size_t max_res = 1024;
        state_t *d_all_best_states;
        CUDA_CALL(cudaMalloc(&d_all_best_states, max_res * sizeof(state_t)));

        // Reset the device counter to 0
        unsigned int zero = 0;
        CUDA_CALL(cudaMemcpyToSymbol(g_match_count, &zero, sizeof(unsigned int)));

        // Pass 2: Launch collection kernel
        collect_all_optima_sparse_kernel<<<numBlocks, blockSize>>>(
            d_values, d_columns, d_offsets, n, m, remaining,
            global_min, d_all_best_states, max_res, d_initial_energies
        );

        // Copy back the total count and the states
        unsigned int h_match_count;
        CUDA_CALL(cudaMemcpyFromSymbol(&h_match_count, g_match_count, sizeof(unsigned int)));
        if (h_match_count > max_res) h_match_count = max_res; // Cap to buffer size

        std::vector<state_t> final_host_states(h_match_count);
        CUDA_CALL(cudaMemcpy(final_host_states.data(), d_all_best_states, h_match_count * sizeof(state_t), cudaMemcpyDeviceToHost));

        // Now build the results vector for every state in final_host_states
        std::vector<std::vector<sT>> results;
        for (state_t s : final_host_states) {
            results.push_back(binary_representation_to_state_vector<sT>(s, n));
        }

        // Cleanup
        CUDA_CALL(cudaFree(d_values)); CUDA_CALL(cudaFree(d_columns)); CUDA_CALL(cudaFree(d_offsets));
        CUDA_CALL(cudaFree(d_packed_states)); CUDA_CALL(cudaFree(d_output_energies)); CUDA_CALL(cudaFree(d_initial_energies));

        return results;
    }
};


template struct GPUQUBOBruteForcer<IndexType, ValueType, StateType, DenseMatrix<ValueType>>;
template class GPUQUBOBruteForcer<IndexType, ValueType, StateType, SparseMatrix<ValueType, IndexType>>;