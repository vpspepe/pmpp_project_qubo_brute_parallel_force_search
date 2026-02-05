#include "cuda_util.h"
#include "datatypes.h"
#include "gpu_brute_force.h"
#include "kernel.h"
#include "state_vector.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

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

template <typename iT, typename vT, typename sT, typename MatrixType>
std::vector<std::vector<sT>>
GPUQUBOBruteForcer<iT, vT, sT, MatrixType>::brute_force_optima(
    MatrixType const &mat) {
  throw std::runtime_error("GPUQUBOBruteForcer not implemented generic.");
  return {};
}

// Specialization for DenseMatrix
template <typename iT, typename vT, typename sT>
struct GPUQUBOBruteForcer<iT, vT, sT, DenseMatrix<vT>>
    : public QUBOBruteForcer<iT, vT, sT, DenseMatrix<vT>> {

  std::vector<std::vector<sT>>
  brute_force_optima(DenseMatrix<vT> const &mat) override {

    std::cout << "DEBUG: Starting GPU Two-Pass Optimization..." << std::endl;

    size_t n = mat.rows;

    // 1. Calculate Grid Dimensions
    // We stick to the heuristic that worked well in your previous code
    int m_fixed_bits = 0;
    if (n > 14)
      m_fixed_bits = n - 14;
    if (m_fixed_bits > 24)
      m_fixed_bits = 24;
    if (m_fixed_bits < 0)
      m_fixed_bits = 0;

    unsigned long long total_threads = 1ULL << m_fixed_bits;
    int block_size = 256;
    int grid_size = (total_threads + block_size - 1) / block_size;

    size_t smem_size = (n * n * sizeof(vT)) + (block_size * sizeof(vT));

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

    // 3. Copy Data
    CUDA_CALL(
        cudaMemcpy(d_Q, mat.data, n * n * sizeof(vT), cudaMemcpyHostToDevice));

    // Initialize Min Energy
    vT initial_max = std::numeric_limits<vT>::max();
    CUDA_CALL(cudaMemcpy(d_global_min, &initial_max, sizeof(vT),
                         cudaMemcpyHostToDevice));

    // 4. PASS 1: Find Minimum Energy
    kernel_find_min_energy<vT><<<grid_size, block_size, smem_size>>>(
        d_Q, n, m_fixed_bits, d_global_min);
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize());

    vT h_min_energy;
    CUDA_CALL(cudaMemcpy(&h_min_energy, d_global_min, sizeof(vT),
                         cudaMemcpyDeviceToHost));

    // 5. PASS 2: Collect All Solutions
    CUDA_CALL(cudaMemset(d_counter, 0, sizeof(unsigned int)));

    kernel_collect_solutions<vT><<<grid_size, block_size, smem_size>>>(
        d_Q, n, m_fixed_bits, h_min_energy, d_solutions, d_counter,
        max_solutions);
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize());

    // Retrieve Count
    unsigned int h_count;
    CUDA_CALL(cudaMemcpy(&h_count, d_counter, sizeof(unsigned int),
                         cudaMemcpyDeviceToHost));

    std::cout << "DEBUG: Solutions found: " << h_count << std::endl;

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

// Explicit Instantiations
template struct GPUQUBOBruteForcer<IndexType, ValueType, StateType,
                                   DenseMatrix<ValueType>>;
template struct GPUQUBOBruteForcer<IndexType, ValueType, StateType,
                                   SparseMatrix<ValueType, IndexType>>;
