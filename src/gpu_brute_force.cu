#include "cuda_util.h"
#include "datatypes.h"
#include "gpu_brute_force.h"
#include "kernel.h"
#include "state_vector.h"

#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

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

// Explicit Instantiations
template struct GPUQUBOBruteForcer<IndexType, ValueType, StateType,
                                   DenseMatrix<ValueType>>;
template struct GPUQUBOBruteForcer<IndexType, ValueType, StateType,
                                   SparseMatrix<ValueType, IndexType>>;
