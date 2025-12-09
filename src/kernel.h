#pragma once
#include <cuda_runtime.h>

// Dense Kernel Declaration
template <typename vT>
__global__ void
kernel_brute_force_dense(const vT *global_Q, int n, int n_fixed_bits,
                         unsigned long long num_states_per_thread,
                         vT *global_best_energy,
                         unsigned long long *global_best_state,
                         unsigned long long *found_states_buffer, // NOVO
                         unsigned int *found_counter,             // NOVO
                         unsigned int max_global_solutions        // NOVO
);
// Sparse Kernel Declaration
template <typename vT, typename iT>
__global__ void kernel_brute_force_sparse(
    const vT *values, const iT *offsets, const iT *columns, int n,
    int n_fixed_bits, unsigned long long num_states_per_thread,
    vT *global_best_energy, unsigned long long *global_best_state);
