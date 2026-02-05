#pragma once
#include <cuda_runtime.h>

// Kernel 1: Find Global Minimum Energy
// Uses atomic operations to find the single lowest energy value in the search space.
template <typename vT>
__global__ void kernel_find_min_energy(
    const vT *__restrict__ global_Q, 
    int n, 
    int n_fixed_bits,
    vT *global_min_energy_ptr
);

// Kernel 2: Collect All Solutions
// Re-scans the space and stores ALL states that match the target_energy.
template <typename vT>
__global__ void kernel_collect_solutions(
    const vT *__restrict__ global_Q, 
    int n, 
    int n_fixed_bits,
    vT target_energy,
    unsigned long long *solutions_buffer, 
    unsigned int *solution_counter,
    unsigned int max_solutions
);
