#include "cuda_util.h"
#include "kernel.h"
#include <cfloat>
#include <type_traits>

// Helper macro for absolute difference to handle both int and float/double
#define ABS_DIFF(a, b) ((a) > (b) ? (a) - (b) : (b) - (a))

// =============================================================================
// ATOMIC MIN WRAPPERS
// CUDA's native atomicMin supports int, but float/double requires CAS loops
// on architectures < sm_90.
// =============================================================================

__device__ double atomicMin_double(double *address, double val) {
  unsigned long long *address_as_ull = (unsigned long long *)address;
  unsigned long long old = *address_as_ull, assumed;
  do {
    assumed = old;
    double old_val = __longlong_as_double(assumed);
    if (old_val <= val)
      break;
    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val));
  } while (assumed != old);
  return __longlong_as_double(old);
}

__device__ float atomicMin_float(float *address, float val) {
  int *address_as_int = (int *)address;
  int old = *address_as_int, assumed;
  do {
    assumed = old;
    float old_val = __int_as_float(assumed);
    if (old_val <= val)
      break;
    old = atomicCAS(address_as_int, assumed, __float_as_int(val));
  } while (assumed != old);
  return __int_as_float(old);
}

// Generic overloads for the kernel to use
__device__ inline void gpu_atomic_min(int *address, int val) {
  atomicMin(address, val);
}
__device__ inline void gpu_atomic_min(float *address, float val) {
  atomicMin_float(address, val);
}
__device__ inline void gpu_atomic_min(double *address, double val) {
  atomicMin_double(address, val);
}

// =============================================================================
// KERNEL 1: FIND MINIMUM ENERGY
// =============================================================================
template <typename vT>
__global__ void kernel_find_min_energy(const vT *__restrict__ global_Q, int n,
                                       int n_fixed_bits,
                                       vT *global_min_energy_ptr) {
  // Shared Memory: Matrix Q + Reduction Buffer
  extern __shared__ char smem[];
  vT *shared_Q = reinterpret_cast<vT *>(smem);
  vT *s_min_results = (vT *)&shared_Q[n * n];

  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int dim = blockDim.x;

  // 1. Cooperative Load of Matrix Q
  int num_elements = n * n;
  for (int i = tid; i < num_elements; i += dim) {
    shared_Q[i] = global_Q[i];
  }
  __syncthreads();

  // 2. Ghost Thread Guard
  // If we launch more threads than there are prefixes (e.g. N=10, Threads=256),
  // the extra threads must be neutralized.
  unsigned long long global_tid = (unsigned long long)bid * dim + tid;
  unsigned long long total_patterns = 1ULL << n_fixed_bits;
  bool is_valid_thread = (global_tid < total_patterns);

  vT local_min;

  if (!is_valid_thread) {
    // Initialize with MAX value so it loses the reduction
    if constexpr (std::is_integral<vT>::value)
      local_min = 2147483647;
    else if constexpr (sizeof(vT) == 8)
      local_min = DBL_MAX;
    else
      local_min = FLT_MAX;
  } else {
    // --- VALID THREAD LOGIC ---
    unsigned long long fixed_prefix = global_tid;
    int n_vary = n - n_fixed_bits;
    unsigned long long limit = 1ULL << n_vary;

    // Pre-calculate Fixed Interactions to optimize the loop
    vT cached_fixed_interactions[32]; // Max N=32 usually
    vT current_energy = 0;

    // Initial Energy (Fixed Bits Part)
    // Calculating interactions among fixed bits
    for (int i = 0; i < n_fixed_bits; ++i) {
      // Note: fixed_prefix bits are at the "top" (n-1 down to n_vary)
      // But for calculation simplicity, we map them to indices 0..n_fixed-1
      // relative However, looking at your original logic, you construct the
      // full state. Let's stick to the efficient delta update method.

      if ((fixed_prefix >> i) & 1) {
        int row = n_vary + i; // Map to the upper part of the matrix
        // Diagonal
        current_energy += shared_Q[row * n + row];
        // Interaction with previous fixed bits
        for (int j = 0; j < i; ++j) {
          if ((fixed_prefix >> j) & 1) {
            int col = n_vary + j;
            // Consistent with your original kernel: sum one side
            current_energy += shared_Q[row * n + col];
          }
        }
      }
    }

    // Cache Interactions (Fixed <-> Varying)
    for (int k = 0; k < n_vary; ++k) {
      vT interaction = 0;
      int k_real_idx = k; // Varying bits are 0..n_vary-1
      for (int j = 0; j < n_fixed_bits; ++j) {
        if ((fixed_prefix >> j) & 1) {
          int col = n_vary + j;
          // Sum interactions
          interaction += shared_Q[k_real_idx * n + col];
        }
      }
      cached_fixed_interactions[k] = interaction;
    }

    local_min = current_energy;

    // Gray Code Loop
    unsigned long long x_vary = 0;
    for (unsigned long long i = 1; i < limit; i++) {
      int k = __ffsll(i) - 1; // Bit index to flip

      // 1. Fixed Interaction
      vT delta = cached_fixed_interactions[k];
      // 2. Diagonal
      delta += shared_Q[k * n + k];

      // 3. Variable Interaction
      for (int j = 0; j < n_vary; ++j) {
        if (j != k && ((x_vary >> j) & 1)) {
          // Using Sum of both sides Q[k][j] + Q[j][k] to be strictly correct
          // with full dense matrix energy definition
          delta += shared_Q[k * n + j];
        }
      }

      int sign = ((x_vary >> k) & 1) ? -1 : 1;

      // RE-CALCULATION FOR CONSISTENCY WITH YOUR OLD CODE:
      delta = cached_fixed_interactions[k]; // Correction attempt

      vT simple_delta = shared_Q[k * n + k]; // Diagonal

      // Interactions with Fixed Bits
      for (int j = 0; j < n_fixed_bits; ++j) {
        if ((fixed_prefix >> j) & 1) {
          simple_delta += shared_Q[k * n + (n_vary + j)];
        }
      }

      // Interactions with Variable Bits
      for (int j = 0; j < n_vary; ++j) {
        if (j != k && ((x_vary >> j) & 1)) {
          simple_delta += shared_Q[k * n + j];
        }
      }

      current_energy += (sign * simple_delta);
      x_vary ^= (1ULL << k);

      if (current_energy < local_min) {
        local_min = current_energy;
      }
    }
  } // End valid thread

  // 3. Block Reduction
  s_min_results[tid] = local_min;
  __syncthreads();

  for (unsigned int s = dim / 2; s > 0; s >>= 1) {
    if (tid < s) {
      if (s_min_results[tid + s] < s_min_results[tid]) {
        s_min_results[tid] = s_min_results[tid + s];
      }
    }
    __syncthreads();
  }

  // 4. Global Update
  if (tid == 0) {
    gpu_atomic_min(global_min_energy_ptr, s_min_results[0]);
  }
}

// =============================================================================
// KERNEL 2: COLLECT SOLUTIONS
// =============================================================================
template <typename vT>
__global__ void kernel_collect_solutions(const vT *__restrict__ global_Q, int n,
                                         int n_fixed_bits, vT target_energy,
                                         unsigned long long *solutions_buffer,
                                         unsigned int *solution_counter,
                                         unsigned int max_solutions) {
  extern __shared__ char smem[];
  vT *shared_Q = reinterpret_cast<vT *>(smem);
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int dim = blockDim.x;

  int num_elements = n * n;
  for (int i = tid; i < num_elements; i += dim) {
    shared_Q[i] = global_Q[i];
  }
  __syncthreads();

  unsigned long long global_tid = (unsigned long long)bid * dim + tid;
  unsigned long long total_patterns = 1ULL << n_fixed_bits;

  // Ghost thread check
  if (global_tid >= total_patterns)
    return;

  unsigned long long fixed_prefix = global_tid;
  int n_vary = n - n_fixed_bits;
  unsigned long long limit = 1ULL << n_vary;

  // --- REPLICATE EXACT ENERGY LOGIC FROM KERNEL 1 ---
  vT current_energy = 0;

  // Initial Energy
  for (int i = 0; i < n_fixed_bits; ++i) {
    if ((fixed_prefix >> i) & 1) {
      int row = n_vary + i;
      current_energy += shared_Q[row * n + row];
      for (int j = 0; j < i; ++j) {
        if ((fixed_prefix >> j) & 1) {
          int col = n_vary + j;
          current_energy += shared_Q[row * n + col] + shared_Q[col * n + row];
        }
      }
    }
  }

  // Relaxed Epsilon for float comparisons
  vT epsilon = 0;
  if constexpr (!std::is_integral<vT>::value)
    epsilon = 1e-3;

  // Check Initial
  if (ABS_DIFF(current_energy, target_energy) <= epsilon) {
    unsigned int idx = atomicAdd(solution_counter, 1);
    if (idx < max_solutions) {
      solutions_buffer[idx] = (fixed_prefix << n_vary);
    }
  }

  unsigned long long x_vary = 0;
  for (unsigned long long i = 1; i < limit; i++) {
    int k = __ffsll(i) - 1;

    vT simple_delta = shared_Q[k * n + k]; // Diagonal

    // Interactions with Fixed
    for (int j = 0; j < n_fixed_bits; ++j) {
      if ((fixed_prefix >> j) & 1) {
        simple_delta += shared_Q[k * n + (n_vary + j)];
      }
    }

    // Interactions with Variable
    for (int j = 0; j < n_vary; ++j) {
      if (j != k && ((x_vary >> j) & 1)) {
        simple_delta += shared_Q[k * n + j];
      }
    }

    int sign = ((x_vary >> k) & 1) ? -1 : 1;
    current_energy += (sign * simple_delta);
    x_vary ^= (1ULL << k);

    if (ABS_DIFF(current_energy, target_energy) <= epsilon) {
      unsigned int idx = atomicAdd(solution_counter, 1);
      if (idx < max_solutions) {
        solutions_buffer[idx] = (fixed_prefix << n_vary) | x_vary;
      }
    }
  }
}

// Explicit Instantiations
template __global__ void kernel_find_min_energy<int>(const int *, int, int,
                                                     int *);
template __global__ void kernel_find_min_energy<float>(const float *, int, int,
                                                       float *);
template __global__ void kernel_find_min_energy<double>(const double *, int,
                                                        int, double *);

template __global__ void
kernel_collect_solutions<int>(const int *, int, int, int, unsigned long long *,
                              unsigned int *, unsigned int);
template __global__ void kernel_collect_solutions<float>(const float *, int,
                                                         int, float,
                                                         unsigned long long *,
                                                         unsigned int *,
                                                         unsigned int);
template __global__ void kernel_collect_solutions<double>(const double *, int,
                                                          int, double,
                                                          unsigned long long *,
                                                          unsigned int *,
                                                          unsigned int);
