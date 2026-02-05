#include "gpu_brute_force.h"
#include "datatypes.h"
#include "cuda_util.h"
#include "state_vector.h"
#include <cmath>
#include <vector>
#include <cusparse.h> // Novo include para cuSPARSE

// Forward declaration of the kernel
template <typename iT, typename vT, typename sT>
__global__ void sparse_qubo_kernel(
    const vT* values, const iT* columns, const iT* offsets, 
    size_t n, size_t m, size_t n_sub, 
    sT* output_states, vT* output_energies,
    const vT* initial_energies // NOVO: Energias iniciais pré-calculadas
) {
    // 1. Thread ID represents the fixed LOWER bits (LSB)
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long num_tasks = 1ULL << m;

    if (tid >= num_tasks) return;

    // 2. Initialize State
    unsigned long long current_state = tid; 
    
    // 3. Compute Initial Energy (REPLACED BY cuSPARSE LATER)
    // Agora a thread apenas lê o valor inicial em vez de calcular do zero
    vT current_energy = initial_energies[tid];

    // Initialize Best Local Result
    vT min_energy = current_energy;
    unsigned long long best_state = current_state;

    // 4. Gray Code Loop
    unsigned long long limit = 1ULL << n_sub;

    for (unsigned long long k = 1; k < limit; k++) {
        int local_flip = __ffsll(k) - 1; 
        int flip_bit = local_flip + m; 
        size_t matrix_row_idx = n - 1 - flip_bit; 

        // 5. Calculate Delta Energy
        size_t row = matrix_row_idx;
        iT start = offsets[row];
        iT end = offsets[row + 1];

        vT energy_diff = 0.0;
        size_t current_bit_val = (current_state >> flip_bit) & 1ULL;
        vT sign = (current_bit_val == 0) ? 1.0 : -1.0;

        for (iT idx = start; idx < end; idx++) {
            iT col = columns[idx];
            if (col == row) {
                energy_diff += values[idx];
            } else {
                size_t col_bit = (current_state >> (n - 1 - col)) & 1ULL;
                if (col_bit) {
                    energy_diff += values[idx];
                }
            }
        }

        energy_diff *= sign;

        // 6. Update State and Energy
        current_energy += energy_diff;
        current_state ^= (1ULL << flip_bit); 

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

// ... (Funções intermediárias permanecem iguais) ...

/** * Specialization for sparse matrices
*/
template<typename iT, typename vT, typename sT>
struct GPUQUBOBruteForcer<iT, vT, sT, SparseMatrix<vT, iT>> : public QUBOBruteForcer<iT, vT, sT, SparseMatrix<vT, iT>>
{
    std::vector<std::vector<sT>> brute_force_optima(SparseMatrix<vT, iT> const & mat) override {
        size_t n = mat.rows;

        // --- 1. Calculate M (Fixed Bits) ---
        size_t m = static_cast<size_t>(n * 0.2);
        size_t remaining = n - m;

        if (remaining >= 25) {
            remaining = 25;
            m = n - 25; 
        }

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

        // --- 3. Allocate Output Arrays & Initial Energies ---
        size_t *d_packed_states;
        vT *d_output_energies;
        vT *d_initial_energies; // Espaço para as energias iniciais calculadas via cuSPARSE

        CUDA_CALL(cudaMalloc(&d_packed_states, num_tasks * sizeof(size_t)));
        CUDA_CALL(cudaMalloc(&d_output_energies, num_tasks * sizeof(vT)));
        CUDA_CALL(cudaMalloc(&d_initial_energies, num_tasks * sizeof(vT)));

        // --- 3.1 SETUP CUSPARSE (NOVO) ---
        cusparseHandle_t handle;
        cusparseCreate(&handle);
        // TODO: Adicionar lógica de simetrização e SpMV aqui para preencher d_initial_energies

        // --- 4. Launch Kernel ---
        int blockSize = 256; 
        unsigned long long numBlocks = (num_tasks + blockSize - 1) / blockSize;

        sparse_qubo_kernel<iT, vT, size_t><<<dim3(numBlocks), dim3(blockSize)>>>(
            d_values, d_columns, d_offsets,
            n, m, remaining,
            d_packed_states, d_output_energies,
            d_initial_energies // Passando o novo vetor
        );
        CUDA_CALL(cudaGetLastError());
        CUDA_CALL(cudaDeviceSynchronize());

        // --- 5. Retrieve & Reduce Results ---
        // (Lógica de redução permanece igual...)
        // ...
        
        // Cleanup
        cusparseDestroy(handle);
        CUDA_CALL(cudaFree(d_values));
        CUDA_CALL(cudaFree(d_columns));
        CUDA_CALL(cudaFree(d_offsets));
        CUDA_CALL(cudaFree(d_packed_states));
        CUDA_CALL(cudaFree(d_output_energies));
        CUDA_CALL(cudaFree(d_initial_energies));

        return results;
    }
};