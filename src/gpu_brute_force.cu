#include "cuda_util.h"
#include "datatypes.h"
#include "gpu_brute_force.h"
#include "kernel.h" // Atualize o kernel.h com a nova assinatura!
#include "state_vector.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

template <typename iT, typename vT, typename sT, typename MatrixType>
std::vector<std::vector<sT>>
GPUQUBOBruteForcer<iT, vT, sT, MatrixType>::brute_force_optima(
    MatrixType const &mat) {
  throw std::runtime_error("GPUQUBOBruteForcer not implemented generic.");
  return {};
}

template <typename iT, typename vT, typename sT>
struct GPUQUBOBruteForcer<iT, vT, sT, DenseMatrix<vT>>
    : public QUBOBruteForcer<iT, vT, sT, DenseMatrix<vT>> {

  std::vector<std::vector<sT>>
  brute_force_optima(DenseMatrix<vT> const &mat) override {

    size_t n = mat.rows;

    // Heurística de Prefix Fixing:
    // Queremos criar threads suficientes para ocupar a GPU, mas dar trabalho
    // suficiente (sufixo) para cada thread compensar o overhead de criação.
    // Estratégia: Tentar deixar aprox. 14 bits para o sufixo (16.384
    // iterações/thread).
    int max_fixed_bits = 16;
    int n_fixed_bits = 0;
    if (n > max_fixed_bits)
      n_fixed_bits = max_fixed_bits;
    else
      n_fixed_bits = static_cast<int>(n);
    // Limite de segurança: Não criar mais threads do que o grid suporta
    // facilmente
    // if (n_fixed_bits > 24)
    //   n_fixed_bits = 24;

    size_t num_threads = 1ULL << n_fixed_bits;
    unsigned long long states_per_thread = 1ULL << (n - n_fixed_bits);

    int threads_per_block = 256;
    int blocks = (num_threads + threads_per_block - 1) / threads_per_block;

    // Memória Compartilhada: Matriz + Buffer de Redução de Energia
    size_t smem_size = n * n * sizeof(vT) + threads_per_block * sizeof(vT);

    // Alocação
    vT *d_Q;
    unsigned long long *d_found_states;
    unsigned int *d_counter;

    // Buffer global generoso (ex: 500k soluções)
    unsigned int max_solutions = 500000;

    CUDA_CALL(cudaMalloc(&d_Q, n * n * sizeof(vT)));
    CUDA_CALL(cudaMalloc(&d_found_states,
                         max_solutions * sizeof(unsigned long long)));
    CUDA_CALL(cudaMalloc(&d_counter, sizeof(unsigned int)));

    // Zera contador
    CUDA_CALL(cudaMemset(d_counter, 0, sizeof(unsigned int)));
    // Copia Matriz
    CUDA_CALL(
        cudaMemcpy(d_Q, mat.data, n * n * sizeof(vT), cudaMemcpyHostToDevice));

    // Lança Kernel (Passo Único!)
    // Passamos nullptr para os antigos arrays de output que não usamos mais
    kernel_brute_force_dense<<<blocks, threads_per_block, smem_size>>>(
        d_Q, n, n_fixed_bits, states_per_thread, nullptr, nullptr, // Ignorados
        d_found_states, d_counter, max_solutions // Novos parâmetros
    );
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize());

    // Recupera contagem
    unsigned int h_counter;
    CUDA_CALL(cudaMemcpy(&h_counter, d_counter, sizeof(unsigned int),
                         cudaMemcpyDeviceToHost));

    if (h_counter > max_solutions)
      h_counter = max_solutions;

    // Recupera estados brutos
    std::vector<unsigned long long> h_raw_states(h_counter);
    if (h_counter > 0) {
      CUDA_CALL(cudaMemcpy(h_raw_states.data(), d_found_states,
                           h_counter * sizeof(unsigned long long),
                           cudaMemcpyDeviceToHost));
    }

    // --- FILTRAGEM FINAL NA CPU ---
    // Como cada bloco retornou o SEU melhor, o vetor global contém os campeões
    // de cada bloco. Ainda precisamos filtrar o campeão global entre eles.

    vT global_min = std::numeric_limits<vT>::max();

    // 1. Recalcular energias (rápido na CPU para < 500k itens) e achar o mínimo
    // Nota: Poderíamos ter retornado a energia da GPU, mas recalcular aqui
    // simplifica a gestão de memória e evita race conditions na escrita de um
    // único valor "min_global" na GPU.
    std::vector<vT> energies(h_counter);

    for (size_t i = 0; i < h_counter; ++i) {
      // Usamos a função compute_energy do qubo_energy.h (CPU version)
      // Precisamos converter o bits para vetor ou adaptar compute_energy para
      // aceitar bits O qubo_energy.h TEM uma sobrecarga para (DenseMatrix,
      // size_t state)! Perfeito.
      energies[i] = compute_energy(mat, (size_t)h_raw_states[i]);

      if (energies[i] < global_min)
        global_min = energies[i];
    }

    // 2. Selecionar apenas os que são iguais ao mínimo global
    std::vector<std::vector<sT>> results;
    vT epsilon = static_cast<vT>(1e-5);

    for (size_t i = 0; i < h_counter; ++i) {
      if (std::abs(energies[i] - global_min) < epsilon) {
        results.push_back(
            binary_representation_to_state_vector<sT>(h_raw_states[i], n));
      }
    }

    // Limpeza
    CUDA_CALL(cudaFree(d_Q));
    CUDA_CALL(cudaFree(d_found_states));
    CUDA_CALL(cudaFree(d_counter));

    return results;
  }
};

// Instanciação
template class GPUQUBOBruteForcer<IndexType, ValueType, StateType,
                                  DenseMatrix<ValueType>>;
template class GPUQUBOBruteForcer<IndexType, ValueType, StateType,
                                  SparseMatrix<ValueType, IndexType>>;
