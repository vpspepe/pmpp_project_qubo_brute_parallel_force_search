#include "cuda_util.h"
#include "datatypes.h"
#include "gpu_brute_force.h"
#include "kernel.h"       // Interface dos kernels
#include "state_vector.h" // Necessário para converter bits para vector<sT>

#include <algorithm> // Para std::min/max
#include <cmath>
#include <limits>
#include <vector>

// =================================================================================================
// Implementação Padrão (Erro para tipos desconhecidos)
// =================================================================================================
template <typename iT, typename vT, typename sT, typename MatrixType>
std::vector<std::vector<sT>>
GPUQUBOBruteForcer<iT, vT, sT, MatrixType>::brute_force_optima(
    MatrixType const &mat) {
  throw std::runtime_error(
      "GPUQUBOBruteForcer not implemented for this matrix type.");
  return std::vector<std::vector<sT>>();
}

// =================================================================================================
// Especialização: Matrizes Densas (DenseMatrix)
// =================================================================================================
template <typename iT, typename vT, typename sT>
struct GPUQUBOBruteForcer<iT, vT, sT, DenseMatrix<vT>>
    : public QUBOBruteForcer<iT, vT, sT, DenseMatrix<vT>> {

  std::vector<std::vector<sT>>
  brute_force_optima(DenseMatrix<vT> const &mat) override {

    // -------------------------------------------------------------------------
    // 1. Configuração de Dimensões e Paralelismo
    // -------------------------------------------------------------------------
    size_t n = mat.rows;

    // Heurística de Prefix Fixing:
    // Queremos criar threads suficientes para ocupar a GPU, mas dar trabalho
    // suficiente (sufixo) para cada thread compensar o overhead de criação.
    // Estratégia: Tentar deixar aprox. 14 bits para o sufixo (16.384
    // iterações/thread).
    int n_fixed_bits = 0;
    if (n > 14)
      n_fixed_bits = n - 14;

    // Limite de segurança: Não criar mais threads do que o grid suporta
    // facilmente
    if (n_fixed_bits > 24)
      n_fixed_bits = 24;

    // Número total de threads (Prefixos únicos)
    size_t num_threads = 1ULL << n_fixed_bits;
    // Tamanho do loop de cada thread (Sufixos a explorar)
    unsigned long long states_per_thread = 1ULL << (n - n_fixed_bits);

    int threads_per_block = 256;
    // Número de blocos necessários
    int blocks = (num_threads + threads_per_block - 1) / threads_per_block;

    // -------------------------------------------------------------------------
    // 2. Cálculo da Memória Compartilhada
    // -------------------------------------------------------------------------
    // Precisamos alocar espaço para:
    // A) A Matriz Q inteira (n * n * sizeof(vT))
    // B) Buffer de Redução de Energia (threads_per_block * sizeof(vT))
    // C) Buffer de Redução de Estado (threads_per_block * sizeof(unsigned long
    // long))
    size_t smem_matrix = n * n * sizeof(vT);
    size_t smem_reduction_energy = threads_per_block * sizeof(vT);
    size_t smem_reduction_state =
        threads_per_block * sizeof(unsigned long long);

    size_t shared_mem_size =
        smem_matrix + smem_reduction_energy + smem_reduction_state;

    // -------------------------------------------------------------------------
    // 3. Alocação de Memória na GPU
    // -------------------------------------------------------------------------
    vT *d_Q = nullptr;
    vT *d_energies = nullptr;
    unsigned long long *d_states = nullptr;

    // Alocamos buffers de saída com tamanho 'blocks' (resultado da redução no
    // kernel)
    CUDA_CALL(cudaMalloc(&d_Q, n * n * sizeof(vT)));
    CUDA_CALL(cudaMalloc(&d_energies, blocks * sizeof(vT)));
    CUDA_CALL(cudaMalloc(&d_states, blocks * sizeof(unsigned long long)));

    // -------------------------------------------------------------------------
    // 4. Transferência de Dados
    // -------------------------------------------------------------------------
    CUDA_CALL(
        cudaMemcpy(d_Q, mat.data, n * n * sizeof(vT), cudaMemcpyHostToDevice));

    // -------------------------------------------------------------------------
    // 5. Lançamento do Kernel
    // -------------------------------------------------------------------------
    kernel_brute_force_dense<<<blocks, threads_per_block, shared_mem_size>>>(
        d_Q, n, n_fixed_bits, states_per_thread, d_energies, d_states);
    CUDA_CALL(cudaGetLastError());      // Checa erros de lançamento (ex: shared
                                        // memory insuficiente)
    CUDA_CALL(cudaDeviceSynchronize()); // Aguarda a GPU terminar

    // -------------------------------------------------------------------------
    // 6. Recuperação dos Resultados (GPU -> CPU)
    // -------------------------------------------------------------------------
    std::vector<vT> h_energies(blocks);
    std::vector<unsigned long long> h_states(blocks);

    CUDA_CALL(cudaMemcpy(h_energies.data(), d_energies, blocks * sizeof(vT),
                         cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(h_states.data(), d_states,
                         blocks * sizeof(unsigned long long),
                         cudaMemcpyDeviceToHost));

    // -------------------------------------------------------------------------
    // 7. Redução Final na CPU (Encontrar Mínimo Global)
    // -------------------------------------------------------------------------
    vT global_min = std::numeric_limits<vT>::max();

    // Passo A: Encontrar o menor valor de energia entre todos os blocos
    for (auto e : h_energies) {
      if (e < global_min)
        global_min = e;
    }

    // Passo B: Coletar todos os estados que atingiram esse mínimo (pode haver
    // empates)
    std::vector<std::vector<sT>> results;
    vT epsilon =
        static_cast<vT>(1e-5); // Tolerância para comparação de ponto flutuante

    for (size_t i = 0; i < blocks; ++i) {
      if (std::abs(h_energies[i] - global_min) < epsilon) {
        // Converte o inteiro compactado (unsigned long long) para o formato de
        // vetor esperado
        results.push_back(
            binary_representation_to_state_vector<sT>(h_states[i], n));
      }
    }

    // -------------------------------------------------------------------------
    // 8. Limpeza
    // -------------------------------------------------------------------------
    CUDA_CALL(cudaFree(d_Q));
    CUDA_CALL(cudaFree(d_energies));
    CUDA_CALL(cudaFree(d_states));

    return results;
  }
};

// =================================================================================================
// Especialização: Matrizes Esparsas (SparseMatrix) - Placeholder
// =================================================================================================
template <typename iT, typename vT, typename sT>
struct GPUQUBOBruteForcer<iT, vT, sT, SparseMatrix<vT, iT>>
    : public QUBOBruteForcer<iT, vT, sT, SparseMatrix<vT, iT>> {

  std::vector<std::vector<sT>>
  brute_force_optima(SparseMatrix<vT, iT> const &mat) override {
    // A ser implementado futuramente
    return std::vector<std::vector<sT>>();
  }
};

// =================================================================================================
// Instanciação Explícita dos Templates
// =================================================================================================
template class GPUQUBOBruteForcer<IndexType, ValueType, StateType,
                                  DenseMatrix<ValueType>>;
template class GPUQUBOBruteForcer<IndexType, ValueType, StateType,
                                  SparseMatrix<ValueType, IndexType>>;
