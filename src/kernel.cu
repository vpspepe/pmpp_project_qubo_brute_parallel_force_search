#include "cuda_util.h"
#include "kernel.h"
#include "qubo_energy.h" // Funções auxiliares (certifique-se que tenham __device__)

// =================================================================================================
// Implementação do Kernel Denso
// =================================================================================================

template <typename vT>
__global__ void
kernel_brute_force_dense(const vT *global_Q, int n, int n_fixed_bits,
                         unsigned long long num_states_per_thread,
                         vT *global_best_energy,
                         unsigned long long *global_best_state) {
  // -------------------------------------------------------------------------
  // 1. Configuração da Shared Memory
  // -------------------------------------------------------------------------
  // Layout: [ Matriz Q (n*n) ] [ Redução Energia (blockDim) ] [ Redução Estado
  // (blockDim) ]
  extern __shared__ char smem[];
  vT *shared_Q = reinterpret_cast<vT *>(smem);

  // Ponteiros para a área de redução (logo após a matriz)
  vT *s_energy = (vT *)&shared_Q[n * n];
  unsigned long long *s_state = (unsigned long long *)&s_energy[blockDim.x];

  // -------------------------------------------------------------------------
  // 2. Carregamento Colaborativo da Matriz
  // -------------------------------------------------------------------------
  int tid_in_block = threadIdx.x;
  int matrix_size = n * n;

  for (int i = tid_in_block; i < matrix_size; i += blockDim.x) {
    shared_Q[i] = global_Q[i];
  }

  __syncthreads(); // Barreira: Espera o carregamento terminar

  // -------------------------------------------------------------------------
  // 3. Estado Inicial e Energia Base
  // -------------------------------------------------------------------------
  unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long current_state = tid << (n - n_fixed_bits);

  // Cálculo da energia inicial completa (O(N^2))
  vT current_energy = 0;
  for (int i = 0; i < n; ++i) {
    if ((current_state >> (n - 1 - i)) & 1) {
      for (int j = i; j < n; ++j) {
        if ((current_state >> (n - 1 - j)) & 1) {
          current_energy += shared_Q[i * n + j];
        }
      }
    }
  }

  // Variáveis locais para rastrear o mínimo da thread
  vT local_min_energy = current_energy;
  unsigned long long local_best_state = current_state;

  // -------------------------------------------------------------------------
  // 4. Loop de Gray Code (Busca Local)
  // -------------------------------------------------------------------------
  for (unsigned long long i = 0; i < num_states_per_thread - 1; ++i) {

    // A. Descobrir qual bit flipar (Lógica Gray Code)
    int flip_bit_index_suffix = __ffsll(i + 1) - 1;
    int k = (n - 1) - flip_bit_index_suffix;

    // B. Calcular Delta de Energia
    vT delta = shared_Q[k * n + k]; // Diagonal

    for (int j = 0; j < n; ++j) {
      if (j == k)
        continue;
      // Se o vizinho j está ativo
      if ((current_state >> (n - 1 - j)) & 1) {
        delta += shared_Q[k * n + j];
      }
    }

    // C. Atualizar Estado e Energia
    auto P = n - 1 - k;
    bool is_one = (current_state >> P) & 1;

    if (is_one) {
      current_energy -= delta;       // 1 -> 0: Remove energia
      current_state &= ~(1ULL << P); // Desliga bit
    } else {
      current_energy += delta;      // 0 -> 1: Adiciona energia
      current_state |= (1ULL << P); // Liga bit
    }

    // D. Atualizar Mínimo Local
    if (current_energy < local_min_energy) {
      local_min_energy = current_energy;
      local_best_state = current_state;
    }
  }

  // -------------------------------------------------------------------------
  // 5. Redução Paralela no Bloco
  // -------------------------------------------------------------------------
  // Cada thread salva seu melhor resultado na Shared Memory
  s_energy[tid_in_block] = local_min_energy;
  s_state[tid_in_block] = local_best_state;

  __syncthreads();

  // Redução em árvore (Tree Reduction) para encontrar o mínimo do bloco
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid_in_block < s) {
      if (s_energy[tid_in_block + s] < s_energy[tid_in_block]) {
        s_energy[tid_in_block] = s_energy[tid_in_block + s];
        s_state[tid_in_block] = s_state[tid_in_block + s];
      }
    }
    __syncthreads();
  }

  // -------------------------------------------------------------------------
  // 6. Escrita Final (Por Bloco)
  // -------------------------------------------------------------------------
  // Apenas a thread 0 do bloco escreve o resultado final na memória global
  if (tid_in_block == 0) {
    global_best_energy[blockIdx.x] = s_energy[0];
    global_best_state[blockIdx.x] = s_state[0];
  }
}

// -----------------------------------------------------------------------------
// Instanciação Explícita do Template
// -----------------------------------------------------------------------------
template __global__ void
kernel_brute_force_dense<double>(const double *, int, int, unsigned long long,
                                 double *, unsigned long long *);

template __global__ void
kernel_brute_force_dense<float>(const float *, int, int, unsigned long long,
                                float *, unsigned long long *);
