#include "cuda_util.h"
#include "kernel.h"
#include "qubo_energy.h"

// Capacidade do buffer local da thread (Registradores/L1)
#define MAX_LOCAL_SOLUTIONS 16

template <typename vT>
__global__ void kernel_brute_force_dense(
    const vT *global_Q, int n, int n_fixed_bits,
    unsigned long long num_states_per_thread,
    vT *global_best_energy, // (Não usado nesta estratégia, mas mantido na
                            // interface)
    unsigned long long
        *global_best_state // (Não usado, usamos o buffer dinâmico abaixo)
    ,
    unsigned long long *found_states_buffer // BUFFER GLOBAL DE SAÍDA
    ,
    unsigned int *found_counter // CONTADOR GLOBAL
    ,
    unsigned int max_global_solutions // PROTEÇÃO
) {
  // -------------------------------------------------------------------------
  // 1. Shared Memory
  // -------------------------------------------------------------------------
  extern __shared__ char smem[];
  vT *shared_Q = reinterpret_cast<vT *>(smem);

  // Espaço para redução de energia do bloco
  vT *s_block_min = (vT *)&shared_Q[n * n];

  int tid_in_block = threadIdx.x;
  int matrix_size = n * n;

  // Carga Colaborativa
  for (int i = tid_in_block; i < matrix_size; i += blockDim.x) {
    shared_Q[i] = global_Q[i];
  }
  __syncthreads();

  // -------------------------------------------------------------------------
  // 2. Setup Inicial
  // -------------------------------------------------------------------------
  unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long current_state = tid << (n - n_fixed_bits);

  // Energia Inicial
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

  // --- SEU VETOR DE SOLUÇÕES (Local Array) ---
  vT my_best_energy = current_energy;
  unsigned long long my_solutions[MAX_LOCAL_SOLUTIONS];
  int my_count = 0;

  // Inicializa com o estado atual
  my_solutions[0] = current_state;
  my_count = 1;

  // -------------------------------------------------------------------------
  // 3. Loop Gray Code (Single Pass)
  // -------------------------------------------------------------------------
  // Tolerância para float
  vT epsilon = 1e-5;

  for (unsigned long long i = 0; i < num_states_per_thread - 1; ++i) {
    int flip_bit = (n - 1) - (__ffsll(i + 1) - 1);

    vT delta = shared_Q[flip_bit * n + flip_bit];
    for (int j = 0; j < n; ++j) {
      if (j == flip_bit)
        continue;
      if ((current_state >> (n - 1 - j)) & 1) {
        int r = (flip_bit < j) ? flip_bit : j;
        int c = (flip_bit < j) ? j : flip_bit;
        delta += shared_Q[r * n + c];
      }
    }

    if ((current_state >> (n - 1 - flip_bit)) & 1) {
      current_energy -= delta;
      current_state &= ~(1ULL << (n - 1 - flip_bit));
    } else {
      current_energy += delta;
      current_state |= (1ULL << (n - 1 - flip_bit));
    }

    // --- LÓGICA DO VETOR DE SOLUÇÕES ---
    vT diff = current_energy - my_best_energy;

    if (diff < -epsilon) {
      // ACHOU MELHOR: Zera vetor, atualiza melhor, adiciona novo
      my_best_energy = current_energy;
      my_count = 0;
      my_solutions[my_count++] = current_state;
    } else if (diff < epsilon && diff > -epsilon) {
      // ACHOU IGUAL: Adiciona ao vetor (se couber)
      if (my_count < MAX_LOCAL_SOLUTIONS) {
        my_solutions[my_count++] = current_state;
      }
    }
  }

  // -------------------------------------------------------------------------
  // 4. Redução por Bloco (Encontrar o melhor do bloco)
  // -------------------------------------------------------------------------
  // Usamos shared memory para achar o mínimo entre as 256 threads
  s_block_min[tid_in_block] = my_best_energy;
  __syncthreads();

  // Redução em árvore clássica
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid_in_block < s) {
      if (s_block_min[tid_in_block + s] < s_block_min[tid_in_block]) {
        s_block_min[tid_in_block] = s_block_min[tid_in_block + s];
      }
    }
    __syncthreads();
  }

  // O valor mínimo do bloco agora está em s_block_min[0]
  vT block_best_val = s_block_min[0];

  // -------------------------------------------------------------------------
  // 5. Escrita Condicional (Filtro)
  // -------------------------------------------------------------------------
  // Só escrevemos se a minha energia for igual à melhor do meu bloco
  vT diff_block = my_best_energy - block_best_val;
  if (diff_block < 0)
    diff_block = -diff_block;

  if (diff_block < epsilon) {
    // Sou um vencedor local! Escrevo meus estados na memória global.
    // Reservamos espaço para TODOS os meus estados de uma vez
    unsigned int start_idx = atomicAdd(found_counter, my_count);

    for (int k = 0; k < my_count; ++k) {
      if (start_idx + k < max_global_solutions) {
        found_states_buffer[start_idx + k] = my_solutions[k];
      }
    }

    // Opcional: A primeira thread vencedora poderia escrever a energia
    // em um lugar separado para a CPU saber qual é o valor minímo.
    // Mas a CPU pode recalcular a energia de qualquer estado retornado.
  }
}

// Instanciação Explícita (Observe os novos parâmetros)
template __global__ void kernel_brute_force_dense<double>(
    const double *, int, int, unsigned long long, double *,
    unsigned long long *, unsigned long long *, unsigned int *, unsigned int);
template __global__ void kernel_brute_force_dense<float>(
    const float *, int, int, unsigned long long, float *, unsigned long long *,
    unsigned long long *, unsigned int *, unsigned int);
