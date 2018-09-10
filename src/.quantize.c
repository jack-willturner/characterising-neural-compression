#include "quantize.h"

void *csr2csr_q(const csr_t *csr, csr_q_t *csr_q) {

  csr_q->poz = 0;
  csr_q->neg = 0;

  int idx = 0;
  while (!csr_q->poz || !csr_q->neg) {
    if (csr_q->values[idx] > 0)
      csr_q->poz = csr_q->values[idx];
    else
      csr_q->neg = csr_q->values[idx];
  }

  csr_q->b_values = (unsigned char*)(malloc)(csr_q->nnz / 8 * sizeof(unsigned char));

  idx = 0;
  for (int i = 0; i < csr->nnz; ++i) {
    if (csr->values[i] == csr_q->poz)
      b_values[i] |= 1 << i % 8;

    if (i % 8 == 0)
      idx++;


  }

}




//////////////////////////////// CSR - operations


/*
  Sparse Matrix - Dense vector routines

  A is a M x K sparse matrix
  x is a K x 1 dense vector
  y is a M x 1 dense vector
*/
void q_csrmv(const csr_t *A, const float *x, float *y) {
  for (int i = 0; i < A->nrows; ++i) {
    // For every nonzero element in this row
    y[i] = 0.0f;
    for (int j = A->rowptr[i]; j < A->rowptr[i + 1]; ++j)
      y[i] += A->values[j] * x[A->colind[j]];
  }
}

/*
  Sparse Matrix - Dense matrix routines

  A is a M x K sparse matrix
  B is a K x N dense matrix
  C is a M x N dense matrix
*/
void q_csrmm(const int M, const int N, const int K, 
            const csr_t *A, const float *B, float *C) {
  memset(C, 0, M * N * sizeof(float));
  for (int i = 0; i < A->nrows; ++i) {

    // For every nonzero element in this row
    for (int j = A->rowptr[i]; j < A->rowptr[i + 1]; ++j) {

      // Scale the corresponding row of B with the nonzero value of A
      register float value = A->values[j];
      register int Brow = A->colind[j];

      for (int k = 0; k < N; ++k) {
        C[i * N + k] += value * B[Brow * N + k];
      }

    }
  }
}