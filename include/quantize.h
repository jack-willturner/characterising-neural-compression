#ifndef QUANTIZE_H
#define QUANTIZE_H

#include <stddef.h>
#include <sparse.h>

// Compressed Sparse Row Quantized (CSR-Q) matrix format
typedef struct {
  int nrows, ncols, nnz;
  float poz, neg;
  int *rowptr;
  int *colind;
  float *values;
  unsigned char* b_values;

} csr_q_t;

size_t csr_q_size(const csr_q_t *mat);
void csr_free(csr_q_t *mat);

csr_q_t *csr2csr_q(const csr_t *csr, csr_q_t *csr_q);

void q_csrmv(const csr_q_t *A, const float *x, float *y);
void q_csrmm(const int M, const int N, const int K, 
            const csr_q_t *A, const float *B, float *C);


#endif	// QUANTIZE_H