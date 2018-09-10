// Sparse matrix storage formats and operations
//
//  Valentin - 25.04.2018
//
// - the Coordinate (COO) format,
// - the Compressed Sparse Row (CSR) format and
// - the Compressed Sparse Column (CSC) format.


#ifndef SPARSE_H
#define SPARSE_H

#include <stddef.h>

// Sparse matrix formats
typedef enum {COO, CSR} SparseMatFormat;

// Coordinate (COO) matrix format
typedef struct {
  size_t nrows, ncols, nnz;
  unsigned *rowind;
  unsigned *colind;
  float *values;
} coo_t;

size_t coo_size(const coo_t *mat);
void coo_free(coo_t *mat);

// Compressed Sparse Row (CSR) matrix format
typedef struct {
  int nrows, ncols, nnz;
  int *rowptr;
  int *colind;
  float *values;
} csr_t;

size_t csr_size(const csr_t *mat);
void csr_free(csr_t *mat);

// Compressed Sparse Column (CSC) matrix format
typedef struct {
  int nrows, ncols, nnz;
  int *rowind;
  int *colptr;
  float *values;
} csc_t;

size_t csc_size(const csc_t *mat);
void csc_free(csc_t *mat);

// Converts a matrix stored in a dense format to the COO sparse matrix format
coo_t *dense2coo(int nrows, int ncols, const float *data);
coo_t *dense2coo2(int nrows, int ncols, const float **data);
void print_coo(const coo_t *coo);

// Converts a matrix stored in the COO format to the CSR format
csr_t *coo2csr(const coo_t *mat);
csr_t *dense2csr(int nrows, int ncols, const float *data);
csr_t *dense2csr2(int nrows, int ncols, const float **data);
void print_csr(const csr_t *csr);

// Converts a matrix stored in the COO format to the CSC format
csc_t *coo2csc(const coo_t *mat);
csc_t *dense2csc(int nrows, int ncols, const float *data);
csc_t *dense2csc2(int nrows, int ncols, const float **data);
void print_csc(const csc_t *csc);


///// Operations

void s_csrmv(const csr_t *A, const float *x, float *y);
void s_csrmm(const int M, const int N, const int K, 
            const csr_t *A, const float *B, float *C);

float s_csr_conv(const csr_t *A, const float **B, const int start_i, const int start_j);

#endif  // SPARSE_H