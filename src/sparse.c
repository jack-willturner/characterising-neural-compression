// Sparse matrix storage formats and operations
//
//  Valentin - 25.04.2018
//
// - the Coordinate (COO) format,
// - the Compressed Sparse Row (CSR) format and
// - the Compressed Sparse Column (CSC) format.


#include <malloc.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include "sparse.h"

static const float ZERO_THRESHOLD = 0.000001f;

static size_t count_nnz(const float *data, size_t size) {
  size_t cnt = 0;
  size_t i;
  for (i = 0; i < size; ++i) {
    //if (data[i] != 0.f) {
    if (data[i] > ZERO_THRESHOLD || data[i] < -ZERO_THRESHOLD) {
      ++cnt;
    }
  }

  return cnt;
}


///////// COO

size_t coo_size(const coo_t *coo) {
  return 2 * coo->nnz * sizeof(unsigned) + coo->nnz * sizeof(float);
}

void coo_free(coo_t *coo) {
  free(coo->rowind);
  free(coo->colind);
  free(coo->values);
  free(coo);
}


// provides a coo_t format from dense representation
coo_t *dense2coo(int nrows, int ncols, const float *data) {
  coo_t *coo = malloc(sizeof(coo_t));
  coo->nrows = nrows;
  coo->ncols = ncols;
  coo->nnz = count_nnz(data, coo->nrows * coo->ncols);
  coo->rowind = malloc(coo->nnz * sizeof(unsigned));
  coo->colind = malloc(coo->nnz * sizeof(unsigned));
  coo->values = malloc(coo->nnz * sizeof(float));

  int cnt = 0;
  int i, j;
  for (i = 0; i < coo->nrows; ++i) {
    for (j = 0; j < coo->ncols; ++j) {
      int idx = i * coo->ncols + j;
      //if (data[idx] != 0.f) {
      if (data[idx] > ZERO_THRESHOLD || data[idx] < -ZERO_THRESHOLD) {
        coo->rowind[cnt] = i;
        coo->colind[cnt] = j;
        coo->values[cnt] = data[idx];
        ++cnt;
      }
    }
  }

  assert(cnt == coo->nnz);

  return coo;
}

// overload conversion function for data being passed as a double pointer
coo_t *dense2coo2(int nrows, int ncols, const float **data) {
  
  // simply convert the double pointer matrix into a single pointer
  float *tmp = (float*)malloc(nrows * ncols * sizeof(float));

  int i, j;
  for (i = 0; i < nrows; ++i)
    for (j = 0; j < ncols; ++j) {
      unsigned idx = i * ncols + j;
      tmp[idx] = data[i][j];
    }
  
  coo_t *coo = dense2coo(nrows, ncols, tmp);
  free(tmp);
  return coo;
}

void print_coo(const coo_t *coo) {
  printf("COO format..\nrowind: [");
  int i;
  for (i = 0; i < coo->nnz; ++i) 
    printf("%d, ", coo->rowind[i]);
  printf("];\n");

  printf("colind: [");
  for (i = 0; i < coo->nnz; ++i) 
    printf("%d, ", coo->colind[i]);
  printf("];\n");

  printf("values: [");
  for (i = 0; i < coo->nnz; ++i) 
    printf("%f, ", coo->values[i]);
  printf("];\n");

}

///////// CSR

size_t csr_size(const csr_t *csr) {
  return ((csr->nrows + 1 + csr->nnz) * sizeof(int) + csr->nnz * sizeof(float));
}


void csr_free(csr_t *csr) {
  free(csr->rowptr);
  free(csr->colind);
  free(csr->values);
  free(csr);
}


csr_t *coo2csr(const coo_t *coo) {
  int row, val_i = 0, row_i = 0, row_prev = 0, i, j;
  csr_t *csr = malloc(sizeof(csr_t));
  csr->nrows = coo->nrows;
  csr->ncols = coo->ncols;
  csr->nnz = coo->nnz;
  csr->rowptr = malloc((csr->nrows + 1) * sizeof(int));
  csr->colind = malloc(csr->nnz * sizeof(int));
  csr->values = malloc(csr->nnz * sizeof(float));

  csr->rowptr[row_i++] = 0;
  for (i = 0; i < coo->nnz; ++i) {
    row = coo->rowind[i];
    assert(row >= row_prev);
    assert(row < coo->nrows);
    
    if (row != row_prev) {
      for (j = 0; j < row - row_prev; j++) {
        csr->rowptr[row_i++] = val_i;
      }

      row_prev = row;
    }

    val_i++;
  }
  while (row_i < csr->nrows) {
    csr->rowptr[row_i++] = val_i;
  }

  csr->rowptr[row_i] = val_i;
  assert(row_i == csr->nrows);
  assert(csr->rowptr[csr->nrows] == csr->nnz);
  memcpy(csr->colind, coo->colind, csr->nnz * sizeof(int));
  memcpy(csr->values, coo->values, csr->nnz * sizeof(float));

  return csr;
}

csr_t *dense2csr(int nrows, int ncols, const float *data) {
  coo_t *coo = dense2coo(nrows, ncols, data);
  csr_t *csr = coo2csr(coo);
  coo_free(coo);
  return csr;
}

csr_t *dense2csr2(int nrows, int ncols, const float **data) {
  coo_t *coo = dense2coo2(nrows, ncols, data);
  csr_t *csr = coo2csr(coo);
  coo_free(coo);
  return csr;
}

void print_csr(const csr_t *csr) {
  printf("CSR format..\nrow_offset: [");
  int i;
  for (i = 0; i < csr->nrows + 1; ++i)
    printf("%d, ", csr->rowptr[i]);
  printf("];\n");

  printf("colind: [");
  for (i = 0; i < csr->nnz; ++i)
    printf("%d, ", csr->colind[i]);
  printf("];\n");

  printf("values: [");
  for (i = 0; i < csr->nnz; ++i)
    printf("%f, ", csr->values[i]);
  printf("];\n");
}


///////// CSC

size_t csc_size(const csc_t *csc) {
  return ((csc->ncols + 1 + csc->nnz) * sizeof(int) + csc->nnz * sizeof(float));
}

void csc_free(csc_t *csc) {
  free(csc->colptr);
  free(csc->rowind);
  free(csc->values);
  free(csc);
}

// Converts a matrix stored in the COO format to the CSC format
csc_t *coo2csc(const coo_t *coo) {
  csc_t *csc = malloc(sizeof(csc_t));
  csc->nrows = coo->nrows;
  csc->ncols = coo->ncols;
  csc->nnz = coo->nnz;
  csc->colptr = malloc((csc->ncols + 1) * sizeof(int));
  csc->rowind = malloc(csc->nnz * sizeof(int));
  csc->values = malloc(csc->nnz * sizeof(float));
  memset(csc->colptr, 0, (csc->ncols + 1) * sizeof(int));

  int i;
  for (i = 0; i < coo->nnz; i++) {
    csc->colptr[coo->colind[i] + 1]++;
  }

  for (i = 1; i <= coo->ncols + 1; i++) {
    csc->colptr[i] += csc->colptr[i - 1];
  }

  int *tmp = malloc((csc->ncols + 1) * sizeof(int));
  memcpy(tmp, csc->colptr, (csc->ncols + 1) * sizeof(int));
  for (i = 0; i < csc->nnz; i++) {
    csc->values[tmp[coo->colind[i]]] = coo->values[i];
    csc->rowind[tmp[coo->colind[i]]] = coo->rowind[i];
    tmp[coo->colind[i]]++;
  }
  assert(csc->colptr[csc->ncols] == csc->nnz);
  free(tmp);

  return csc;
}

csc_t *dense2csc(int nrows, int ncols, const float *data) {
  coo_t *coo = dense2coo(nrows, ncols, data);
  csc_t *csc = coo2csc(coo);
  coo_free(coo);
  return csc;
}

csc_t *dense2csc2(int nrows, int ncols, const float **data) {
  coo_t *coo = dense2coo2(nrows, ncols, data);
  csc_t *csc = coo2csc(coo);
  coo_free(coo);
  return csc;
}

void print_csc(const csc_t *csc) {
  printf("CSC format..\nrowind: [");
  int i;
  for (i = 0; i < csc->nnz; ++i) 
    printf("%d, ", csc->rowind[i]);
  printf("];\n");

  printf("col_offset: [");
  for (i = 0; i < csc->ncols  ; ++i) 
    printf("%d, ", csc->colptr[i]);
  printf("];\n");

  printf("values: [");
  for (i = 0; i < csc->nnz; ++i) 
    printf("%f, ", csc->values[i]);
  printf("];\n");
}


//////////////////////////////// CSR - operations


/*
  Sparse Matrix - Dense vector routines

  A is a M x K sparse matrix
  x is a K x 1 dense vector
  y is a M x 1 dense vector
*/
void s_csrmv(const csr_t *A, const float *x, float *y) {
  int i, j;
  for (i = 0; i < A->nrows; ++i) {
    // For every nonzero element in this row
    y[i] = 0.0f;
    for (j = A->rowptr[i]; j < A->rowptr[i + 1]; ++j)
      y[i] += A->values[j] * x[A->colind[j]];
  }
}

/*
  Sparse Matrix - Dense matrix routines

  A is a M x K sparse matrix
  B is a K x N dense matrix
  C is a M x N dense matrix
*/
void s_csrmm(const int M, const int N, const int K, 
            const csr_t *A, const float *B, float *C) {
  memset(C, 0, M * N * sizeof(float));
  int i, j, k;
  for (i = 0; i < A->nrows; ++i) {

    // For every nonzero element in this row
    for (j = A->rowptr[i]; j < A->rowptr[i + 1]; ++j) {

      // Scale the corresponding row of B with the nonzero value of A
      register float value = A->values[j];
      register int Brow = A->colind[j];

      for (k = 0; k < N; ++k) {
        C[i * N + k] += value * B[Brow * N + k];
      }

    }
  }
}


float s_csr_conv(const csr_t *A, const float **B, const int start_i, const int start_j) {
  //float *C = (float*)malloc(M * N * sizeof(float));
  //memset(C, 0, M * N * sizeof(float));

  float sum = 0;
  int i, j;
  for (i = 0; i < A->nrows; ++i) {
    
    // for every nonzero element in this row
    for (j = A->rowptr[i]; j < A->rowptr[i + 1]; ++j) {

      // Scale the corresponding row of B with the nonzero value of A
      float value = A->values[j];
      int col = A->colind[j];

      sum += value * B[start_i + i][start_j + col];
    }
  }

  return sum;
}
