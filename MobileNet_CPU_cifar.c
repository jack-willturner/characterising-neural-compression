/*
	Pretrained MobileNet Convolutional Neural Network in C language and OpenMP API
	GitHUB Page: https://github.com/jcanore/vgg16
	Author: ZFTurbo/jocare

	Compilation: gcc -O3 MobileNet_CPU_cifar.c -lm -fopenmp -o MobileNet_CPU_cifar
	Usage: MobileNet_CPU_cifar <weights_path> <file_with_list_of_images> <output file> <output convolution features (optional)>
	Example: MobileNet_CPU_cifar ../../weights/weights.txt" ../../img/image_list.txt results_imagenet_conv.txt 1
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <string.h>

#include "sparse.h"

double get_seconds(struct timeval tStart, struct timeval tEnd) {
	return ((tEnd.tv_sec - tStart.tv_sec) * 1000000 + tEnd.tv_usec - tStart.tv_usec) / 1.e6;
}

#define SIZE 32
#define CONV_SIZE 3
#define CONV_LEVELS 27
//#define _CRT_SECURE_NO_WARNINGS 1

// precompile variables
// assure default values if nothing provided
#ifndef	SPARSE_CONVOLUTIONS
#define SPARSE_CONVOLUTIONS 0	// default dense convolutions
#endif	// SPARSE_CONVOLUTIONS

#ifndef FIRST_CONV_SPARSE
#define FIRST_CONV_SPARSE 0		// this is almost never 1
#endif	// FIRST_CONV_SPARSE

#ifndef	SPARSE_FULLY_CONNECTED
#define SPARSE_FULLY_CONNECTED 0	// this is not implemented yet
#endif	// SPARSE_FULLY_CONNECTED

#ifndef FISHER_PRUNING
#define FISHER_PRUNING 0	// set for fisher pruning, all previous variables changed to dense
#endif	// FISHER_PRUNING

#ifndef NUMBER_OF_THREADS
#define NUMBER_OF_THREADS 1		// number of threads to run on
//#define NUMBER_OF_THREADS omp_get_num_procs() - 1
#endif	// NUMBER_OF_THREADS

/****************************************************************************************************************************/

int im_sizes[27] = { 32, 32, 16, 16, 16, 16, 8, 8, 8, 8, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2 };
int strides[26]  = { 1,   2,  1,  1,  1,  2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1 };


int mem_block_shape[3] = { 1024, 32, 32 }; // allocate the absolute maximum amount of space we will need
float ***block1;
float ***block2;

float *****wc;	// weights convolution

float ***wd;	// weights dense
float **bd;	// biases dense

float **batchnorm_weights;
float **batchnorm_biases;
float **batchnorm_means; // running mean and variance from training used to estimate population statistics
float **batchnorm_vars;

int mem_block_dense_shape = { 1024 * 2 * 2 };  // size of output from last convolutional layer
float *mem_block1_dense;
float *mem_block2_dense;

#if SPARSE_CONVOLUTIONS
	// sparse conv
	csr_t ****wc_sparse;

#endif	// SPARSE_CONVOLUTIONS


#if FISHER_PRUNING
#define SPARSE_CONVOLUTIONS 0	// force dense convolutions

/* // ORIGINAL FISHER EXPERIMENTS
	int cshape[27][4] = {
		{ 32,  3, CONV_SIZE, CONV_SIZE },
		{ 32,  1, CONV_SIZE, CONV_SIZE },
	  { 43, 32, 1, 1 },
		{ 43,  1, CONV_SIZE, CONV_SIZE },
	  { 85, 43, 1, 1 },
		{ 85, 1, CONV_SIZE, CONV_SIZE },
	  { 70, 85, 1, 1 },
		{ 70, 1, CONV_SIZE, CONV_SIZE },
	  { 150, 70, 1, 1 },
		{ 150, 1, CONV_SIZE, CONV_SIZE },
	  { 69, 150, 1, 1 },
		{ 69, 1, CONV_SIZE, CONV_SIZE },
	  { 188, 69, 1, 1 },
		{ 188, 1, CONV_SIZE, CONV_SIZE },
	  { 72, 188, 1, 1 },
		{ 72, 1, CONV_SIZE, CONV_SIZE },
	  { 122, 72, 1, 1 },
		{ 122, 1, CONV_SIZE, CONV_SIZE },
	  { 106, 122, 1, 1 },
		{ 106, 1, CONV_SIZE, CONV_SIZE },
	  { 96, 106, 1, 1 },
		{ 96, 1, CONV_SIZE, CONV_SIZE },
	  { 81, 96, 1, 1 },
		{ 81, 1, CONV_SIZE, CONV_SIZE },
	  { 75, 81, 1, 1 },
		{ 75, 1, CONV_SIZE, CONV_SIZE },
	  { 100, 75, 1, 1 }

	};
	int dshape[1][2]= {
	   { 100, 10}
	};
*/

// FIXED 90% ACCURACY EXPERIMENTS
int cshape[27][4] = {
	{ 32,  3, CONV_SIZE, CONV_SIZE },
	{ 32,  1, CONV_SIZE, CONV_SIZE },
	{ 43, 32, 1, 1 },
	{ 43,  1, CONV_SIZE, CONV_SIZE },
	{ 85, 43, 1, 1 },
	{ 85, 1, CONV_SIZE, CONV_SIZE },
	{ 70, 85, 1, 1 },
	{ 70, 1, CONV_SIZE, CONV_SIZE },
	{ 150, 70, 1, 1 },
	{ 150, 1, CONV_SIZE, CONV_SIZE },
	{ 69, 150, 1, 1 },
	{ 69, 1, CONV_SIZE, CONV_SIZE },
	{ 188, 69, 1, 1 },
	{ 188, 1, CONV_SIZE, CONV_SIZE },
	{ 72, 188, 1, 1 },
	{ 72, 1, CONV_SIZE, CONV_SIZE },
	{ 122, 72, 1, 1 },
	{ 122, 1, CONV_SIZE, CONV_SIZE },
	{ 106, 122, 1, 1 },
	{ 106, 1, CONV_SIZE, CONV_SIZE },
	{ 96, 106, 1, 1 },
	{ 96, 1, CONV_SIZE, CONV_SIZE },
	{ 81, 96, 1, 1 },
	{ 81, 1, CONV_SIZE, CONV_SIZE },
	{ 75, 81, 1, 1 },
	{ 75, 1, CONV_SIZE, CONV_SIZE },
	{ 100, 75, 1, 1 }

};
int dshape[1][2]= {
	 { 100, 10}
};


#else // PLAIN

	int cshape[27][4] = {
		{ 32,  3, CONV_SIZE, CONV_SIZE },
		{ 32,  1, CONV_SIZE, CONV_SIZE },
	  { 64, 32, 1, 1 },
		{ 64,  1, CONV_SIZE, CONV_SIZE },
	  { 128, 64, 1, 1 },
		{ 128, 1, CONV_SIZE, CONV_SIZE },
	  { 128, 128, 1, 1 },
		{ 128, 1, CONV_SIZE, CONV_SIZE },
	  { 256, 128, 1, 1 },
		{ 256, 1, CONV_SIZE, CONV_SIZE },
	  { 256, 256, 1, 1 },
		{ 256, 1, CONV_SIZE, CONV_SIZE },
	  { 512, 256, 1, 1 },
		{ 512, 1, CONV_SIZE, CONV_SIZE },
	  { 512, 512, 1, 1 },
		{ 512, 1, CONV_SIZE, CONV_SIZE },
	  { 512, 512, 1, 1 },
		{ 512, 1, CONV_SIZE, CONV_SIZE },
	  { 512, 512, 1, 1 },
		{ 512, 1, CONV_SIZE, CONV_SIZE },
	  { 512, 512, 1, 1 },
		{ 512, 1, CONV_SIZE, CONV_SIZE },
	  { 512, 512, 1, 1 },
		{ 512, 1, CONV_SIZE, CONV_SIZE },
	  { 1024, 512, 1, 1 },
		{ 1024, 1, CONV_SIZE, CONV_SIZE },
	  { 1024, 1024, 1, 1 }
	};

	int dshape[1][2]= {
	   { 1024, 10}
	};

#endif	// FISHER_PRUNING

/****************************************************************************************************************************/

void reset_mem_block(float ***mem) {
	int i, j, k;
	for (i = 0; i < mem_block_shape[0]; i++) {
		for (j = 0; j < mem_block_shape[1]; j++) {
			for (k = 0; k < mem_block_shape[2]; k++) {
				mem[i][j][k] = 0.0;
			}
		}
	}
}

/****************************************************************************************************************************/

void reset_mem_block_dense(float *mem) {
	int i;
	for (i = 0; i < mem_block_dense_shape; i++) {
		mem[i] = 0.0;
	}
}

/****************************************************************************************************************************/

void init_memory() {

	int i, j, k, l;

	int max_channels = 1024;
	int max_im_size  = 32;

	block1 = malloc(max_channels * sizeof(float**));
	block2 = malloc(max_channels * sizeof(float**));

	// allocate block memory
	for(i = 0; i < max_channels; i++) {
		block1[i] = malloc(max_im_size * sizeof(float*));
		block2[i] = malloc(max_im_size * sizeof(float*));
		for(j = 0; j < max_im_size; j++) {
			block1[i][j] = malloc(max_im_size * sizeof(float));
			block2[i][j] = malloc(max_im_size * sizeof(float));
		}
	}

	#if SPARSE_CONVOLUTIONS
	wc_sparse = (csr_t****) malloc(CONV_LEVELS * sizeof(csr_t***));

	for (l = 0; l < CONV_LEVELS; l++) {

		wc_sparse[l] = (csr_t***) malloc(cshape[l][0] * sizeof(csr_t**));

		for (i = 0; i < cshape[l][0]; i++) {
			wc_sparse[l][i] = (csr_t**) malloc(cshape[l][1] * sizeof(csr_t*));
		}
	}

	// wc memory allocated below will be freed in read_weights if SPARSE_CONVOLUTIONS

	#endif	// SPARSE_CONVOLUTIONS


	wc = malloc(CONV_LEVELS * sizeof(float****));

	// allocate kernel memory
	for(l = 0; l < CONV_LEVELS; l++) {
		wc[l] = malloc(cshape[l][0] * sizeof(float***));

		for(i = 0; i < cshape[l][0]; i++) {
			wc[l][i] = malloc(cshape[l][1] * sizeof(float**));

			for(j = 0; j < cshape[l][1]; j++) {
				wc[l][i][j] = malloc(cshape[l][2] * sizeof(float*));
				for(k = 0; k < cshape[l][2]; k++) {
					wc[l][i][j][k] = malloc(cshape[l][3]* sizeof(float));
				}
			}
		}
	}

	// allocate batchnorm memory

	batchnorm_weights =  malloc(27 * sizeof(float*));
	batchnorm_biases  =  malloc(27 * sizeof(float*));
	batchnorm_means   =  malloc(27 * sizeof(float*));
	batchnorm_vars    =  malloc(27 * sizeof(float*));

	for (l = 0; l < CONV_LEVELS; l++) {
	batchnorm_weights[l] = malloc(cshape[l][0] * sizeof(float));
	batchnorm_biases[l]  = malloc(cshape[l][0] * sizeof(float));
	batchnorm_means[l]   = malloc(cshape[l][0] * sizeof(float));
	batchnorm_vars[l]    = malloc(cshape[l][0] * sizeof(float));
	}

	wd = malloc(1 * sizeof(float**));
	bd = malloc(1 * sizeof(float*));

	for (l = 0; l < 1; l++) {
		wd[l] = malloc(dshape[l][0] * sizeof(float*));
		for (i = 0; i < dshape[l][0]; i++) {
			wd[l][i] = malloc(dshape[l][1] * sizeof(float));
		}
		bd[l] = malloc(dshape[l][1] * sizeof(float));
	}

	// allocate dense memory
	mem_block1_dense = calloc(mem_block_dense_shape, sizeof(float));
	mem_block2_dense = calloc(mem_block_dense_shape, sizeof(float));
}

/****************************************************************************************************************************/

void free_memory() {

	int i, j, k, l;

	// Free convolution weights
	for (l = 0; l < CONV_LEVELS; l++) {

#if SPARSE_CONVOLUTIONS
		for (i = 0; i < cshape[l][0]; i++) {
			for (j = 0; j < cshape[l][1]; j++) {
				free(wc_sparse[l][i][j]);
			}
			free(wc_sparse[l][i]);
		}
		free(wc_sparse[l]);
#else
		for (i = 0; i < cshape[l][0]; i++) {
			for (j = 0; j < cshape[l][1]; j++) {
				for (k = 0; k < cshape[l][2]; k++) {
					free(wc[l][i][j][k]);
				}
				free(wc[l][i][j]);
			}
			free(wc[l][i]);
		}
		free(wc[l]);
#endif
	}
//	free(wc);
//	free(bc);

	#if SPARSE_CONVOLUTIONS
		free(wc_sparse);
	#else
		free(wc);
	#endif	// SPARSE_CONVOLUTIONS

	// Free dense weights
	for (l = 0; l < 1; l++) {
		for (i = 0; i < dshape[l][0]; i++) {
			free(wd[l][i]);
		}
		free(wd[l]);
		free(bd[l]);
	}
	free(wd);
	free(bd);

	// Free memblocks
	for (i = 0; i < mem_block_shape[0]; i++) {
		for (j = 0; j < mem_block_shape[1]; j++) {
			free(block1[i][j]);
			free(block2[i][j]);
		}
		free(block1[i]);
		free(block2[i]);
	}
	free(block1);
	free(block2);

	free(mem_block1_dense);
	free(mem_block2_dense);
}

/****************************************************************************************************************************/

void read_weights(char *in_file, int lvls) {

	float dval;
	int i, j, k, l, m, z;
	FILE *iin;
	int total_lvls_read = 0;

// 	printf("\nin_file es: %s\n\n", in_file);

	iin = fopen64(in_file, "r");
	if (iin == NULL) {
		printf("Weights file %s absent\n", in_file);
		exit(1);
	}

	// Reading convolution weights (store them flipped from begining)
	// no biases

	for (l = 0; l < CONV_LEVELS; l++) {
		printf("Read conv block %d weights\n", l);
		for (i = 0; i < cshape[l][0]; i++) {
			for (j = 0; j < cshape[l][1]; j++) {
				for (k = 0; k < cshape[l][2]; k++) {
					for (m = 0; m < cshape[l][3]; m++) {
						fscanf(iin, "%f", &dval);
						wc[l][i][j][k][m] = dval;
					}
				}
			}
		}
		total_lvls_read += 1;
	}

	for (z = 0; z < CONV_LEVELS; z++) {
		// batchnorm weights and biases
		printf("Read batchnorm block %d weights\n", z);
		for (i = 0; i < cshape[z][0]; i++) {
				fscanf(iin, "%f", &dval);
				batchnorm_weights[z][i] = dval;
		}

		for (i = 0; i < cshape[z][0]; i++) {
			fscanf(iin, "%f", &dval);
			//printf("bias %i : %f \n", i, dval);
			batchnorm_biases[z][i] = dval;
		}

		for (i = 0; i < cshape[z][0]; i++) {
			fscanf(iin, "%f", &dval);
			//printf("bias %i : %f \n", i, dval);
			batchnorm_means[z][i] = dval;
		}

		for (i = 0; i < cshape[z][0]; i++) {
			fscanf(iin, "%f", &dval);
			//printf("bias %i : %f \n", i, dval);
			batchnorm_vars[z][i] = dval;
		}
	}

	if (total_lvls_read >= lvls && lvls != -1)
		return;

	// Reading dense weights
	int num_dense_layers = 1;
	for (z = 0; z < num_dense_layers; z++) {

		printf("Read dense block %d weights\n", z);
		for (i = 0; i < dshape[z][0]; i++) {
			for (j = 0; j < dshape[z][1]; j++) {
				fscanf(iin, "%f", &dval);
				//printf("weight: %i : %f \n", i, dval);
				wd[z][i][j] = dval;
			}
		}
		for (i = 0; i < dshape[z][1]; i++) {
			fscanf(iin, "%f", &dval);
			//printf("bias %i : %f \n", i, dval);
			bd[z][i] = dval;
		}
	}
	fclose(iin);


	/////////////**************** SPARSE ************/////////////////////////////

#if SPARSE_CONVOLUTIONS

	// convert to sparse format
	for (l = 0; l < CONV_LEVELS; l++)
		for (i = 0; i < cshape[l][0]; i++)
			for (j = 0; j < cshape[l][1]; j++) {
				//printf("going for %d/%d, %d/%d, %d/%d\n", l, 13, i, cshape[l][0], j, cshape[l][1]);
				csr_t* a = dense2csr2(cshape[l][2], cshape[l][3], wc[l][i][j]);
				//print_csr(a);
				wc_sparse[l][i][j] = a;
				//printf("done..%d/%d, %d/%d, %d/%d\n", l, 13, i, cshape[l][0], j, cshape[l][1]);
			}


	// Free convolution weights
#if FIRST_CONV_SPARSE == 0
	l = 0;
	// allocate new memory for first conv and copy from wc
	float *****wc_first_conv = (float*****) malloc(1 * sizeof(float****));
	wc_first_conv[l] = (float****) malloc(cshape[l][0] * sizeof(float***));
	int k1, k2;
	for (i = 0; i < cshape[l][0]; i++) {
		wc_first_conv[l][i] = (float***) malloc(cshape[l][1] * sizeof(float**));
		for (j = 0; j < cshape[l][1]; j++) {
			wc_first_conv[l][i][j] = (float**) malloc(cshape[l][2] * sizeof(float*));
			for (k1 = 0; k1 < cshape[l][2]; k1++) {
				wc_first_conv[l][i][j][k1] = (float*) malloc(cshape[l][3] * sizeof(float));
				for (k2 = 0; k2 < cshape[l][3]; k2++)
					wc_first_conv[l][i][j][k1][k2] = wc[l][i][j][k1][k2];
			}
		}
	}
#endif	// FIRST_CONV_SPARSE == 0

	// free up all dense conv layer representation
	for (l = 0; l < CONV_LEVELS; l++) {
		for (i = 0; i < cshape[l][0]; i++) {
			for (j = 0; j < cshape[l][1]; j++) {
				for (k = 0; k < cshape[l][2]; k++) {
					free(wc[l][i][j][k]);
				}
				free(wc[l][i][j]);
				}
				free(wc[l][i]);
			}
			free(wc[l]);
		}
		free(wc);

#if FIRST_CONV_SPARSE == 0
	// replace old wc pointer with the data for only first conv layer created above
	wc = wc_first_conv;
#endif	// FIRST_CONV_SPARSE == 0

#endif	// SPARSE_CONVOLUTIONS


}

/****************************************************************************************************************************/

void read_image(char *in_file) {

	int i, j, l;
	FILE *iin;
	float dval;

	iin = fopen(in_file, "r");
	if (iin == NULL) {
		printf("Image file %s absent\n", in_file);
		exit(1);
	}

	/* Reading image */
	for (i = 0; i < SIZE; i++) {
		for (j = 0; j < SIZE; j++) {
			for (l = 0; l < 3; l++) {
				fscanf(iin, "%f", &dval);
				block1[l][i][j] = dval;
			}
		}
	}
}

/****************************************************************************************************************************/

void convolution_3_x_3(float **matrix, float **kernel, float **out, int size, int stride) {

	int i, j;
	float sum;

	float zeropad[size+2][size+2];
	memset(zeropad, 0, ((size+2)*(size+2)*sizeof(float)));	// jack

	for (i = 0; i < size; i++) {
		for (j = 0; j < size; j++) {
			zeropad[i + 1][j + 1] = matrix[i][j];
		}
	}

	for (i = 0; i < size; i=i+stride) {
		for (j = 0; j < size; j=j+stride) {

			sum =
				zeropad[i    ][j    ] * kernel[0][0] +
				zeropad[i    ][j + 1] * kernel[0][1] +
				zeropad[i    ][j + 2] * kernel[0][2] +

				zeropad[i + 1][j    ] * kernel[1][0] +
				zeropad[i + 1][j + 1] * kernel[1][1] +
				zeropad[i + 1][j + 2] * kernel[1][2] +

				zeropad[i + 2][j    ] * kernel[2][0] +
				zeropad[i + 2][j + 1] * kernel[2][1] +
				zeropad[i + 2][j + 2] * kernel[2][2];
			out[i][j] += sum;
		}
	}
}

/****************************************************************************************************************************/

void convolution_3_x_3_sparse(float **matrix, csr_t* kernel, float **out, int size, int stride) {

//	printf("sparse\n");
	int i, j;
//	float zeropad[SIZE + 2][SIZE + 2] = { 0.0 };
	float zeropad[size+2][size+2];
	memset(zeropad, 0, ((size+2)*(size+2)*sizeof(float)));	// jack

	for (i = 0; i < size; i++) {
		for (j = 0; j < size; j++) {
			zeropad[i + 1][j + 1] = matrix[i][j];
		}
	}


//	float** zeropad = (float**) malloc((size + 2) * sizeof(float*)); //[size+2][size+2];
//	for (i = 0; i < (size + 2); ++i)
//		zeropad[i] = (float*) malloc ((size + 2) * sizeof(float));

//	//memset(zeropad, 0, ((size+2)*(size+2)*sizeof(float)));
//	// padding with zeros
//	for (i = 0; i < size + 2; ++i) {
//		zeropad[i][0] = 0;
//		zeropad[i][size + 1] = 0;
//	}
//	for (i = 1; i < size + 1; ++i) {
//		zeropad[0][i] = 0;
//		zeropad[size + 1][i] = 0;
//	}


//	// copying input value
//	for (i = 0; i < size; ++i) {
//		for (j = 0; j < size; ++j) {
//			zeropad[i + 1][j + 1] = matrix[i][j];
//		}
//	}

//	// convolution
//	for (i = 0; i < size; ++i) {
//		for (j = 0; j < size; ++j) {
//			out[i][j] += s_csr_conv(kernel, zeropad, i, j);
//		}
//	}

//	for (i = 0; i < (size + 2); ++i)
//		free(zeropad[i]);
//	free(zeropad);

	int k,l;
	float sum;
	// convolution
	for (i = 0; i < size; i+=stride) {
		for (j = 0; j < size; j+=stride) {
			//out[i][j] += s_csr_conv(kernel, zeropad, i, j);

			sum = 0.f;
			for (k = 0; k < kernel->nrows; ++k) {

				// for every nonzero element in this row
				for (l = kernel->rowptr[k]; l < kernel->rowptr[k + 1]; ++l) {

      				// Scale the corresponding row of B with the nonzero value of A
					float value = kernel->values[l];
					int col = kernel->colind[l];

					sum += value * zeropad[i + k][j + col];
				}
			}
			out[i][j] += sum;
		}
	}
}

/****************************************************************************************************************************/

void pointwise_convolution(float ****point_kernel, float ***block2, float ***block1, int input_channels, int output_channels, int image_size) {

	int i, j, k, l;
	float sum;

	#pragma omp parallel for private(i,j,k,l) schedule(dynamic,1) num_threads(NUMBER_OF_THREADS)
	for (i = 0; i < output_channels; i++) {
		for(j = 0; j < image_size; j++) {
			for(k = 0; k < image_size; k++) {
				sum = 0.;
				for(l = 0; l < input_channels; l++) {
					sum += block2[l][j][k] * point_kernel[i][l][0][0]; // 0 because they are always 1x1 filters
				}
				block1[i][j][k] = sum;
		    }
		}
	}
}

/****************************************************************************************************************************/

void pointwise_convolution_sparse(float **matrix, csr_t* kernel, float **out, int size) {

//	printf("sparse\n");
	int i, j;
//	float zeropad[SIZE + 2][SIZE + 2] = { 0.0 };
	float zeropad[size+2][size+2];
	memset(zeropad, 0, ((size+2)*(size+2)*sizeof(float)));	// jack

	for (i = 0; i < size; i++) {
		for (j = 0; j < size; j++) {
			zeropad[i + 1][j + 1] = matrix[i][j];
		}
	}

//	float** zeropad = (float**) malloc((size + 2) * sizeof(float*)); //[size+2][size+2];
//	for (i = 0; i < (size + 2); ++i)
//		zeropad[i] = (float*) malloc ((size + 2) * sizeof(float));

//	//memset(zeropad, 0, ((size+2)*(size+2)*sizeof(float)));
//	// padding with zeros
//	for (i = 0; i < size + 2; ++i) {
//		zeropad[i][0] = 0;
//		zeropad[i][size + 1] = 0;
//	}
//	for (i = 1; i < size + 1; ++i) {
//		zeropad[0][i] = 0;
//		zeropad[size + 1][i] = 0;
//	}


//	// copying input value
//	for (i = 0; i < size; ++i) {
//		for (j = 0; j < size; ++j) {
//			zeropad[i + 1][j + 1] = matrix[i][j];
//		}
//	}

//	// convolution
//	for (i = 0; i < size; ++i) {
//		for (j = 0; j < size; ++j) {
//			out[i][j] += s_csr_conv(kernel, zeropad, i, j);
//		}
//	}

//	for (i = 0; i < (size + 2); ++i)
//		free(zeropad[i]);
//	free(zeropad);

	int k,l;
	float sum;
	// convolution
	for (i = 0; i < size; ++i) {
		for (j = 0; j < size; ++j) {
			//out[i][j] += s_csr_conv(kernel, zeropad, i, j);

			sum = 0.f;
			for (k = 0; k < kernel->nrows; ++k) {

				// for every nonzero element in this row
				for (l = kernel->rowptr[k]; l < kernel->rowptr[k + 1]; ++l) {

      				// Scale the corresponding row of B with the nonzero value of A
					float value = kernel->values[l];
					int col = kernel->colind[l];

					sum += value * zeropad[i + k][j + col];
				}
			}
			out[i][j] += sum;
		}
	}
}

/****************************************************************************************************************************/

void batchnorm_and_relu(float ***in, float ***out, float *weights, float *bias, float *mean, float *var, int num_channels, int image_size) {

	int channel, i, j;
	// ((x - mean) * invstd) * w + b

	#pragma omp parallel for private(channel,i,j) schedule(dynamic,1) num_threads(NUMBER_OF_THREADS)
	for(channel = 0; channel < num_channels; channel++) {
		float invstd = 1. / sqrt(var[channel] + 0.000001);

		for(i = 0; i < image_size; i++) {
			for(j = 0; j < image_size; j++) {
				out[channel][i][j] = (weights[channel] * invstd ) * in[channel][i][j] + (bias[channel] - ((weights[channel] * mean[channel]) * invstd));
				//out[channel][i][j] = ((in[channel][i][j] - mean[channel]) * invstd) * weights[channel] + bias[channel];
				if (out[channel][i][j] < 0.f)
					out[channel][i][j] = 0.f;
			}
		}
	}
}

/****************************************************************************************************************************/

void depthwise_convolution(float ***block1, float ***block2, float ****depth_kernel, float ****point_kernel, int level) {

	int i, j;
	int input_channels  = cshape[level][0];
	int output_channels = cshape[level+1][0];

	//printf("level %i: %i ==> %i\n", level, input_channels, output_channels);

	#pragma omp parallel for private(i) schedule(dynamic,1) num_threads(NUMBER_OF_THREADS)
	for(i = 0; i < input_channels; i++) {
	    #if SPARSE_CONVOLUTIONS
		    convolution_3_x_3_sparse(block1[i], wc_sparse[level][i][0], block2[i], im_sizes[level], strides[level]);
	    #else
		    convolution_3_x_3(block1[i], depth_kernel[i][0], block2[i], im_sizes[level], strides[level]);
	    #endif
	}
	batchnorm_and_relu(block2, block1, batchnorm_weights[level], batchnorm_biases[level], batchnorm_means[level], batchnorm_vars[level], input_channels, im_sizes[level+1]);
	reset_mem_block(block2);

	level++;

	// now do linear combination of the elements in output and write them back into the first memory block
#if SPARSE_CONVOLUTIONS
	#pragma omp parallel for private(i,j) schedule(dynamic,1) num_threads(NUMBER_OF_THREADS)
	for(i = 0; i < output_channels; i++) {
		for(j = 0; j < input_channels; j++) {
			pointwise_convolution_sparse(block2[j], wc_sparse[level][i][j], block1[j], im_sizes[level] );
		}
	}

#else
	pointwise_convolution(point_kernel, block1, block2, input_channels, output_channels, im_sizes[level]);
#endif

	batchnorm_and_relu(block2, block1, batchnorm_weights[level], batchnorm_biases[level], batchnorm_means[level], batchnorm_vars[level], output_channels, im_sizes[level+1]);
	reset_mem_block(block2);
}

/****************************************************************************************************************************/

void add_bias_and_relu_flatten(float *out, float *bs, int size, int relu) {

	int i;

	for (i = 0; i < size; i++) {
		out[i] += bs[i];
// 		printf("%f\n", out[i]);
		if (relu == 1) {
			if (out[i] < 0)
				out[i] = 0.f;
		}
	}
}

/****************************************************************************************************************************/

void flatten(float ***in, float *out, int sh0, int sh1, int sh2) {

	int i, j, k, total = 0;

	for (i = 0; i < sh0; i++) {
		for (j = 0; j < sh1; j++) {
			for (k = 0; k < sh2; k++) {
				out[total] = in[i][j][k];
				total += 1;
			}
		}
	}
}

/****************************************************************************************************************************/

void dense(float *in, float **weights, float *out, int sh_in, int sh_out) {

	int i, j;

	#pragma omp parallel for private(i, j) schedule(dynamic,1) num_threads(NUMBER_OF_THREADS)
	for (i = 0; i < sh_out; i++) {
		float sum = 0.0;
		for (j = 0; j < sh_in; j++) {
			sum += in[j] * weights[j][i];
		}
		out[i] = sum;
	}
}

/****************************************************************************************************************************/

void write_out_block(int layer, float ***block) {

	int layer_name = layer;// * 2 - 1;
	char filename[16];
	sprintf(filename, "outputs/output%d", layer_name);

	FILE *f = fopen(filename, "w");

	if (f == NULL){
		printf("Error opening file!\n");
		exit(1);
	}

	for(int i = 0; i < 32; i++) {
		for(int j = 0; j < mem_block_shape[1]; j++) {
			for(int k = 0; k < mem_block_shape[2]; k++) {
				fprintf(f, "%f \n", block[i][j][k]);
			}
		}
	}

	fclose(f);
}

/****************************************************************************************************************************/

void write_out_layer(int layer) {

	int layer_name = layer;// * 2 - 1;
	char filename[7];
	sprintf(filename, "layer%d", layer_name);

	FILE *f = fopen(filename, "w");

	int depth = 1;

	if (f == NULL){
		printf("Error opening file!\n");
		exit(1);
	}

	for(int o = 0; o < cshape[layer][0]; o++) {
		for(int i = 0; i < cshape[layer][1]; i++) {
			for(int k_h = 0; k_h < cshape[layer][2]; k_h++) {
				for(int k_w = 0; k_w < cshape[layer][3]; k_w++) {
					fprintf(f, "%f ", wc[layer][o][i][k_h][k_w]);
				}
			}
			fprintf(f, "\n");
		}
	}

	fclose(f);

	layer_name = layer + 1;
	char filename2[7];
	sprintf(filename2, "layer%d", layer_name);
	// get batchnorms
	FILE *f2 = fopen(filename2, "w");

	if (f2 == NULL){
		printf("Error opening file!\n");
		exit(1);
	}

	for(int i = 0; i < cshape[layer][0]; i++) {
			fprintf(f2, "%f \n", batchnorm_weights[layer][i]);
	}
	fprintf(f2, "\n\n\n");

	for(int i = 0; i < cshape[layer][0]; i++) {
			fprintf(f2, "%f \n", batchnorm_biases[layer][i]);
	}
	fprintf(f2, "\n\n\n");

	for(int i = 0; i < cshape[layer][0]; i++) {
			fprintf(f2, "%f \n", batchnorm_means[layer][i]);
	}
	fprintf(f2, "\n\n\n");
	for(int i = 0; i < cshape[layer][0]; i++) {
			fprintf(f2, "%f \n", batchnorm_vars[layer][i]);
	}

	fclose(f);
}

/****************************************************************************************************************************/

void output_predictions(FILE *out, int only_convolution, int size, int cur_size) {

	int i;
	int c=0;

	if (only_convolution == 1) {
		//for (i = 0; i < 512*7*7; i++) {
		for (i = 0; i < size * cur_size * cur_size; i++) {
		    fprintf(out, "%g\n", mem_block1_dense[i]);
		}
	}
	else {
		double maximum=-1;

			    // dshape[0][1] ==> 10
		for (i = 0; i < dshape[0][1]; i++) {
				    fprintf(out, "%g\n", mem_block2_dense[i]);

				    if(mem_block1_dense[i]>maximum){
					maximum=mem_block2_dense[i];
					c=i+1;
				    }
		}

		fprintf(out, "\n");
		printf("This image depicts class: %d\n",c);
	}
}

/****************************************************************************************************************************/

void get_mobilenet_predict() {

	int level = 0;
	int i, j;

	// normal convolution
	#pragma omp parallel for private(i, j) schedule(dynamic,1) num_threads(NUMBER_OF_THREADS)
	for (i = 0; i < cshape[level][0]; i++) {
		for (j = 0; j < cshape[level][1]; j++) {
				#if FIRST_CONV_SPARSE
					convolution_3_x_3_sparse(block1[j], wc_sparse[level][i][j], block2[i], im_sizes[level], 1);
				#else
					convolution_3_x_3(block1[j], wc[level][i][j], block2[i], im_sizes[level], 1);
				#endif
		}
	}

	batchnorm_and_relu(block2, block1, batchnorm_weights[level], batchnorm_biases[level], batchnorm_means[level], batchnorm_vars[level], 32, 32);
	reset_mem_block(block2);

	// depthwise convolutions
	for(level = 1; level < (CONV_LEVELS - 1); level=level+2) {
		depthwise_convolution(block1, block2, wc[level], wc[level+1], (level));
	}

	// flatten
	flatten(block1, mem_block1_dense, cshape[level][0], im_sizes[level], im_sizes[level]);

	// dense
	level = 0;
	dense(mem_block1_dense, wd[level], mem_block2_dense, dshape[level][0], dshape[level][1]);
	add_bias_and_relu_flatten(mem_block2_dense, bd[level], dshape[level][1], 0);
	reset_mem_block_dense(mem_block1_dense);

	return;
}

/****************************************************************************************************************************/

char *trimwhitespace(char *str) {

	char *end;

	// Trim leading space
	while (isspace((unsigned char)*str)) str++;

	if (*str == 0)  // All spaces?
		return str;

	// Trim trailing space
	end = str + strlen(str) - 1;
	while (end > str && isspace((unsigned char)*end)) end--;

	// Write new null terminator
	*(end + 1) = 0;

	return str;
}

/****************************************************************************************************************************/

int main(int argc, char *argv[]) {

	FILE *file_list, *results;
	char buf[1024];
	struct timeval tStart, tEnd;

	double deltaTime;
	char *weights_file;
	char *image_list_file;
	char *output_file;
	int lvls = -1;
	int only_convolution = 0;

	//-----------------------------------------------------------------------

	printf("Using %d threads\n", NUMBER_OF_THREADS);

	if (argc != 4 && argc != 5) {
		printf("Usage: <program.exe> <weights file> <images list file> <output file> <only_convolution [optional]>\n");
		return 0;
	}

	weights_file = argv[1];

	//printf("%s\n", weights_file);
	image_list_file = argv[2];
	output_file = argv[3];

	if (argc == 5) {
		lvls = 20;
		only_convolution = 1;
	}

	//-----------------------------------------------------------------------

	init_memory();

	file_list = fopen(image_list_file, "r");

	if (file_list == NULL) {
		printf("Check file list location: %s\n", image_list_file);
		return 1;
	}

	results = fopen(output_file, "w");
	if (results == NULL) {
		printf("Couldn't open file for writing: %s\n", output_file);
		return 1;
	}

	gettimeofday(&tStart, NULL);
	read_weights(weights_file, lvls);
	gettimeofday(&tEnd, NULL);
	deltaTime = get_seconds(tStart, tEnd);
	printf("Reading weights: %.3lf sec\n", deltaTime);

	while (!feof(file_list)) {

		fgets(buf, 1024, file_list);
		if (strlen(buf) == 0) {
			break;
		}
// 		printf("%d\n", strlen(buf));
		read_image(trimwhitespace(buf));

		gettimeofday(&tStart, NULL);
		get_mobilenet_predict(results, only_convolution);

		gettimeofday(&tEnd, NULL);
		deltaTime = get_seconds(tStart, tEnd);
		printf("Infer image %s: %.3lf sec\n", buf, deltaTime);

		output_predictions(results, only_convolution, 1024, 1);
	}

	//free_memory();
	fclose(file_list);

	return 0;
}
