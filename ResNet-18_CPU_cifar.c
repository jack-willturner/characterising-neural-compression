/*
	Pretrained ResNet-18 Convolutional Neural Network in C language and OpenMP API
	GitHUB Page: https://github.com/jcanore/vgg16
	Author: Jack/jocare

	Compilation: gcc -O3 ResNet-18_CPU_cifar.c -lm -fopenmp -o ResNet-18_CPU_cifar
	Usage: ResNet-18_CPU_cifar <weights_path> <file_with_list_of_images> <output file> <output convolution features (optional)>
	Example: ResNet-18_CPU_cifar ../../weights/weights.txt" ../../img/image_list.txt results_imagenet_conv.txt 1
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#include "sparse.h"

double get_seconds(struct timeval tStart, struct timeval tEnd) {
	return ((tEnd.tv_sec - tStart.tv_sec) * 1000000 + tEnd.tv_usec - tStart.tv_usec) / 1.e6;
}

#define SIZE 32
#define CONV_SIZE 3
#define CONV_LEVELS 20
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

int im_sizes[20] = { 32, 32, 32, 32, 32, 16, 16, 16, 16, 8, 8, 8, 8, 8, 4, 4, 4, 4, 4 };

// Weights and image block START
float ***image;

#if FISHER_PRUNING
#define SPARSE_CONVOLUTIONS 0	// force dense convolutions

/* // ORIGINAL FISHER EXPERIMENTS
int cshape[20][4] = {
	{ 64, 3,    CONV_SIZE, CONV_SIZE },
	{ 13, 64,   CONV_SIZE, CONV_SIZE },
	{ 64, 13,   CONV_SIZE, CONV_SIZE },
	{ 11, 64,   CONV_SIZE, CONV_SIZE },
	{ 64, 11,   CONV_SIZE, CONV_SIZE },
	{ 31, 64,   CONV_SIZE, CONV_SIZE },
	{ 128, 31,  CONV_SIZE, CONV_SIZE },
	{ 31, 64,  1, 1 },
	{ 128, 128,  CONV_SIZE, CONV_SIZE },
	{ 13, 128,  CONV_SIZE, CONV_SIZE },
	{ 40, 13,   CONV_SIZE, CONV_SIZE },
	{ 256, 40,  CONV_SIZE, CONV_SIZE },
	{ 40, 13, 1, 1 },
	{ 256, 256, CONV_SIZE, CONV_SIZE },
	{ 19, 256, CONV_SIZE, CONV_SIZE },
	{ 19, 19, CONV_SIZE, CONV_SIZE },
	{ 512, 19, CONV_SIZE, CONV_SIZE },
	{ 19, 19, 1, 1 },
	{ 512, 512, CONV_SIZE, CONV_SIZE },
	{ 12, 512, CONV_SIZE, CONV_SIZE }
};

// batch normalization layer shapes
int bshape[20] = { 64, 13, 64, 11, 64, 31, 128, 128, 13, 128, 40, 256, 256, 19, 256, 19, 512, 512, 12, 512 };

// dense layer
int dshape[1][2]= {
   { 512, 10}
};

*/

// FIXED 90% ACCURACY EXPERIMENTS
int cshape[20][4] = {
	{ 64, 3,    CONV_SIZE, CONV_SIZE },
	{ 9, 64,   CONV_SIZE, CONV_SIZE },
	{ 64, 9,   CONV_SIZE, CONV_SIZE },
	{ 10, 64,   CONV_SIZE, CONV_SIZE },
	{ 64, 10,   CONV_SIZE, CONV_SIZE },
	{ 23, 64,   CONV_SIZE, CONV_SIZE },
	{ 128, 23,  CONV_SIZE, CONV_SIZE },
	{ 128, 64,  1, 1 },
	{ 7, 128,  CONV_SIZE, CONV_SIZE },
	{ 128, 7,  CONV_SIZE, CONV_SIZE },
	{ 30, 128,   CONV_SIZE, CONV_SIZE },
	{ 256, 30,  CONV_SIZE, CONV_SIZE },
	{ 256, 128, 1, 1 },
	{ 15, 256, CONV_SIZE, CONV_SIZE },
	{ 256, 15, CONV_SIZE, CONV_SIZE },
	{ 15, 256, CONV_SIZE, CONV_SIZE },
	{ 512, 15, CONV_SIZE, CONV_SIZE },
	{ 512, 256, 1, 1 },
	{ 10, 512, CONV_SIZE, CONV_SIZE },
	{ 512, 10, CONV_SIZE, CONV_SIZE }
};

// batch normalization layer shapes
int bshape[20] = { 64, 9, 64, 10, 64, 23, 128, 128, 7, 128, 30, 256, 256, 15, 256, 15, 512, 512, 10, 512 };

// dense layer
int dshape[1][2]= {
   { 512, 10}
};


#else // FISHER PRUNING

int cshape[20][4] = {
	{ 64, 3,    CONV_SIZE, CONV_SIZE },
	{ 64, 64,   CONV_SIZE, CONV_SIZE },
	{ 64, 64,   CONV_SIZE, CONV_SIZE },
	{ 64, 64,   CONV_SIZE, CONV_SIZE },
	{ 64, 64,   CONV_SIZE, CONV_SIZE },
	{ 128, 64,  CONV_SIZE, CONV_SIZE },
	{ 128, 128, CONV_SIZE, CONV_SIZE },
	{ 128, 64,  1, 1 },
	{ 128, 128, CONV_SIZE, CONV_SIZE },
	{ 128, 128, CONV_SIZE, CONV_SIZE },
	{ 256, 128, CONV_SIZE, CONV_SIZE },
	{ 256, 256, CONV_SIZE, CONV_SIZE },
	{ 256, 128, 1, 1 },
	{ 256, 256, CONV_SIZE, CONV_SIZE },
	{ 256, 256, CONV_SIZE, CONV_SIZE },
	{ 512, 256, CONV_SIZE, CONV_SIZE },
	{ 512, 512, CONV_SIZE, CONV_SIZE },
	{ 512, 256, 1, 1 },
	{ 512, 512, CONV_SIZE, CONV_SIZE },
	{ 512, 512, CONV_SIZE, CONV_SIZE }
};

// batch normalization layer shapes
int bshape[CONV_LEVELS] = { 64, 64, 64, 64, 64, 64, 128, 128, 128, 128, 128, 256, 256, 256, 256, 256, 512, 512, 512, 512 };

// dense layer
int dshape[1][2]= {
   { 512, 10}
};

#endif	// FISHER_PRUNING

float *****wc;	// weights convolution
float **bc;	// biases convolution

float ***wd;	// weights dense
float **bd;	// biases dense

#if SPARSE_CONVOLUTIONS
	// sparse conv
	csr_t ****wc_sparse;

#endif	// SPARSE_CONVOLUTIONS


float batchnorm_weights[CONV_LEVELS][512];
float batchnorm_biases[CONV_LEVELS][512];

// Blocks for intermediate convolutions
int mem_block_shape[3] = {512, SIZE, SIZE};	// not optimal defining 512 statically
float ***mem_block1;
float ***mem_block2;
float ***shortcut_mem;

// Blocks for dense flatten layers
int mem_block_dense_shape = { 512 * 3 * 3 };	// size of layer before the fully connected
float *mem_block1_dense;
float *mem_block2_dense;

// Weights and image block END


/****************************************************************************************************************************/

void reset_mem_block(float ***mem) {
	int i, j, k;
	for (i = 0; i < mem_block_shape[0]; i++) {
		for (j = 0; j < mem_block_shape[1]; j++) {
			for (k = 0; k < mem_block_shape[2]; k++) {
				mem[i][j][k] = 0.f;
			}
		}
	}
}

/****************************************************************************************************************************/

void reset_mem_block_dense(float *mem) {
	int i;
	for (i = 0; i < mem_block_dense_shape; i++) {
		mem[i] = 0.f;
	}
}

/****************************************************************************************************************************/

void init_memory() {
	int i, j, k, l;

	// Init image memory
	image = malloc(3 * sizeof(float**));

	for (i = 0; i < 3; i++) {
		image[i] = malloc(SIZE * sizeof(float*));
		for (j = 0; j < SIZE; j++) {
			image[i][j] = malloc(SIZE * sizeof(float));
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

	// Init convolution weights
	wc = malloc(CONV_LEVELS * sizeof(float****));
	bc = malloc(CONV_LEVELS * sizeof(float*));

	for (l = 0; l < CONV_LEVELS; l++) {
		wc[l] = malloc(cshape[l][0] * sizeof(float***));
		for (i = 0; i < cshape[l][0]; i++) {
			wc[l][i] = malloc(cshape[l][1] * sizeof(float**));
			for (j = 0; j < cshape[l][1]; j++) {
				wc[l][i][j] = malloc(cshape[l][2] * sizeof(float*));
				for (k = 0; k < cshape[l][2]; k++) {
					wc[l][i][j][k] = malloc(cshape[l][3] * sizeof(float));
				}
			}
		}
		bc[l] = malloc(cshape[l][0] * sizeof(float));
	}

	// Init dense weights
	wd = malloc(2 * sizeof(float**));
	bd = malloc(2 * sizeof(float*));

	for (l = 0; l < 1; l++) {
		wd[l] = malloc(dshape[l][0] * sizeof(float*));
		for (i = 0; i < dshape[l][0]; i++) {
			wd[l][i] = malloc(dshape[l][1] * sizeof(float));
		}
		bd[l] = malloc(dshape[l][1] * sizeof(float));
	}

	// Init mem_blocks	// this size could be dynamic
	mem_block1 = malloc(mem_block_shape[0] * sizeof(float**));
	mem_block2 = malloc(mem_block_shape[0] * sizeof(float**));
	shortcut_mem = malloc(mem_block_shape[0] * sizeof(float**));

	for (i = 0; i < mem_block_shape[0]; i++) {
		mem_block1[i] = malloc(mem_block_shape[1] * sizeof(float*));
		mem_block2[i] = malloc(mem_block_shape[1] * sizeof(float*));
		shortcut_mem[i] = malloc(mem_block_shape[1] * sizeof(float*));
		for (j = 0; j < mem_block_shape[1]; j++) {
			mem_block1[i][j] = malloc(mem_block_shape[2] * sizeof(float));
			mem_block2[i][j] = malloc(mem_block_shape[2] * sizeof(float));
			shortcut_mem[i][j] = malloc(mem_block_shape[2] * sizeof(float));
		}
	}
// 	reset_mem_block(mem_block1);
// 	reset_mem_block(mem_block2);

	// Init mem blocks dense
	mem_block1_dense = calloc(mem_block_dense_shape, sizeof(float));
	mem_block2_dense = calloc(mem_block_dense_shape, sizeof(float));

	// Init batchnorm blocks
	//batchnorm_weights =  malloc(2 * sizeof(float*));
	//batchnorm_biases  =  malloc(2 * sizeof(float*));

	//for (int z = 0; z < 20; z++) {
		//batchnorm_weights[z] = malloc(512 * sizeof(float));
		//batchnorm_biases[z]  = malloc(512 * sizeof(float));
	//}
}

/****************************************************************************************************************************/

void free_memory() {
	int i, j, k, l;

	// Free image memory
	for (i = 0; i < 3; i++) {
		for (j = 0; j < SIZE; j++) {
			free(image[i][j]);
		}
		free(image[i]);
	}
	free(image);

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
		free(bc[l]);
	}
//	free(wc);
//	free(bc);

#if SPARSE_CONVOLUTIONS
	free(wc_sparse);
#else
	free(wc);
#endif	// SPARSE_CONVOLUTIONS

	free(bc);

	// Free dense weights
	/*
	for (l = 0; l < 2; l++) {
		for (i = 0; i < dshape[l][0]; i++) {
			free(wd[l][i]);
		}
		free(wd[l]);
		free(bd[l]);
	}
	free(wd);
	free(bd); */

	// Free memblocks
	for (i = 0; i < mem_block_shape[0]; i++) {
		for (j = 0; j < mem_block_shape[1]; j++) {
			free(mem_block1[i][j]);
			free(mem_block2[i][j]);
			free(shortcut_mem[i][j]);
		}
		free(mem_block1[i]);
		free(mem_block2[i]);
		free(shortcut_mem[i]);
	}
	free(mem_block1);
	free(mem_block2);
	free(shortcut_mem);

	free(mem_block1_dense);
	free(mem_block2_dense);
}

/****************************************************************************************************************************/

void read_weights(char *in_file, int lvls) {
	/*
	weights are written out as:
		- 20 x convolutional weights NO bias
		- 20 x batchnorm weights with bias
		- 1  x fc weights with bias
	*/

	float dval;
	int i, j, k, l, z;
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
	for (z = 0; z < CONV_LEVELS; z++) {

		printf("Read conv block %d weights\n", z);
		for (i = 0; i < cshape[z][0]; i++) {
			for (j = 0; j < cshape[z][1]; j++) {
				for (k = 0; k < cshape[z][2]; k++) {
					for (l = 0; l < cshape[z][3]; l++) {
						fscanf(iin, "%f", &dval);
						wc[z][i][j][k][l] = dval;
					}
				}
			}
		}
		total_lvls_read += 1;
	}

	/* // run this to check conv weights are correct
	z = 19;

	// print back to verify
	for (i = 0; i < cshape[z][0]; i++) {
		for (j = 0; j < cshape[z][1]; j++) {
			for (k = 0; k < cshape[z][2]; k++) {
				for (l = 0; l < cshape[z][3]; l++) {
					printf("conv 5: %f \n", wc[z][i][j][k][l]);
				}
			}
		}
	}

	return;
	*/
	for (z = 0; z < CONV_LEVELS; z++) {
		// batchnorm weights and biases
		printf("Read batchnorm block %d weights\n", z);
		for (i = 0; i < bshape[z]; i++) {
				fscanf(iin, "%f", &dval);
				//printf("weight %i : %f \n", i, dval);
				batchnorm_weights[z][i] = dval;
			}

		for (i = 0; i < bshape[z]; i++) {
			fscanf(iin, "%f", &dval);
			//printf("bias %i : %f \n", i, dval);
			batchnorm_biases[z][i] = dval;
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
				image[l][i][j] = dval;
// 				printf("i[%d][%d][%d]:%f\n", i, j, l, dval);
			}
		}
	}
}

/****************************************************************************************************************************/

void convolution_3_x_3(float **matrix, float **kernel, float **out, int size, int stride) {

	int i, j;
	float sum;
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

//	for (i = 0; i < (size + 2); ++i)
//		free(zeropad[i]);
//	free(zeropad);
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

void convolution_1_x_1(float **matrix, float **kernel, float **out, int size) {

	int i, j;
	float sum;
	float zeropad[size+2][size+2];
	memset(zeropad, 0, ((size+2)*(size+2)*sizeof(float)));

	for (i = 0; i < size; i++) {
		for (j = 0; j < size; j++) {
			zeropad[i + 1][j + 1] = matrix[i][j];
		}
	}

	for (i = 0; i < size; i++) {
		for (j = 0; j < size; j++) {
			sum = zeropad[i][j] * kernel[0][0];
			out[i][j] += sum;
		}
	}
}

/****************************************************************************************************************************/

void convolution_1_x_1_sparse(float **matrix, csr_t *kernel, float **out, int size) {
	int i, j;
	float sum;
	float zeropad[size+2][size+2];
	memset(zeropad, 0, ((size+2)*(size+2)*sizeof(float)));

	for (i = 0; i < size; i++) {
		for (j = 0; j < size; j++) {
			zeropad[i + 1][j + 1] = matrix[i][j];
		}
	}

	int k,l;
	for (i = 0; i < size; i++) {
		for (j = 0; j < size; j++) {
			//sum = zeropad[i][j] * kernel[0][0];
			//out[i][j] += sum;

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

// no bias
void add_relu(float **out, int size) {
	int i, j;

	for (i = 0; i < size; i++) {
		for (j = 0; j < size; j++) {
			if (out[i][j] < 0)
				out[i][j] = 0.f;
		}
	}
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
				total++;
			}
		}
	}
}

/****************************************************************************************************************************/

void dense(float *in, float **weights, float *out, int sh_in, int sh_out) {

	int i, j;

	#pragma omp parallel for private(j) schedule(dynamic,1) num_threads(NUMBER_OF_THREADS)
	for (i = 0; i < sh_out; i++) {
		float sum = 0.0;
		for (j = 0; j < sh_in; j++) {
			sum += in[j] * weights[j][i];
		}
		out[i] = sum;
	}
}

/****************************************************************************************************************************/

void batchnorm(float ***in, float ***out, float *weights, float *bias, int num_channels, int im_size) {
	int channel, i, j;

 	#pragma omp parallel for private(j) schedule(dynamic,1) num_threads(NUMBER_OF_THREADS)
	for(channel = 0; channel < num_channels; channel++) {
		for(i = 0; i < im_size; i++) {
			for(j = 0; j < im_size; j++) {
				out[channel][i][j] = in[channel][i][j] * weights[channel] + bias[channel];
			}
		}
	}
}

/****************************************************************************************************************************/

void batchnorm_and_relu(float ***in, float ***out, float *weights, float *bias, int num_channels, int im_size) {
	int channel, i, j;

	#pragma omp parallel for private(i,j) schedule(dynamic,1) num_threads(NUMBER_OF_THREADS)
	for(channel = 0; channel < num_channels; channel++) {
		for(i = 0; i < im_size; i++) {
			for(j = 0; j < im_size; j++) {
				out[channel][i][j] = in[channel][i][j] * weights[channel] + bias[channel];
				if (out[channel][i][j] < 0.f)
					out[channel][i][j] = 0.f;
			}
		}
	}
}

/****************************************************************************************************************************/

void dump_image() {
	int i, j, k;
	for (i = 0; i < 3; i++) {
		for (j = 0; j < SIZE; j++) {
			for (k = 0; k < SIZE; k++) {
				printf("%.12lf\n", image[i][j][k]);
			}
		}
	}
}

/****************************************************************************************************************************/

void output_predictions(FILE *out, int only_convolution, int size, int cur_size) {

	int i;
	int c=0;

	if (only_convolution == 1) {
		for (i = 0; i < size * cur_size * cur_size; i++) {
			fprintf(out, "%g ", mem_block1_dense[i]);
		}
	}
	else {
		double maximum=-1;

//	    	dshape[0][1] ==> 10
		for (i = 0; i < dshape[0][1]; i++) {
			fprintf(out, "%g ", mem_block2_dense[i]);

			if(mem_block1_dense[i]>maximum){
				maximum=mem_block2_dense[i];
				c=i+1;
			}
		}
		fprintf(out, "\n");
		printf("-------------------------\n");
		printf("This image depicts class: %d\n",c);
	}
}

/****************************************************************************************************************************/

void conv_norm_block(int level, int shortcut) {

	int in_planes = cshape[level][1];
	int i, j, k;

	// if shortcut then save image for layer
	if(shortcut==1) {
		int i, j, k;
		for (i = 0; i < mem_block_shape[0]; i++) {
			for (j = 0; j < mem_block_shape[1]; j++) {
				for (k = 0; k < mem_block_shape[2]; k++) {
					shortcut_mem[i][j][k] = mem_block1[i][j][k];
				}
			}
		}
	}

	//int in_planes = cshape[level][0]
	int out_planes = cshape[level][0];
	int stride = 1;

	//-------------------------------------------------------------------------------------------------------------------------------

	// conv 1
	#pragma omp parallel for private(j) schedule(dynamic,1) num_threads(NUMBER_OF_THREADS)
	for (i = 0; i < out_planes; i++) {
		for (j = 0; j < in_planes; j++) {
			#if SPARSE_CONVOLUTIONS
						convolution_3_x_3_sparse(mem_block1[j], wc_sparse[level][i][j], mem_block2[i], im_sizes[level], stride);
			#else
						convolution_3_x_3(mem_block1[j], wc[level][i][j], mem_block2[i], im_sizes[level], stride);
			#endif	// SPARSE_CONVOLUTIONS
		}
	}
	batchnorm_and_relu(mem_block2, mem_block1, batchnorm_weights[level], batchnorm_biases[level], out_planes, im_sizes[level]);
	// batchnorm(mem_block2, mem_block1, batchnorm_weights[level], batchnorm_biases[level], out_planes, im_sizes[level]);
	// for(i = 0; i < out_planes; i++) {
	// 	add_relu(mem_block1[i], im_sizes[level]);
	// }
	reset_mem_block(mem_block2);

	//-------------------------------------------------------------------------------------------------------------------------------

	// conv 2
	level += 1;

	in_planes  = cshape[level][1];
	out_planes = cshape[level][0];

	//-------------------------------------------------------------------------------------------------------------------------------

	#pragma omp parallel for private(j) schedule(dynamic,1) num_threads(NUMBER_OF_THREADS)
	for (i = 0; i < out_planes; i++) {
	    for (j = 0; j < in_planes; j++) {
			    #if SPARSE_CONVOLUTIONS
						    convolution_3_x_3_sparse(mem_block1[j], wc_sparse[level][i][j], mem_block2[i], im_sizes[level], stride);
			    #else
						    convolution_3_x_3(mem_block1[j], wc[level][i][j], mem_block2[i], im_sizes[level], stride);
			    #endif	// SPARSE_CONVOLUTIONS
	}
	}
	    batchnorm_and_relu(mem_block2, mem_block1, batchnorm_weights[level], batchnorm_biases[level], out_planes, im_sizes[level]);
	    // batchnorm(mem_block2, mem_block1, batchnorm_weights[level], batchnorm_biases[level], out_planes, im_sizes[level]);
	    // for(i = 0; i < out_planes; i++) {
	    // 	add_relu(mem_block1[i], im_sizes[level]);
	    // }
	    reset_mem_block(mem_block2);

	    // if shortcut: conv bn + out
	    if(shortcut==1) {

		    level += 1;

		    in_planes  = cshape[level][1];
		    out_planes = cshape[level][0];

		    for (i = 0; i < out_planes; i++) {
			    for (j = 0; j < in_planes; j++) {
				    #if SPARSE_CONVOLUTIONS
							    convolution_1_x_1_sparse(shortcut_mem[j], wc_sparse[level][i][j], mem_block2[i], im_sizes[level]);
				    #else
							    convolution_1_x_1(shortcut_mem[j], wc[level][i][j], mem_block2[i], im_sizes[level]);
				    #endif	// SPARSE_CONVOLUTIONS
			    }
		    }
		    batchnorm_and_relu(mem_block2, shortcut_mem, batchnorm_weights[level], batchnorm_biases[level], out_planes, im_sizes[level]);
		    // batchnorm(mem_block2, shortcut_mem, batchnorm_weights[level], batchnorm_biases[level], out_planes, im_sizes[level]);
		    // for(i = 0; i < out_planes; i++) {
		    // 	add_relu(shortcut_mem[i], im_sizes[level]);
		    // }
		    reset_mem_block(mem_block2);

		    // add results
		    for(i = 0; i < out_planes; i++) {
			    for(j = 0; j < im_sizes[level]; j++) {
				    for(k = 0; k < im_sizes[level]; k++) {
					    mem_block1[i][j][k] = mem_block1[i][j][k] + shortcut_mem[i][j][k];
				    }
			    }
		    }
	    }
}

/****************************************************************************************************************************/

void get_resnet18_predict(FILE *out, int only_convolution) {

	int i, j, k;
	int level = 0;

	// Init intermediate memory
	reset_mem_block(mem_block1);
	reset_mem_block(mem_block2);
	reset_mem_block_dense(mem_block1_dense);
	reset_mem_block_dense(mem_block2_dense);

	//-------------------------------------------------------------------------------------------------------------------------------

	// Layer 1 (Convolution 3 -> 64)

	//add_relu(mem_block2[i], 32);	///???? WHY DO WE NEED THIS HERE?
	// print the image
	/*
	for (i = 0; i < 32; i++) {
		for (j = 0; j < 32; j++) {
			for (k = 0; k < 3; k++) {
				printf("%f \n", image[k][i][j]);
			}
		}
	}
	return;
  */

	int counter = 0;
	for (i = 0; i < cshape[level][0]; i++) {
		for (j = 0; j < cshape[level][1]; j++) {
			#if FIRST_CONV_SPARSE
						convolution_3_x_3_sparse(image[j], wc_sparse[level][i][j], mem_block2[i], im_sizes[level], stride);
			#else
						convolution_3_x_3(image[j], wc[level][i][j], mem_block2[i], im_sizes[level], 1);
			#endif	// FIRST_CONV_SPARSE

		}
		// [print content of mem block]
		/*
		for(int m = 0; m < 32; m++) {
			for(int n = 0; n < 32; n++) {
				printf("%i: %f\n", counter, mem_block1[i][m][n]);
				counter++;
			}
		}
		*/
		//relu(mem_block2[i], 32);
	}

	batchnorm_and_relu(mem_block2, mem_block1, batchnorm_weights[level], batchnorm_biases[level], 64, 32);

	// batchnorm(mem_block2, mem_block1, batchnorm_weights[level], batchnorm_biases[level], 64, 32);
	// for(i = 0; i < cshape[level][0]; i++) {
	// 	add_relu(mem_block1[i], 32);
	// }
	reset_mem_block(mem_block2);

	/*
	counter = 0;
	// print mem block 2:
	for (i = 0; i < 64; i++) {
		for (j = 0; j < 32; j++) {
			for (k = 0; k < 32; k++) {
				counter++;
				if (counter < 100) {
					printf("%i: %f\n",counter, mem_block2[i][j][k]);
				}
			}
		}
	}
	return;
	*/

	level++;

	//-------------------------------------------------------------------------------------------------------------------------------

	int shortcut    = 1;
	int no_shortcut = 0;

	// 2 blocks of 64
	conv_norm_block(level, no_shortcut);   level+=2;
	conv_norm_block(level, no_shortcut);   level+=2;

	// 2 blocks of 128
	conv_norm_block(level, shortcut);   level+=3;
	conv_norm_block(level, no_shortcut); level+=2;

	// 2 blocks of 256
	conv_norm_block(level, shortcut);  level+=3;
	conv_norm_block(level, no_shortcut); level+=2;

	// 2 blocks of 512
	conv_norm_block(level, shortcut);  level+=3;
	conv_norm_block(level, no_shortcut); level+=2;


	level = level - 1;
	// flatten
	flatten(mem_block1, mem_block1_dense, cshape[level][0], im_sizes[level], im_sizes[level]);

	// dense
	level = 0;
	dense(mem_block1_dense, wd[level], mem_block2_dense, dshape[level][0], dshape[level][1]);
	add_bias_and_relu_flatten(mem_block2_dense, bd[level], dshape[level][1], 1);
	reset_mem_block_dense(mem_block1_dense);

	return;
}

/****************************************************************************************************************************/

char *trimwhitespace(char *str){
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
		lvls = CONV_LEVELS;
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
//		normalize_image();
//		dump_image();

		gettimeofday(&tStart, NULL);
// 		get_resnet18_predict(only_convolution);
		get_resnet18_predict(results, only_convolution);

		gettimeofday(&tEnd, NULL);
		deltaTime = get_seconds(tStart, tEnd);
		printf("Infer image %s: %.3lf sec\n", buf, deltaTime);

// 		output_predictions(results, only_convolution);
		output_predictions(results, only_convolution, 512, 3);
	}

	free_memory();
	fclose(file_list);

	return 0;
}
