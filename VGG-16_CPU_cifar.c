/*
	Pretrained VGG-16 Convolutional Neural Network in C language and OpenMP API
	GitHUB Page: https://github.com/jcanore/vgg16
	Author: ZFTurbo/jocare

	Compilation: gcc -O3 VGG-16_CPU_cifar.c -lm -fopenmp -o VGG-16_CPU_cifar
	Usage: VGG-16_CPU_cifar <weights_path> <file_with_list_of_images> <output file> <output convolution features (optional)>
	Example: VGG-16_CPU_cifar ../../weights/weights.txt" ../../img/image_list.txt results_imagenet_conv.txt 1
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
#define CONV_LEVELS 13
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

// Weights and image block START
float ***image;

#if FISHER_PRUNING
#define SPARSE_CONVOLUTIONS 0	// force dense convolutions
/* ORIGINAL EXPERIMENTS
int cshape[13][4] = {
    { 56, 3, CONV_SIZE, CONV_SIZE },
    { 62, 56, CONV_SIZE, CONV_SIZE },
    { 121, 62, CONV_SIZE, CONV_SIZE },
    { 127, 121, CONV_SIZE, CONV_SIZE },
    { 232, 127, CONV_SIZE, CONV_SIZE },
    { 229, 232, CONV_SIZE, CONV_SIZE },
    { 183, 229, CONV_SIZE, CONV_SIZE },
    { 134, 183, CONV_SIZE, CONV_SIZE },
    { 101, 134, CONV_SIZE, CONV_SIZE },
    { 70, 101, CONV_SIZE, CONV_SIZE },
    { 60, 70, CONV_SIZE, CONV_SIZE },
    { 64, 60, CONV_SIZE, CONV_SIZE },
    { 79, 64, CONV_SIZE, CONV_SIZE }
};

int dshape[2][2]= {
  { 79, 47 },
  { 47, 10}
};

*/


// FIXED 90% ACCURACY EXPERIMENTS
int cshape[13][4] = {
    { 64, 3, CONV_SIZE, CONV_SIZE },
    { 62, 64, CONV_SIZE, CONV_SIZE },
    { 128, 62, CONV_SIZE, CONV_SIZE },
    { 127, 128, CONV_SIZE, CONV_SIZE },
    { 256, 127, CONV_SIZE, CONV_SIZE },
    { 246, 256, CONV_SIZE, CONV_SIZE },
    { 230, 246, CONV_SIZE, CONV_SIZE },
    { 265, 230, CONV_SIZE, CONV_SIZE },
    { 199, 265, CONV_SIZE, CONV_SIZE },
    { 162, 199, CONV_SIZE, CONV_SIZE },
    { 146, 162, CONV_SIZE, CONV_SIZE },
    { 155, 146, CONV_SIZE, CONV_SIZE },
    { 183, 155, CONV_SIZE, CONV_SIZE }
};

int dshape[2][2]= {
  { 183, 1164 },
  { 164, 10}
};


#else	// FISHER_PRUNING
int cshape[CONV_LEVELS][4] = {
	{ 64, 3, CONV_SIZE, CONV_SIZE },
	{ 64, 64, CONV_SIZE, CONV_SIZE },
	{ 128, 64, CONV_SIZE, CONV_SIZE },
	{ 128, 128, CONV_SIZE, CONV_SIZE },
	{ 256, 128, CONV_SIZE, CONV_SIZE },
	{ 256, 256, CONV_SIZE, CONV_SIZE },
	{ 256, 256, CONV_SIZE, CONV_SIZE },
	{ 512, 256, CONV_SIZE, CONV_SIZE },
	{ 512, 512, CONV_SIZE, CONV_SIZE },
	{ 512, 512, CONV_SIZE, CONV_SIZE },
	{ 512, 512, CONV_SIZE, CONV_SIZE },
	{ 512, 512, CONV_SIZE, CONV_SIZE },
	{ 512, 512, CONV_SIZE, CONV_SIZE }
};

int dshape[2][2]= {
   { 512, 512 },
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

// Blocks for intermediate convolutions
int mem_block_shape[3] = {512, SIZE, SIZE};	// not optimal defining 512 statically
float ***mem_block1;
float ***mem_block2;

// Blocks for dense flatten layers
int mem_block_dense_shape = { 512 * 1 * 1 };	// size of layer before the fully connected
float *mem_block1_dense;
float *mem_block2_dense;

// Weights and image block END


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

	for (l = 0; l < 2; l++) {
		wd[l] = malloc(dshape[l][0] * sizeof(float*));
		for (i = 0; i < dshape[l][0]; i++) {
			wd[l][i] = malloc(dshape[l][1] * sizeof(float));
		}
		bd[l] = malloc(dshape[l][1] * sizeof(float));
	}

	// Init mem_blocks	// jocare: this size could be dynamic
	mem_block1 = malloc(mem_block_shape[0] * sizeof(float**));
	mem_block2 = malloc(mem_block_shape[0] * sizeof(float**));

	for (i = 0; i < mem_block_shape[0]; i++) {
		mem_block1[i] = malloc(mem_block_shape[1] * sizeof(float*));
		mem_block2[i] = malloc(mem_block_shape[1] * sizeof(float*));
		for (j = 0; j < mem_block_shape[1]; j++) {
			mem_block1[i][j] = malloc(mem_block_shape[2] * sizeof(float));
			mem_block2[i][j] = malloc(mem_block_shape[2] * sizeof(float));
		}
	}
// 	reset_mem_block(mem_block1);
// 	reset_mem_block(mem_block2);

	// Init mem blocks dense
	mem_block1_dense = calloc(mem_block_dense_shape, sizeof(float));
	mem_block2_dense = calloc(mem_block_dense_shape, sizeof(float));
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

#if SPARSE_CONVOLUTIONS
	free(wc_sparse);
#else
	free(wc);
#endif	// SPARSE_CONVOLUTIONS

	free(bc);

	// Free dense weights
	for (l = 0; l < 2; l++) {
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
			free(mem_block1[i][j]);
			free(mem_block2[i][j]);
		}
		free(mem_block1[i]);
		free(mem_block2[i]);
	}
	free(mem_block1);
	free(mem_block2);

	free(mem_block1_dense);
	free(mem_block2_dense);
}

/****************************************************************************************************************************/

void read_weights(char *in_file, int lvls) {
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
	for (z = 0; z < CONV_LEVELS; z++) {

//		if (total_lvls_read >= lvls && lvls != -1)
//		break;

		printf("Read conv block %d weights\n", z);
		for (i = 0; i < cshape[z][0]; i++) {
			for (j = 0; j < cshape[z][1]; j++) {
				for (k = 0; k < cshape[z][2]; k++) {
					for (l = 0; l < cshape[z][3]; l++) {
						fscanf(iin, "%f", &dval);
						//wc[z][i][j][CONV_SIZE - k - 1][CONV_SIZE - l - 1] = dval;
						wc[z][i][j][k][l] = dval;
					}
				}
			}
		}
		for (i = 0; i < cshape[z][0]; i++) {
			fscanf(iin, "%f", &dval);
// 			printf("dval: %.4f ", dval);
			bc[z][i] = dval;
		}
		total_lvls_read += 1;
	}

	if (total_lvls_read >= lvls && lvls != -1)
		return;

	//int count=0;

	// Reading dense weights
	for (z = 0; z < 2; z++) {
//		int count=0;
//		if (total_lvls_read >= lvls && lvls != -1)
//			break;

		printf("Read dense block %d weights\n", z);
//		for (i = 0; i < dshape[z][1]; i++) { // 512, 512; 512, 10
//			for (j = 0; j < dshape[z][0]; j++) {
		for (i = 0; i < dshape[z][0]; i++) {
			for (j = 0; j < dshape[z][1]; j++) {
				fscanf(iin, "%f", &dval);
				wd[z][i][j] = dval;

// 				if (count < 100) {
// 				    printf("%f\n", dval);
// 				    count++;
// 				}
			}
		}
//		for (i = 0; i < dshape[z][1]; i++) {
//			for (j = 0; j < dshape[z][0]; j++) {
//				if(z==1) {
//					printf("weight[%i][%i]: %f\n",i,j, wd[z][i][j]) ;
//				}
//			}
//		}

//		//dshape[z][0]
//		if(z==1){
//			for (i = 0; i < 10; i++) {
//				for (j = 0; j < 512; j++) {
// 					printf("weight[%i][%i]: %f\n",i, j, wd[1][i][j]);
//				}
//			}
//		}

		for (i = 0; i < dshape[z][1]; i++) {
			fscanf(iin, "%f", &dval);
			bd[z][i] = dval;
		}
// 		total_lvls_read += 1;
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

void normalize_image() {

	int i, j, l;

// 	t.sub_(m).div_(s)

//	float coef[3] = { 103.939, 116.779, 123.68 };
  	float coef[2][3] = {{0.4914, 0.4822, 0.4465}, {0.2023, 0.1994, 0.2010}};

	for (l = 0; l < 3; l++) {
		for (i = 0; i < SIZE; i++) {
			for (j = 0; j < SIZE; j++) {
//				image[l][i][j] -= coef[l];

 				image[l][i][j] /= 255;
//    				image[l][i][j] -= coef[1][l];
//    				image[l][i][j] = image[l][i][j] / coef[2][l];
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
				zeropad[i   ][j    ] * kernel[0][0] +
				zeropad[i   ][j + 1] * kernel[0][1] +
				zeropad[i   ][j + 2] * kernel[0][2] +

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

void convolution_3_x_3_sparse(float **matrix, csr_t* kernel, float **out, int size) {

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

			sum = 0;
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

void add_bias_and_relu(float **out, float bs, int size) {
	int i, j;

	for (i = 0; i < size; i++) {
		for (j = 0; j < size; j++) {
			out[i][j] += bs;
			if (out[i][j] < 0)
				out[i][j] = 0.0;
			// printf("%.12lf\n", out[i][j]);
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
				out[i] = 0.0;
		}
	}
}

/****************************************************************************************************************************/

float max_of_4(float a, float b, float c, float d) {
	if (a >= b && a >= c && a >= d) {
		return a;
	}
	if (b >= c && b >= d) {
		return b;
	}
	if (c >= d) {
		return c;
	}
	return d;
}

/****************************************************************************************************************************/

void maxpooling(float **out, int size) {
	int i, j;
	for (i = 0; i < size; i+=2) {
		for (j = 0; j < size; j+=2) {
			out[i / 2][j / 2] = max_of_4(out[i][j], out[i + 1][j], out[i][j + 1], out[i + 1][j + 1]);
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

	#pragma omp parallel for private(j) schedule(dynamic,1) num_threads(NUMBER_OF_THREADS)
	for (i = 0; i < sh_out; i++) {
		float sum = 0.0;
		for (j = 0; j < sh_in; j++) {
			sum += in[j] * weights[i][j];
		}
		out[i] = sum;
	}
}

/****************************************************************************************************************************/

void softmax(float *out, int sh_out) {

	int i;
	float max_val, sum;
	max_val = out[0];

	for (i = 1; i < sh_out; i++) {
		if (out[i] > max_val)
			max_val = out[i];
	}

	sum = 0.0;

	for (i = 0; i < sh_out; i++) {
		out[i] = exp(out[i] - max_val);
		sum += out[i];
	}
	for (i = 0; i < sh_out; i++) {
		out[i] /= sum;
	}
}

/****************************************************************************************************************************/

void dump_memory_structure_conv(float ***mem, int sh0, int sh1, int sh2) {
	int i, j, k;
	for (i = 0; i < sh0; i++) {
		for (j = 0; j < sh1; j++) {
			for (k = 0; k < sh2; k++) {
				printf("%.12lf\n", mem[i][j][k]);
			}
		}
	}
}

/****************************************************************************************************************************/

void dump_memory_structure_conv_to_file(float ***mem, int sh0, int sh1, int sh2) {
	FILE *out;
	int i, j, k;
	out = fopen("debug_c.txt", "w");
	for (i = 0; i < sh0; i++) {
		for (j = 0; j < sh1; j++) {
			for (k = 0; k < sh2; k++) {
				fprintf(out, "%.12lf\n", mem[i][j][k]);
			}
		}
	}
	fclose(out);
}

/****************************************************************************************************************************/

void dump_memory_structure_dense(float *mem, int sh0) {
	int i;
	for (i = 0; i < sh0; i++) {
		printf("%.12lf\n", mem[i]);
	}
}

/****************************************************************************************************************************/

void dump_memory_structure_dense_to_file(float *mem, int sh0) {
	FILE *out;
	int i;
	out = fopen("debug_c.txt", "w");
	for (i = 0; i < sh0; i++) {
		fprintf(out, "%.12lf\n", mem[i]);
	}
	fclose(out);
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

		for (i = 0; i < dshape[1][1]; i++) {
			fprintf(out, "%g ", mem_block1_dense[i]);
			if(mem_block1_dense[i]>maximum){
				maximum=mem_block1_dense[i];
				c=i+1;
			}
		}
		fprintf(out, "\n");
		printf("-------------------------\n");
		printf("This image depicts class: %d\n",c);
	}
}

/****************************************************************************************************************************/

//void get_VGG16_predict(FILE *out, int only_convolution) {
void get_VGG16_predict(int only_convolution) {

	int i, j;
	int level, cur_size, cur_stride;

	// Init intermediate memory
	reset_mem_block(mem_block1);
	reset_mem_block(mem_block2);
	reset_mem_block_dense(mem_block1_dense);
	reset_mem_block_dense(mem_block2_dense);

	//-------------------------------------------------------------------------------------------------------------------------------

	// Layer 1 (Convolution 3 -> 64)
	level = 0;
	cur_size = SIZE;
	cur_stride = 1;
//	printf("(cur_size): %d\n", cur_size);

	#pragma omp parallel for private(j) schedule(dynamic,1) num_threads(NUMBER_OF_THREADS)
	for (i = 0; i < cshape[level][0]; i++) {
		for (j = 0; j < cshape[level][1]; j++) {

#if FIRST_CONV_SPARSE
			convolution_3_x_3_sparse(image[j], wc_sparse[level][i][j], mem_block1[i], cur_size);
#else
			convolution_3_x_3(image[j], wc[level][i][j], mem_block1[i], cur_size, cur_stride);
#endif	// FIRST_CONV_SPARSE
		}
		add_bias_and_relu(mem_block1[i], bc[level][i], cur_size);
	}

//     	printf("level, cshape[level][0], cur_size, lines: %d, %d, %d, %d\n", level, cshape[level][0], cur_size, cshape[level][0]*cur_size*cur_size);
//     	flatten(mem_block1, mem_block1_dense, cshape[level][0], cur_size, cur_size);
//     	output_predictions(out, only_convolution, cshape[level][0], cur_size);

	//-------------------------------------------------------------------------------------------------------------------------------

	// Layer 2 (Convolution 64 -> 64)
	level = 1;
	#pragma omp parallel for private(j) schedule(dynamic,1) num_threads(NUMBER_OF_THREADS)
	for (i = 0; i < cshape[level][0]; i++) {
		for (j = 0; j < cshape[level][1]; j++) {

#if SPARSE_CONVOLUTIONS
			convolution_3_x_3_sparse(mem_block1[j], wc_sparse[level][i][j], mem_block2[i], cur_size);
#else
			convolution_3_x_3(mem_block1[j], wc[level][i][j], mem_block2[i], cur_size, cur_stride);
#endif	// SPARSE_CONVOLUTIONS
		}
		add_bias_and_relu(mem_block2[i], bc[level][i], cur_size);
	}
	reset_mem_block(mem_block1);

//  	printf("level, cshape[level][0], cur_size, lines: %d, %d, %d, %d\n", level, cshape[level][0], cur_size, cshape[level][0]*cur_size*cur_size);
//     	flatten(mem_block2, mem_block1_dense, cshape[level][0], cur_size, cur_size);
//     	output_predictions(out, only_convolution, cshape[level][0], cur_size);

	//-------------------------------------------------------------------------------------------------------------------------------

	// Layer 3 (MaxPooling)
	#pragma omp parallel for schedule(dynamic,1) num_threads(NUMBER_OF_THREADS)
	for (i = 0; i < cshape[level][0]; i++) {
		maxpooling(mem_block2[i], cur_size);
	}
	cur_size /= 2;
//	printf("(cur_size): %d\n", cur_size);

	//-------------------------------------------------------------------------------------------------------------------------------

	// Layer 4 (Convolution 64 -> 128)
	level = 2;
	#pragma omp parallel for private(j) schedule(dynamic,1) num_threads(NUMBER_OF_THREADS)
	for (i = 0; i < cshape[level][0]; i++) {
		for (j = 0; j < cshape[level][1]; j++) {
#if SPARSE_CONVOLUTIONS
			convolution_3_x_3_sparse(mem_block2[j], wc_sparse[level][i][j], mem_block1[i], cur_size);
#else
			convolution_3_x_3(mem_block2[j], wc[level][i][j], mem_block1[i], cur_size, cur_stride);
#endif	// SPARSE_CONVOLUTIONS
		}
		add_bias_and_relu(mem_block1[i], bc[level][i], cur_size);
	}
	reset_mem_block(mem_block2);

// 	printf("level, cshape[level][0], cur_size, lines: %d, %d, %d, %d\n", level, cshape[level][0], cur_size, cshape[level][0]*cur_size*cur_size);
//     	flatten(mem_block1, mem_block1_dense, cshape[level][0], cur_size, cur_size);
//     	output_predictions(out, only_convolution, cshape[level][0], cur_size);

	//-------------------------------------------------------------------------------------------------------------------------------

	// Layer 5 (Convolution 128 -> 128)
	level = 3;
	#pragma omp parallel for private(j) schedule(dynamic,1) num_threads(NUMBER_OF_THREADS)
	for (i = 0; i < cshape[level][0]; i++) {
		for (j = 0; j < cshape[level][1]; j++) {
#if SPARSE_CONVOLUTIONS
			convolution_3_x_3_sparse(mem_block1[j], wc_sparse[level][i][j], mem_block2[i], cur_size);
#else
			convolution_3_x_3(mem_block1[j], wc[level][i][j], mem_block2[i], cur_size, cur_stride);
#endif	// SPARSE_CONVOLUTIONS
		}
		add_bias_and_relu(mem_block2[i], bc[level][i], cur_size);
	}
	reset_mem_block(mem_block1);

// 	printf("level, cshape[level][0], cur_size, lines: %d, %d, %d, %d\n", level, cshape[level][0], cur_size, cshape[level][0]*cur_size*cur_size);
//     	flatten(mem_block2, mem_block1_dense, cshape[level][0], cur_size, cur_size);
//     	output_predictions(out, only_convolution, cshape[level][0], cur_size);

	//-------------------------------------------------------------------------------------------------------------------------------

	// Layer 6 (MaxPooling)
	#pragma omp parallel for schedule(dynamic,1) num_threads(NUMBER_OF_THREADS)
	for (i = 0; i < cshape[level][0]; i++) {
		maxpooling(mem_block2[i], cur_size);
	}
	cur_size /= 2;
//	printf("(cur_size): %d\n", cur_size);

	//-------------------------------------------------------------------------------------------------------------------------------

	// Layer 7 (Convolution 128 -> 256)
	level = 4;
	#pragma omp parallel for private(j) schedule(dynamic,1) num_threads(NUMBER_OF_THREADS)
	for (i = 0; i < cshape[level][0]; i++) {
		for (j = 0; j < cshape[level][1]; j++) {
#if SPARSE_CONVOLUTIONS
			convolution_3_x_3_sparse(mem_block2[j], wc_sparse[level][i][j], mem_block1[i], cur_size);
#else
			convolution_3_x_3(mem_block2[j], wc[level][i][j], mem_block1[i], cur_size, cur_stride);
#endif	// SPARSE_CONVOLUTIONS
		}
		add_bias_and_relu(mem_block1[i], bc[level][i], cur_size);
	}
	reset_mem_block(mem_block2);

// 	printf("level, cshape[level][0], cur_size, lines: %d, %d, %d, %d\n", level, cshape[level][0], cur_size, cshape[level][0]*cur_size*cur_size);
//     	flatten(mem_block1, mem_block1_dense, cshape[level][0], cur_size, cur_size);
//     	output_predictions(out, only_convolution, cshape[level][0], cur_size);

	//-------------------------------------------------------------------------------------------------------------------------------

	// Layer 8 (Convolution 256 -> 256)
	level = 5;
	#pragma omp parallel for private(j) schedule(dynamic,1) num_threads(NUMBER_OF_THREADS)
	for (i = 0; i < cshape[level][0]; i++) {
		for (j = 0; j < cshape[level][1]; j++) {
#if SPARSE_CONVOLUTIONS
			convolution_3_x_3_sparse(mem_block1[j], wc_sparse[level][i][j], mem_block2[i], cur_size);
#else
			convolution_3_x_3(mem_block1[j], wc[level][i][j], mem_block2[i], cur_size, cur_stride);
#endif	// SPARSE_CONVOLUTIONS
		}
		add_bias_and_relu(mem_block2[i], bc[level][i], cur_size);
	}
	reset_mem_block(mem_block1);

// 	printf("level, cshape[level][0], cur_size, lines: %d, %d, %d, %d\n", level, cshape[level][0], cur_size, cshape[level][0]*cur_size*cur_size);
//     	flatten(mem_block2, mem_block1_dense, cshape[level][0], cur_size, cur_size);
//     	output_predictions(out, only_convolution, cshape[level][0], cur_size);

	//-------------------------------------------------------------------------------------------------------------------------------

	// Layer 9 (Convolution 256 -> 256)
	level = 6;
	#pragma omp parallel for private(j) schedule(dynamic,1) num_threads(NUMBER_OF_THREADS)
	for (i = 0; i < cshape[level][0]; i++) {
		for (j = 0; j < cshape[level][1]; j++) {
#if SPARSE_CONVOLUTIONS
			convolution_3_x_3_sparse(mem_block2[j], wc_sparse[level][i][j], mem_block1[i], cur_size);
#else
			convolution_3_x_3(mem_block2[j], wc[level][i][j], mem_block1[i], cur_size, cur_stride);
#endif	// SPARSE_CONVOLUTIONS
		}
		add_bias_and_relu(mem_block1[i], bc[level][i], cur_size);
	}
	reset_mem_block(mem_block2);

// 	printf("level, cshape[level][0], cur_size, lines: %d, %d, %d, %d\n", level, cshape[level][0], cur_size, cshape[level][0]*cur_size*cur_size);
//     	flatten(mem_block1, mem_block1_dense, cshape[level][0], cur_size, cur_size);
//     	output_predictions(out, only_convolution, cshape[level][0], cur_size);

	//-------------------------------------------------------------------------------------------------------------------------------

	// Layer 10 (MaxPooling)
	#pragma omp parallel for schedule(dynamic,1) num_threads(NUMBER_OF_THREADS)
	for (i = 0; i < cshape[level][0]; i++) {
		maxpooling(mem_block1[i], cur_size);
	}
	cur_size /= 2;
//	printf("(cur_size): %d\n", cur_size);

	//-------------------------------------------------------------------------------------------------------------------------------

	// Layer 11 (Convolution 256 -> 512)
	level = 7;
	#pragma omp parallel for private(j) schedule(dynamic,1) num_threads(NUMBER_OF_THREADS)
	for (i = 0; i < cshape[level][0]; i++) {
		for (j = 0; j < cshape[level][1]; j++) {
#if SPARSE_CONVOLUTIONS
			convolution_3_x_3_sparse(mem_block1[j], wc_sparse[level][i][j], mem_block2[i], cur_size);
#else
			convolution_3_x_3(mem_block1[j], wc[level][i][j], mem_block2[i], cur_size, cur_stride);
#endif	// SPARSE_CONVOLUTIONS
		}
		add_bias_and_relu(mem_block2[i], bc[level][i], cur_size);
	}
	reset_mem_block(mem_block1);

//  	printf("level, cshape[level][0], cur_size, lines: %d, %d, %d, %d\n", level, cshape[level][0], cur_size, cshape[level][0]*cur_size*cur_size);
// 	flatten(mem_block2, mem_block1_dense, cshape[level][0], cur_size, cur_size);
// 	output_predictions(out, only_convolution, cshape[level][0], cur_size);

	//-------------------------------------------------------------------------------------------------------------------------------

	// Layer 12 (Convolution 512 -> 512)
	level = 8;
	#pragma omp parallel for private(j) schedule(dynamic,1) num_threads(NUMBER_OF_THREADS)
	for (i = 0; i < cshape[level][0]; i++) {
		for (j = 0; j < cshape[level][1]; j++) {
#if SPARSE_CONVOLUTIONS
			convolution_3_x_3_sparse(mem_block2[j], wc_sparse[level][i][j], mem_block1[i], cur_size);
#else
			convolution_3_x_3(mem_block2[j], wc[level][i][j], mem_block1[i], cur_size, cur_stride);
#endif	// SPARSE_CONVOLUTIONS
		}
		add_bias_and_relu(mem_block1[i], bc[level][i], cur_size);
	}
	reset_mem_block(mem_block2);

// 	printf("level, cshape[level][0], cur_size, lines: %d, %d, %d, %d\n", level, cshape[level][0], cur_size, cshape[level][0]*cur_size*cur_size);
//     	flatten(mem_block1, mem_block1_dense, cshape[level][0], cur_size, cur_size);
//	output_predictions(out, only_convolution, cshape[level][0], cur_size);

	//-------------------------------------------------------------------------------------------------------------------------------

	// Layer 13 (Convolution 512 -> 512)
	level = 9;
	#pragma omp parallel for private(j) schedule(dynamic,1) num_threads(NUMBER_OF_THREADS)
	for (i = 0; i < cshape[level][0]; i++) {
		for (j = 0; j < cshape[level][1]; j++) {
#if SPARSE_CONVOLUTIONS
			convolution_3_x_3_sparse(mem_block1[j], wc_sparse[level][i][j], mem_block2[i], cur_size);
#else
			convolution_3_x_3(mem_block1[j], wc[level][i][j], mem_block2[i], cur_size, cur_stride);
#endif	// SPARSE_CONVOLUTIONS
		}
		add_bias_and_relu(mem_block2[i], bc[level][i], cur_size);
	}
	reset_mem_block(mem_block1);

// 	printf("level, cshape[level][0], cur_size, lines: %d, %d, %d, %d\n", level, cshape[level][0], cur_size, cshape[level][0]*cur_size*cur_size);
// 	flatten(mem_block2, mem_block1_dense, cshape[level][0], cur_size, cur_size);
// 	output_predictions(out, only_convolution, cshape[level][0], cur_size);

	//-------------------------------------------------------------------------------------------------------------------------------

	// Layer 14 (MaxPooling)
	#pragma omp parallel for schedule(dynamic,1) num_threads(NUMBER_OF_THREADS)
	for (i = 0; i < cshape[level][0]; i++) {
		maxpooling(mem_block2[i], cur_size);
	}
	cur_size /= 2;
//	printf("(cur_size): %d\n", cur_size);

	//-------------------------------------------------------------------------------------------------------------------------------

	// Layer 15 (Convolution 512 -> 512)
	level = 10;
	#pragma omp parallel for private(j) schedule(dynamic,1) num_threads(NUMBER_OF_THREADS)
	for (i = 0; i < cshape[level][0]; i++) {
		for (j = 0; j < cshape[level][1]; j++) {
#if SPARSE_CONVOLUTIONS
			convolution_3_x_3_sparse(mem_block2[j], wc_sparse[level][i][j], mem_block1[i], cur_size);
#else
			convolution_3_x_3(mem_block2[j], wc[level][i][j], mem_block1[i], cur_size, cur_stride);
#endif	// SPARSE_CONVOLUTIONS
		}
		add_bias_and_relu(mem_block1[i], bc[level][i], cur_size);
	}
	reset_mem_block(mem_block2);

// 	printf("level, cshape[level][0], cur_size, lines: %d, %d, %d, %d\n", level, cshape[level][0], cur_size, cshape[level][0]*cur_size*cur_size);
// 	flatten(mem_block1, mem_block1_dense, cshape[level][0], cur_size, cur_size);
// 	output_predictions(out, only_convolution, cshape[level][0], cur_size);

	//-------------------------------------------------------------------------------------------------------------------------------

	// Layer 16 (Convolution 512 -> 512)
	level = 11;
	#pragma omp parallel for private(j) schedule(dynamic,1) num_threads(NUMBER_OF_THREADS)
	for (i = 0; i < cshape[level][0]; i++) {
		for (j = 0; j < cshape[level][1]; j++) {
#if SPARSE_CONVOLUTIONS
			convolution_3_x_3_sparse(mem_block1[j], wc_sparse[level][i][j], mem_block2[i], cur_size);
#else
			convolution_3_x_3(mem_block1[j], wc[level][i][j], mem_block2[i], cur_size, cur_stride);
#endif	// SPARSE_CONVOLUTIONS
		}
		add_bias_and_relu(mem_block2[i], bc[level][i], cur_size);
	}
	reset_mem_block(mem_block1);

// 	printf("level, cshape[level][0], cur_size, lines: %d, %d, %d, %d\n", level, cshape[level][0], cur_size, cshape[level][0]*cur_size*cur_size);
// 	flatten(mem_block2, mem_block1_dense, cshape[level][0], cur_size, cur_size);
// 	output_predictions(out, only_convolution, cshape[level][0], cur_size);

	//-------------------------------------------------------------------------------------------------------------------------------

	// Layer 17 (Convolution 512 -> 512)
	level = 12;
	#pragma omp parallel for private(j) schedule(dynamic,1) num_threads(NUMBER_OF_THREADS)
	for (i = 0; i < cshape[level][0]; i++) {
		for (j = 0; j < cshape[level][1]; j++) {
#if SPARSE_CONVOLUTIONS
			convolution_3_x_3_sparse(mem_block2[j], wc_sparse[level][i][j], mem_block1[i], cur_size);
#else
			convolution_3_x_3(mem_block2[j], wc[level][i][j], mem_block1[i], cur_size, cur_stride);
#endif	// SPARSE_CONVOLUTIONS
		}
		add_bias_and_relu(mem_block1[i], bc[level][i], cur_size);
	}
	reset_mem_block(mem_block2);

// 	printf("level, cshape[level][0], cur_size, lines: %d, %d, %d, %d\n", level, cshape[level][0], cur_size, cshape[level][0]*cur_size*cur_size);
// 	flatten(mem_block1, mem_block1_dense, cshape[level][0], cur_size, cur_size);
// 	output_predictions(out, only_convolution, cshape[level][0], cur_size);

	//-------------------------------------------------------------------------------------------------------------------------------

	// Layer 18 (MaxPooling)
	#pragma omp parallel for schedule(dynamic,1) num_threads(NUMBER_OF_THREADS)
	for (i = 0; i < cshape[level][0]; i++) {
		maxpooling(mem_block1[i], cur_size);
	}

	cur_size /= 2;
//	printf("(cur_size): %d\n", cur_size);

	//-------------------------------------------------------------------------------------------------------------------------------

	// Layer 19 (Flatten)
// 	printf("level, cshape[level][0], cur_size, lines: %d, %d, %d, %d\n", level, cshape[level][0], cur_size, cshape[level][0]*cur_size*cur_size);
 	flatten(mem_block1, mem_block1_dense, cshape[level][0], cur_size, cur_size);
// 	output_predictions(out, only_convolution, cshape[level][0], cur_size);


	if (only_convolution == 1) {
		return;
	}

	//-------------------------------------------------------------------------------------------------------------------------------

	// Layer 20 (Dense)
	level = 0;
	dense(mem_block1_dense, wd[level], mem_block2_dense, dshape[level][0], dshape[level][1]);
	add_bias_and_relu_flatten(mem_block2_dense, bd[level], dshape[level][1], 1);

//  	printf("level, dshape[level][0], cur_size, lines: %d, %d, %d, %d\n", level, dshape[level][0], cur_size, 1);
//     	output_predictions(out, only_convolution, dshape[level][0], cur_size);

	reset_mem_block_dense(mem_block1_dense);

	// Layer 21 (Dense)
	level = 1;
	dense(mem_block2_dense, wd[level], mem_block1_dense, dshape[level][0], dshape[level][1]);
	add_bias_and_relu_flatten(mem_block1_dense, bd[level], dshape[level][1], 1);

	softmax(mem_block1_dense, dshape[level][1]);
//	dump_memory_structure_dense_to_file(mem_block2_dense, dshape[level][1]);

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
		lvls = 13;
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
		get_VGG16_predict(only_convolution);
//		get_VGG16_predict(results, only_convolution);

		gettimeofday(&tEnd, NULL);
		deltaTime = get_seconds(tStart, tEnd);
		printf("Infer image %s: %.3lf sec\n", buf, deltaTime);

// 		output_predictions(results, only_convolution);
		output_predictions(results, only_convolution, 512, 1);
	}

	free_memory();
	fclose(file_list);

	return 0;
}
