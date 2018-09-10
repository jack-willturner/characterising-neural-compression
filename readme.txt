

You need to include the weights file (e.g. vgg_plain.txt) file generated from scratchpad/codes/pytorch-cifar/


compile:	gcc -O3 VGG16_CPU_cifar.c -lm -fopenmp -o VGG16_CPU_cifar 

run (example):	./VGG-16_CPU_cifar weights/vgg.txt img/filelist_cf.txt result_cf_conv.txt 1	(only convolutional layers)
run (example):	./VGG-16_CPU_cifar weights/vgg.txt img/filelist_cf.txt result_cf.txt		(all layers)

