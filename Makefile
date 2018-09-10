
TARGET	= VGG-16_CPU_cifar
#TARGET?= ResNet-18_CPU_cifar
#TARGET?= MobileNet_CPU_cifar

CFLAGS  = -g -w -lm -fopenmp
LDFLAGS	=
CC		= gcc

SRC		= src/
INCLUDE	= include/

VAR_PLAIN	=
VAR_DC = -DSPARSE_CONVOLUTIONS=1
VAR_FISHER 	= -DFISHER_PRUNING=1
VAR_TTQ		= -DSPARSE_CONVOLUTIONS=1

MODEL?=vgg-16
#MODEL?=resnet-18
#MODEL?=mobilenet

all:	plain dc fisher ttq run_plain run_dc run_fisher run_ttq

plain:
	$(CC) $(VAR_PLAIN) $(TARGET).c $(SRC)* -I$(INCLUDE) $(CFLAGS) -o $(TARGET)_plain

dc:
	$(CC) $(VAR_DC) $(TARGET).c $(SRC)* -I$(INCLUDE) $(CFLAGS) -o $(TARGET)_dc

fisher:
	$(CC) $(VAR_FISHER) $(TARGET).c $(SRC)* -I$(INCLUDE) $(CFLAGS) -o $(TARGET)_fisher

ttq:
	$(CC) $(VAR_TTQ) $(TARGET).c $(SRC)* -I$(INCLUDE) $(CFLAGS) -o $(TARGET)_ttq


run_plain:
	@echo $ "\n===RUNNING PLAIN MODE"
	./$(TARGET)_plain weights/$(MODEL)_plain.txt img/filelist_cf.txt result_cf_plain.txt

run_dc:
	@echo $ "\n===RUNNING DEEP-COMPRESSION"
	./$(TARGET)_dc weights/$(MODEL)_dc.txt img/filelist_cf.txt result_cf_dc.txt

run_fisher:
	@echo $ "\n===RUNNING FISHER_PRUNING"
	./$(TARGET)_fisher weights/$(MODEL)_fisher.txt img/filelist_cf.txt result_cf_fisher.txt

run_ttq:
	@echo $ "\n===RUNNING QUANTIZATION"
	./$(TARGET)_ttq weights/$(MODEL)_ttq.txt img/filelist_cf.txt result_cf_ttq.txt

clean:
	rm -f $(TARGET)_*
