#!/bin/bash
ARCH=./arch.json
COMPILE_KV260=./outputs_kv260

vai_c_tensorflow2 \
	--model  	./models/adam_100_da_best_quantized.h5 \
	--arch   	${ARCH} \
	--output_dir ${COMPILE_KV260} \
	--net_name   adam_100_da_best_quantized