#!/bin/bash
RPD_DIR=/Projects/rocmProfileData
BASE_DIR=/Projects
# VLLM_DIR=/Projects/vllm

TP="8 4 2"
ITER=10
GEN_LEN="1,32,128"
INPUT_LEN="512,1024,2048,3072"


RPD_PROFILE="--rpd"
FINAL_TRACE_DIR=$BASE_DIR/H100_Mega_trace_dumps

if [[ -v RPD_PROFILE ]] ;
then
    ITER=1
    GEN_LEN="1 32 128"
    INPUT_LEN="512 1024 2048 3072"
    echo "Setting ITER=1"
    echo "Setting INPUT & GEN LEN: $INPUT_LEN  -- $GEN_LEN"
    if test -d "$FINAL_TRACE_DIR"
    then
        echo "RPD dump directory already exists: $FINAL_TRACE_DIR"
    else
        echo "creating rpd traces directory tree: $FINAL_TRACE_DIR"
        mkdir -p $FINAL_TRACE_DIR/{Chrome_trace/{TP1,TP2,TP4,TP8},Top_kernels/{TP1,TP2,TP4,TP8},RPD_files/{TP1,TP2,TP4,TP8}}
    fi
fi

for tp in $TP;
do
    for gen_len in $GEN_LEN;
    do
        for input_len in $INPUT_LEN;
        do
            if [[ -v RPD_PROFILE ]] ;
            then
                rm $BASE_DIR/trace.rpd
                python -m rocpd.schema --create $BASE_DIR/trace.rpd
            fi

            echo "============== RUNNING $MODEL $input_len $gen_len Triton=$VLLM_USE_FLASH_ATTN_TRITON TunableOps=$PYTORCH_TUNABLEOP_ENABLED TP=$tp =================="
            torchrun --standalone --nnodes=1 --nproc-per-node=$tp benchmarks/benchmark_latency.py \
                     --model mistralai/Mixtral-8x7B-v0.1 --batch-size 1 --input-len $input_len --output-len $gen_len \
                     --tensor-parallel-size $tp --num-iters $ITER --download-dir /data/Mixtral/ --report $RPD_PROFILE --dtype float16

            if [[ -v RPD_PROFILE ]] ;
            then
                TRACE_FILE=$FINAL_TRACE_DIR/Chrome_trace/TP${tp}/trace_H100_${input_len}_${gen_len}_TP${tp}.json
                echo "INFO: Creating Trace JSON file $TRACE_FILE"
                python $RPD_DIR/tools/rpd2tracing.py --format object $BASE_DIR/trace.rpd $TRACE_FILE

                TOP_K_FILE=$FINAL_TRACE_DIR/Top_kernels/TP${tp}/top_H100_${input_len}_${gen_len}_TP${tp}.csv
                echo "INFO: Creating CSV file for top kernels $TOP_K_FILE"
                sqlite3 $BASE_DIR/trace.rpd ".mode csv" ".headers on" ".output ${TOP_K_FILE}" "select * from top;"

                NEW_RPD_FILE_NAME=$FINAL_TRACE_DIR/RPD_files/TP${tp}/trace_H100_${input_len}_${gen_len}_TP${tp}.rpd
                echo "INFO: Renaming rpd file to $NEW_RPD_FILE_NAME"
                mv $BASE_DIR/trace.rpd $NEW_RPD_FILE_NAME
            fi
        done
    done
done