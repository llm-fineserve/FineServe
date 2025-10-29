#!/bin/bash


## Setup global paths
ROOT_DIR=/home/jovyan/shared-dir/sbchoi/workspace/FineServe
CFG_YAML_ROOT_DIR=$ROOT_DIR/examples/placement_yamls
WORKLOAD_JSON_ROOT_DIR=$ROOT_DIR


## Cleaning up previous experiments
rm /dev/shm/* 
pkill python3
sleep 1
#bash $ROOT_DIR/scripts/stop_mps.sh

## setup variable names (for naming log directory and files) and paths
baseline=fineserve
tc=simple_example
echo "Running $baseline with test case : $tc"

root_result_dir=$ROOT_DIR/$tc-results
mkdir -p $root_result_dir
echo "LOGGING DIRECTORY: ${root_result_dir}"

yaml_file=$CFG_YAML_ROOT_DIR/fineserve-2models_cfg.yaml
wkload_file=$WORKLOAD_JSON_ROOT_DIR/output.json
echo "testing yaml_file: $yaml_file "
echo "with workload_file: $wkload_file "

## start MPS
#bash $ROOT_DIR/scripts/start_mps.sh
log_dir=$root_result_dir/$baseline-sched_logs
log_name="$tc"_"$baseline".log
echo "scheduling log will be stored under $log_dir"
mkdir -p $log_dir

## Launch!!
bash $ROOT_DIR/scripts/run_sched_and_manager.sh $yaml_file $wkload_file saab fineserve  > $log_dir/$log_name &
pid=$!
wait $pid
pkill python3
sleep 5

## stop MPS
#bash $ROOT_DIR/scripts/stop_mps.sh

