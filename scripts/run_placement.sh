#!/bin/bash

###
# Example script for executing global placement algorithm
###
ROOT_DIR=/home/jovyan/shared-dir/sbchoi/workspace/FineServe


## change file to fineserve-1models.yaml if you want to try single model version
python3 $ROOT_DIR/fineserve/finesched/placement/placement_optimizer.py  \
	--workload-file $ROOT_DIR/examples/fineserve-2models.yaml \
	--cost-file $ROOT_DIR/examples/cost_files/apex-cost.csv \
	--placement-opt slab-aware ## change this option if you want to try other options

