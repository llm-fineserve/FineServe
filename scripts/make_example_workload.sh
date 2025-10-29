#!/bin/bash

python3 fineserve/utils/workload_utils.py \
    --dataset-source /home/jovyan/shared-dir/hpclab/hub/ShareGPT_V3_unfiltered_cleaned_split.json \
    --workload_info_from_yaml True \
    --output-file examples/basic/fineserve_1models_workload.json \
    --model-yaml examples/basic/fineserve-1models.yaml
