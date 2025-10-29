#!/bin/bash

if [[ $# != 4 ]];
then
  echo "Usage: ./run_sched_and_manager.sh {path_to_cfg_yaml} {path_to_workload_json} {local_sched_approach} {baseline}"
  exit 1
fi

path_to_cfg_yaml=$1
path_to_workload_json=$2
sched_approach=$3
baseline=$4

echo "cfg yaml: ${path_to_cfg_yaml}"
echo "workload json: ${path_to_workload_json}"

if [[ $baseline == 'spart' ]];
then
python3 fineserve/entrypoint.py --manager --scheduler  \
--node-rank 0 \
--model-config  $path_to_cfg_yaml \
--workload-file $path_to_workload_json \
--cost-file examples/cost_files/apex-cost.csv \
--schedule-approach fcfs \
--restrict-gpu-memory-utilization

else

python3 fineserve/entrypoint.py --manager --scheduler  \
--node-rank 0 \
--model-config  $path_to_cfg_yaml \
--workload-file $path_to_workload_json \
--cost-file examples/cost_files/apex-cost.csv \
--schedule-approach $sched_approach
fi
