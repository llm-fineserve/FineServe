#!/bin/bash
# the following must be performed with root privilege
# >>> sudo sh scripts/stop_mps.sh

#if [ "$#" -ne 1 ]; then
#    echo "Usage: $0 <mps_dir>"
#    echo "bash scripts/stop_mps.sh /tmp"
#    exit 1
#fi


MPSDIR=/tmp
GPUIDS="0"

echo "terminating MPS for GPU  $GPUIDS"

echo quit | nvidia-cuda-mps-control
pkill -f nvidia-cuda-mps-control
nvidia-smi -i $GPUIDS --compute-mode=DEFAULT
rm -rf ${MPSDIR}/nvidia-mps
rm -rf ${MPSDIR}/nvidia-log
