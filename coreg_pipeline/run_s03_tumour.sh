#!/bin/bash
cd "$(dirname "$0")"
# Network list is read by the PYTHON script from a file -- no shell array, so
# no expansion hazard. bash 3.2 here has no `mapfile`; an earlier driver used
# it, got an empty array, and s03 fell back to all networks.
echo "[$(date +%H:%M:%S)] tumour Layer 2 starting"
python s03_layer2.py --networks-file nets_tumour.txt --seeds 0 \
  >> logs/s03_tumour.log 2>&1
echo "[$(date +%H:%M:%S)] exit=$? | written: $(ls results/layer2/l2_tumour_*.tsv 2>/dev/null | wc -l)/31"
