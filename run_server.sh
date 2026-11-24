#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export YOLO_AUTOINSTALL=false
PYTHON_CMD="${PYTHON_CMD:-python}"
EXPERIMENT_DEVICE="${EXPERIMENT_DEVICE:-0}"
mode="${1:-smoke}"
case "$mode" in
  smoke|full|all) ;;
  *) echo 'Usage: bash run_server.sh [smoke|full|all]' >&2; exit 2 ;;
esac
"$PYTHON_CMD" check_server.py --device "$EXPERIMENT_DEVICE"
variants=(baseline)
frame_args=()
if [[ "$mode" == smoke ]]; then
  frame_args=(--max-frames 50)
elif [[ "$mode" == all ]]; then
  variants=(baseline change1 change2 model_compare)
fi
for script in 02_heatmap.py 03_counting.py; do
  for variant in "${variants[@]}"; do
    "$PYTHON_CMD" "$script" --variant "$variant" --no-display \
      --device "$EXPERIMENT_DEVICE" "${frame_args[@]}"
  done
done
"$PYTHON_CMD" collect_results.py
