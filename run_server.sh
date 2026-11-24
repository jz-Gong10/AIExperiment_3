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
if [[ "$mode" == all ]]; then
  variants=(baseline change1 change2 model_compare)
fi
for script in 02_heatmap.py 03_counting.py; do
  for variant in "${variants[@]}"; do
    # Keep this array nonempty: Bash 4.3 and older reject empty arrays with set -u.
    command=("$PYTHON_CMD" "$script" --variant "$variant" --no-display \
      --device "$EXPERIMENT_DEVICE")
    if [[ "$mode" == smoke ]]; then
      command+=(--max-frames 50)
    fi
    "${command[@]}"
  done
done
"$PYTHON_CMD" collect_results.py
