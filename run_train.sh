#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
data_root="${DATA_ROOT:-/home/xxj/Data}"
gpu_id="${GPU_ID:-0}"
workers="${NUM_WORKERS:-4}"
batch_size="${BATCH_SIZE:-8}"
epochs="${EPOCHS:-200}"

if [[ $# -ne 1 ]]; then
  printf 'Usage: %s {levircd|levircdplus|cdd|whucd|sysucd}\n' "$0" >&2
  exit 2
fi

case "$1" in
  levircd) name="LEVIR"; root="$data_root/LEVIR_CD"; image_a=A; image_b=B; label=label ;;
  levircdplus) name="LEVIR-CD+"; root="$data_root/LEVIR_CD_PLUS/cut_data"; image_a=A; image_b=B; label=label ;;
  cdd) name="CDD"; root="$data_root/CDD"; image_a=A; image_b=B; label=OUT ;;
  whucd) name="WHU"; root="$data_root/WHUCD"; image_a=A; image_b=B; label=label ;;
  sysucd) name="SYSU"; root="$data_root/SYSU_CD"; image_a=time1; image_b=time2; label=label ;;
  *) printf 'Unknown dataset key: %s\n' "$1" >&2; exit 2 ;;
esac

for split in train val; do
  for folder in "$image_a" "$image_b" "$label"; do
    [[ -d "$root/$split/$folder" ]] || { printf 'Missing data directory: %s\n' "$root/$split/$folder" >&2; exit 1; }
  done
done

weight="$repo_root/pretrained_model/pvt_v2_b2.pth"
[[ -f "$weight" ]] || { printf 'Missing PVT-v2-b2 pretrained weight: %s\n' "$weight" >&2; exit 1; }

cd "$repo_root"
CUDA_VISIBLE_DEVICES="$gpu_id" python train.py \
  --data_name "$name" \
  --dataset-root "$root" \
  --image-dir-a "$image_a" --image-dir-b "$image_b" --label-dir "$label" \
  --save_path "$repo_root/UA-BCD_work_dirs" \
  --num-workers "$workers" --batchsize "$batch_size" --epoch "$epochs"
