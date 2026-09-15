#!/usr/bin/env bash
# Race domain: RF/LR main effects + SOI, then classification.
# Seeds match existing models/race artifacts.
set -euo pipefail
cd "$(dirname "$0")"

DATA="data/race_ineq_dataset.csv"
OUT="models/race"

# 1) RF — main effects (seed 102)
python3 code/feature_importance_RF.py \
  --file_path "$DATA" \
  --model_save_dir "$OUT" \
  --importance_type mdi \
  --initial_seed_starter 102 \
  --no-add_SOI

# 2) LR — main effects (seed 99)
python3 code/feature_sign_Logistic.py \
  --file_path "$DATA" \
  --model_save_dir "$OUT" \
  --initial_seed_starter 99 \
  --no-add_SOI

# 3) RF — SOI (seed 99)
python3 code/feature_importance_RF.py \
  --file_path "$DATA" \
  --model_save_dir "$OUT" \
  --importance_type mdi \
  --initial_seed_starter 99 \
  --add_SOI

# 4) LR — SOI (seed 99)
python3 code/feature_sign_Logistic.py \
  --file_path "$DATA" \
  --model_save_dir "$OUT" \
  --initial_seed_starter 99 \
  --add_SOI

# 5) Classification — RF, no SOI (seed 300)
python3 code/ml_classification.py "$DATA" "$OUT" 300

echo "✅ run_race.sh finished"
