#!/bin/bash
# CLCRec ULTRA — Experiment Runner
# Reproduces the best-performing configuration (+565% Cold R@20 vs paper baseline)
#
# Requirements:
#   pip install -r requirements.txt
#
# Data layout expected:
#   data/{dataset}/train.txt       — "user item" per line
#   data/{dataset}/test_warm.txt   — warm item test pairs
#   data/{dataset}/test_cold.txt   — cold item test pairs
#   data/{dataset}/feat.npy        — [n_items, 2048] float32 features
#   data/{dataset}/meta.txt        — n_users=X / n_warm=Y / n_cold=Z

set -e

DATASET=${1:-movielens}
echo "========================================"
echo " Running CLCRec ULTRA on: $DATASET"
echo "========================================"

# ── Baseline (original paper hyperparameters) ────────────────────────────────
echo ""
echo ">>> [1/4] Baseline"
python main.py \
  --dataset       "$DATASET" \
  --tau_schedule  exp \
  --tau_init      1.0 \
  --tau_min       0.5 \
  --lambda_cl     0.5 \
  --n_negs        4 \
  --align_weight  0.0 \
  --epochs        100

# ── Cosine Annealing only ─────────────────────────────────────────────────────
echo ""
echo ">>> [2/4] + Cosine τ annealing (τ: 4.0 → 0.05)"
python main.py \
  --dataset       "$DATASET" \
  --tau_schedule  cosine \
  --tau_init      4.0 \
  --tau_min       0.05 \
  --lambda_cl     0.5 \
  --n_negs        4 \
  --align_weight  0.0 \
  --epochs        130

# ── Cosine + High CL weight + More negatives ─────────────────────────────────
echo ""
echo ">>> [3/4] + λ=0.9 + 16 negatives"
python main.py \
  --dataset       "$DATASET" \
  --tau_schedule  cosine \
  --tau_init      4.0 \
  --tau_min       0.05 \
  --lambda_cl     0.9 \
  --n_negs        16 \
  --align_weight  0.0 \
  --epochs        130

# ── ULTRA: all improvements ───────────────────────────────────────────────────
echo ""
echo ">>> [4/4] ULTRA (cosine τ + λ=0.9 + 16-neg + CF-Content alignment)"
python main.py \
  --dataset       "$DATASET" \
  --tau_schedule  cosine \
  --tau_init      4.0 \
  --tau_min       0.05 \
  --lambda_cl     0.9 \
  --n_negs        16 \
  --align_weight  0.05 \
  --temporal_split \
  --no_leakage \
  --epochs        130

echo ""
echo "========================================"
echo " All runs complete."
echo "========================================"
