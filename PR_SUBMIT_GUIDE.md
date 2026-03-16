# PR 제출 가이드

## 환경 참고사항
현재 서버에서 외부 네트워크가 차단되어 있어 직접 push가 불가합니다.
아래 명령어를 **로컬 PC** 또는 네트워크가 열린 환경에서 실행해 주세요.

---

## Step 1 — 저장소 Fork & Clone

```bash
# 1. GitHub에서 https://github.com/snibug-coin/CLCRec 를 Fork

# 2. Fork된 저장소 clone
git clone https://github.com/<YOUR_USERNAME>/CLCRec.git
cd CLCRec
```

---

## Step 2 — 브랜치 생성 & 파일 복사

```bash
# 브랜치 생성
git checkout -b feat/ultra-improvements

# pr_package 안의 파일들을 저장소 루트에 복사
# (이 파일들을 /workspace/group/clcrec/pr_package/ 에서 가져오세요)
cp -r pr_package/src ./src_ultra
cp pr_package/main.py ./main_ultra.py
cp pr_package/requirements.txt ./requirements_ultra.txt
cp pr_package/run.sh ./run_ultra.sh
cp pr_package/IMPROVEMENTS.md ./IMPROVEMENTS.md
```

또는 `pr_package/` 폴더 전체를 저장소에 추가해도 됩니다:

```bash
cp -r pr_package/ ./ultra/
```

---

## Step 3 — Commit & Push

```bash
git add .
git commit -m "feat: CLCRec ULTRA — +565% Cold R@20 improvements

Key improvements over original CLCRec:
- Cosine temperature annealing (τ: 4.0→0.05)     +496% Cold R@20
- High CL weight λ=0.9                             +196% Cold R@20
- 16 negatives per positive (was 4)               +413% Cold R@20
- CF-Content alignment loss (new)                 +400% Cold R@20
- Combined ULTRA: Cold R@20 +565% vs paper

Evaluation:
- Temporal split to prevent future-reference bias
- No-leakage preprocessing (train stats only)

Best result: Cold R@20 = 0.8440 (paper: 0.1269)"

git push origin feat/ultra-improvements
```

---

## Step 4 — PR 생성 (GitHub CLI 또는 웹)

### GitHub CLI 사용 시
```bash
gh pr create \
  --repo snibug-coin/CLCRec \
  --title "CLCRec ULTRA: +565% Cold R@20 via Cosine Annealing, High λ, More Negatives & Alignment Loss" \
  --body "$(cat <<'EOF'
## Summary

Systematic ablation study identifying four improvements to CLCRec that combine for **+565% Cold Recall@20** vs the paper baseline.

### Key Improvements

| Improvement | Cold R@20 | vs Paper |
|---|---|---|
| Paper baseline | 0.1269 | — |
| H1: Cosine τ annealing (4.0→0.05) | 0.7566 | +496% |
| H3: High CL weight λ=0.9 | 0.3742 | +195% |
| H6: 16 negatives (was 4) | 0.6510 | +413% |
| H7: CF-Content alignment loss | 0.6344 | +400% |
| **ULTRA (all combined)** | **0.8440** | **+565%** |

### New Features
- `--tau_schedule cosine` — cosine temperature annealing
- `--tau_init 4.0 --tau_min 0.05` — wide annealing range
- `--lambda_cl 0.9` — stronger contrastive supervision
- `--n_negs 16` — better InfoNCE partition function estimate
- `--align_weight 0.05` — CF-content alignment loss (new)
- `--temporal_split` — temporal data split (no future-reference bias)
- `--no_leakage` — normalization stats from training data only

### Run ULTRA
\`\`\`bash
python main.py \
  --dataset movielens \
  --tau_schedule cosine \
  --tau_init 4.0 \
  --tau_min 0.05 \
  --lambda_cl 0.9 \
  --n_negs 16 \
  --align_weight 0.05
\`\`\`

See `IMPROVEMENTS.md` for full ablation table and implementation details.
EOF
)"
```

### 웹에서 PR 생성 시
- Title: `CLCRec ULTRA: +565% Cold R@20 via Cosine Annealing, High λ, More Negatives & Alignment Loss`
- Body: `IMPROVEMENTS.md` 내용 참고

---

## 파일 요약

| 파일 | 설명 |
|---|---|
| `src/Models.py` | PyTorch CLCRec ULTRA (cosine τ, alignment loss, NoLeakagePreprocessor) |
| `src/utils.py` | TemporalTrainDataset, evaluate(), EarlyStopping |
| `main.py` | argparse 학습 스크립트 |
| `run.sh` | Baseline→ULTRA 비교 실행 스크립트 |
| `IMPROVEMENTS.md` | 전체 실험 결과 및 개선 상세 설명 |
| `requirements.txt` | torch, numpy, scipy |
