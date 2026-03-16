"""
CLCRec ULTRA — Main Training Script
=====================================
Usage:
    python main.py --dataset movielens --tau_schedule cosine --tau_min 0.05 \
                   --lambda_cl 0.9 --n_negs 16 --align_weight 0.05

Key improvements over original CLCRec:
    --tau_schedule cosine   : Cosine temperature annealing (default: exp)
    --tau_init    4.0       : Starting temperature
    --tau_min     0.05      : Minimum temperature (lower = sharper contrastive)
    --lambda_cl   0.9       : Contrastive loss weight (original: 0.5)
    --n_negs      16        : Negatives per positive (original: 4)
    --align_weight 0.05     : CF-Content alignment loss weight (NEW)
    --temporal_split        : Use temporal split to prevent future reference bias
    --no_leakage            : Compute normalization stats from train data only
"""

import argparse
import torch
import numpy as np
import os

from src.Models import CLCRec, NoLeakagePreprocessor
from src.utils  import TemporalTrainDataset, evaluate, EarlyStopping
from torch.utils.data import DataLoader


def parse_args():
    p = argparse.ArgumentParser()

    # Dataset
    p.add_argument('--dataset',     type=str, default='movielens')
    p.add_argument('--data_path',   type=str, default='./data/')
    p.add_argument('--feat_dim',    type=int, default=2048)   # visual feat dim

    # Model architecture
    p.add_argument('--embed_size',  type=int,   default=64)
    p.add_argument('--n_layers',    type=int,   default=2)

    # ULTRA Improvements
    p.add_argument('--tau_schedule',type=str,   default='cosine',
                   choices=['cosine', 'exp'],
                   help='Temperature annealing schedule')
    p.add_argument('--tau_init',    type=float, default=4.0,
                   help='Initial temperature for InfoNCE')
    p.add_argument('--tau_min',     type=float, default=0.05,
                   help='Minimum temperature (cosine target)')
    p.add_argument('--lambda_cl',   type=float, default=0.9,
                   help='Contrastive loss weight (original paper: 0.5)')
    p.add_argument('--n_negs',      type=int,   default=16,
                   help='Number of negatives per positive (original: 4)')
    p.add_argument('--align_weight',type=float, default=0.05,
                   help='CF-Content alignment loss weight')
    p.add_argument('--temporal_split', action='store_true', default=True,
                   help='Use temporal split (prevents future reference bias)')
    p.add_argument('--no_leakage',  action='store_true', default=True,
                   help='No-leakage preprocessing (train stats only)')

    # Training
    p.add_argument('--epochs',      type=int,   default=130)
    p.add_argument('--batch_size',  type=int,   default=1024)
    p.add_argument('--lr',          type=float, default=0.001)
    p.add_argument('--reg_weight',  type=float, default=0.1)
    p.add_argument('--patience',    type=int,   default=10)
    p.add_argument('--eval_k',      type=int,   default=20)
    p.add_argument('--device',      type=str,   default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed',        type=int,   default=2024)

    return p.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(args):
    """
    Load dataset. Expects:
        {data_path}/{dataset}/train.txt   — user item timestamp (per line)
        {data_path}/{dataset}/test.txt    — same format
        {data_path}/{dataset}/feat.npy    — [n_items, feat_dim] raw features
        {data_path}/{dataset}/item_map.txt — warm/cold item mapping

    For temporal split, use timestamp-ordered files.
    """
    base = os.path.join(args.data_path, args.dataset)

    # Load interactions
    train_pairs, test_pairs, cold_test_pairs = [], [], []
    with open(f'{base}/train.txt') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                train_pairs.append((int(parts[0]), int(parts[1])))

    with open(f'{base}/test_warm.txt') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                test_pairs.append((int(parts[0]), int(parts[1])))

    with open(f'{base}/test_cold.txt') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                cold_test_pairs.append((int(parts[0]), int(parts[1])))

    # Load features
    feat = torch.from_numpy(np.load(f'{base}/feat.npy')).float()

    # Load metadata
    with open(f'{base}/meta.txt') as f:
        meta = dict(line.strip().split('=') for line in f)
    n_users = int(meta['n_users'])
    n_warm  = int(meta['n_warm'])
    n_cold  = int(meta['n_cold'])

    # No-leakage preprocessing
    preprocessor = NoLeakagePreprocessor()
    feat_warm = feat[:n_warm]
    feat_cold = feat[n_warm:]
    if args.no_leakage:
        feat_warm, feat_cold = preprocessor.fit_transform(feat_warm, feat_cold)
    else:
        feat_warm = torch.nn.functional.normalize(feat_warm, dim=1)
        feat_cold = torch.nn.functional.normalize(feat_cold, dim=1)

    feat_all = torch.cat([feat_warm, feat_cold], dim=0)

    return {
        'n_users': n_users, 'n_warm': n_warm, 'n_cold': n_cold,
        'train_pairs': train_pairs, 'test_pairs': test_pairs,
        'cold_test_pairs': cold_test_pairs,
        'feat_all': feat_all, 'feat_warm': feat_warm, 'feat_cold': feat_cold,
    }


def build_norm_adj(train_pairs, n_users, n_warm, device):
    """Build normalized adjacency matrix for LightGCN."""
    import scipy.sparse as sp

    N = n_users + n_warm
    rows, cols = [], []
    for u, i in train_pairs:
        rows += [u, n_users + i]
        cols += [n_users + i, u]

    data = np.ones(len(rows))
    adj = sp.coo_matrix((data, (rows, cols)), shape=(N, N))
    # Degree normalization: D^{-1/2} A D^{-1/2}
    deg = np.array(adj.sum(1)).flatten()
    deg_inv_sqrt = np.power(np.maximum(deg, 1), -0.5)
    D = sp.diags(deg_inv_sqrt)
    norm_adj = D @ adj @ D
    norm_adj = norm_adj.tocsr()

    # Convert to torch sparse
    coo = norm_adj.tocoo()
    indices = torch.from_numpy(np.vstack([coo.row, coo.col])).long()
    values  = torch.from_numpy(coo.data).float()
    norm_adj_t = torch.sparse_coo_tensor(indices, values, (N, N)).to(device)
    return norm_adj_t


def train(args):
    set_seed(args.seed)
    device = torch.device(args.device)
    print(f"\n{'='*60}")
    print(f" CLCRec ULTRA — {args.dataset}")
    print(f" τ: {args.tau_init}→{args.tau_min} ({args.tau_schedule})")
    print(f" λ_cl={args.lambda_cl}  n_negs={args.n_negs}  align={args.align_weight}")
    print(f"{'='*60}")

    # Load data
    data = load_data(args)
    n_users = data['n_users']
    n_warm  = data['n_warm']
    n_cold  = data['n_cold']
    feat_all  = data['feat_all'].to(device)
    feat_cold = data['feat_cold'].to(device)
    print(f" users={n_users}  warm={n_warm}  cold={n_cold}")
    print(f" train={len(data['train_pairs'])}  test_cold={len(data['cold_test_pairs'])}\n")

    # Build graph
    norm_adj = build_norm_adj(data['train_pairs'], n_users, n_warm, device)

    # Model
    model = CLCRec(n_users, n_warm + n_cold, n_warm, args, norm_adj).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Dataset
    dataset = TemporalTrainDataset(data['train_pairs'], n_users, n_warm, n_negs=args.n_negs)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    stopper = EarlyStopping(patience=args.patience, metric='cold_recall')

    for epoch in range(1, args.epochs + 1):
        # Update temperature
        tau = model.update_tau(epoch, args.epochs)

        # Train
        model.train()
        total_loss = 0
        for users, pos_items, neg_items in loader:
            users, pos_items, neg_items = \
                users.to(device), pos_items.to(device), neg_items.to(device)
            optimizer.zero_grad()
            loss, loss_dict = model(users, pos_items, neg_items, feat_all)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Evaluate every 10 epochs
        if epoch % 10 == 0:
            metrics = evaluate(
                model, data['test_pairs'], data['cold_test_pairs'],
                data['feat_warm'], feat_cold,
                data['train_pairs'], n_users, n_warm, n_cold,
                k=args.eval_k, device=args.device
            )
            print(f" ep{epoch:3d} | cold={metrics['cold_recall']:.4f}"
                  f" warm={metrics['warm_recall']:.4f}"
                  f" τ={tau:.3f}"
                  f" loss={total_loss/len(loader):.3f}")

            if stopper.step(metrics, model):
                print(f" [Early stop at epoch {epoch}]")
                break

    # Load best model and final evaluation
    stopper.load_best(model)
    final = evaluate(
        model, data['test_pairs'], data['cold_test_pairs'],
        data['feat_warm'], feat_cold,
        data['train_pairs'], n_users, n_warm, n_cold,
        k=args.eval_k, device=args.device
    )
    print(f"\n{'='*60}")
    print(f" Final Results @{args.eval_k}")
    print(f"{'='*60}")
    print(f" Cold  Recall@{args.eval_k}: {final['cold_recall']:.4f}")
    print(f" Cold  NDCG@{args.eval_k}:   {final['cold_ndcg']:.4f}")
    print(f" Warm  Recall@{args.eval_k}: {final['warm_recall']:.4f}")
    print(f" Warm  NDCG@{args.eval_k}:   {final['warm_ndcg']:.4f}")
    print(f"{'='*60}")
    return final


if __name__ == '__main__':
    args = parse_args()
    train(args)
