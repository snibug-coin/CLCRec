"""
Utility functions for CLCRec training with ULTRA improvements.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict


class TemporalTrainDataset(Dataset):
    """
    Dataset with temporal split to prevent future reference bias.

    Split strategy:
        Train:  interactions with timestamp <= T_train (80%)
        Valid:  interactions with T_train < timestamp <= T_valid (90%)
        Test:   interactions with timestamp > T_valid (10%)

    Cold items = items that appear ONLY after T_train (never seen in training)
    """

    def __init__(self, train_pairs, n_users, n_warm, n_negs=16):
        self.train_pairs = train_pairs  # list of (user, warm_item_idx)
        self.n_users     = n_users
        self.n_warm      = n_warm
        self.n_negs      = n_negs

        # Build user positive sets for negative sampling
        self.user_pos = defaultdict(set)
        for u, i in train_pairs:
            self.user_pos[u].add(i)

    def __len__(self):
        return len(self.train_pairs)

    def __getitem__(self, idx):
        user, pos_item = self.train_pairs[idx]

        # Sample n_negs negative items
        negs = []
        pos_set = self.user_pos[user]
        while len(negs) < self.n_negs:
            neg = np.random.randint(0, self.n_warm)
            if neg not in pos_set:
                negs.append(neg)

        return (
            torch.tensor(user,     dtype=torch.long),
            torch.tensor(pos_item, dtype=torch.long),
            torch.tensor(negs,     dtype=torch.long),
        )


def recall_ndcg_at_k(scores, pos_set, k=20):
    """Compute Recall@K and NDCG@K."""
    topk = scores.argsort(descending=True)[:k]
    hits = sum(1 for i in topk if i.item() in pos_set)
    recall = hits / min(len(pos_set), k)
    ndcg = sum(
        1 / np.log2(rank + 2)
        for rank, i in enumerate(topk.tolist())
        if i in pos_set
    )
    return recall, ndcg


@torch.no_grad()
def evaluate(model, test_pairs, cold_test_pairs, feat_warm, feat_cold,
             train_pairs, n_users, n_warm, n_cold, k=20, device='cpu'):
    """
    Evaluate Recall@K and NDCG@K for warm and cold items.

    Warm scoring:  dot(user_CF, item_CF)
    Cold scoring:  dot(user_CF, proj(cold_content))
    """
    model.eval()

    # Build train positive sets (exclude from eval)
    train_pos = defaultdict(set)
    for u, i in train_pairs:
        train_pos[u].add(i)

    # Precompute all embeddings
    user_emb, item_emb = model.lightgcn_propagate()

    # Cold item projected embeddings (inference only)
    cold_proj = torch.nn.functional.normalize(
        model.content_proj(feat_cold.to(device)), dim=-1
    )  # [n_cold, D]

    # Warm evaluation
    warm_by_user = defaultdict(list)
    for u, i in test_pairs:
        warm_by_user[u].append(i)

    r_warm, n_warm_2, nu_warm = 0, 0, 0
    for u, pos_items in warm_by_user.items():
        u_e = user_emb[u]
        scores = (item_emb * u_e).sum(-1)  # [n_warm]
        # Exclude training items
        for ti in train_pos[u]:
            scores[ti] = -1e9
        r, n = recall_ndcg_at_k(scores, set(pos_items), k)
        r_warm += r; n_warm_2 += n; nu_warm += 1

    # Cold evaluation
    cold_by_user = defaultdict(list)
    for u, i in cold_test_pairs:
        cold_by_user[u].append(i - n_warm)  # reindex to [0, n_cold)

    r_cold, n_cold_2, nu_cold = 0, 0, 0
    for u, pos_items in cold_by_user.items():
        u_e = torch.nn.functional.normalize(user_emb[u].unsqueeze(0), dim=-1)
        scores = (u_e * cold_proj).sum(-1)  # [n_cold]
        r, n = recall_ndcg_at_k(scores, set(pos_items), k)
        r_cold += r; n_cold_2 += n; nu_cold += 1

    return {
        'warm_recall': r_warm / nu_warm if nu_warm > 0 else 0,
        'warm_ndcg':   n_warm_2 / nu_warm if nu_warm > 0 else 0,
        'cold_recall': r_cold / nu_cold if nu_cold > 0 else 0,
        'cold_ndcg':   n_cold_2 / nu_cold if nu_cold > 0 else 0,
    }


class EarlyStopping:
    def __init__(self, patience=10, metric='cold_recall'):
        self.patience  = patience
        self.metric    = metric
        self.best      = 0
        self.counter   = 0
        self.best_state = None

    def step(self, metrics, model):
        val = metrics[self.metric]
        if val > self.best:
            self.best = val
            self.counter = 0
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
            return False  # continue
        self.counter += 1
        return self.counter >= self.patience  # stop

    def load_best(self, model):
        if self.best_state:
            model.load_state_dict(self.best_state)
