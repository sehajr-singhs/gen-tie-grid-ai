"""Shared input normalization for the screening GNN and baselines.

Node features: [p_load, q_load, p_gen, q_gen, is_slack, is_plant, vm_pu,
regime_norm, plant_p_norm] (last two appended per scenario). Edge features:
[r, x, c, rate, tap, is_trafo]. Stats are fit on the training split with an
identical formula in every consumer, so GNN, baselines, and the deployed
screening signal all see the same scaling.
"""
import numpy as np


def append_feats(X, feats, n_nodes=40):
    """X: (n, n_nodes, 7); feats: (n, 2) -> (n, n_nodes, 9)."""
    return np.concatenate([X, np.repeat(feats[:, None, :], n_nodes, axis=1)],
                          axis=2)


def fit_stats(node_feat, feats, edge_feat):
    """node_feat: raw (n,40,7) from the training split; feats (n,2)."""
    X = np.nan_to_num(node_feat.astype(np.float32), nan=0.0)
    X = append_feats(X, feats.astype(np.float32))
    nm = X.reshape(-1, X.shape[-1]).mean(0)
    ns = X.reshape(-1, X.shape[-1]).std(0) + 1e-6
    ef = edge_feat.astype(np.float32)
    em = ef.reshape(-1, ef.shape[-1]).mean(0)
    es = ef.reshape(-1, ef.shape[-1]).std(0) + 1e-6
    return nm, ns, em, es


def normalize_node(X, nm, ns):
    X = np.nan_to_num(X.astype(np.float32), nan=0.0)
    return (X - nm) / ns


def normalize_edge(ef, em, es):
    return (ef.astype(np.float32) - em) / es
