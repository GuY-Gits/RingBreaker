"""
RingBreaker - EIF Class Definitions and Point-in-Time Behavioral Features
Module: anomaly/eif_model.py
"""

import numpy as np
import pandas as pd
from typing import List, Optional

FEATURE_COLUMNS = [
    "tx_amount",
    "amount_to_prior_avg",
    "tx_velocity_1h",
    "tx_velocity_24h",
    "tx_velocity_7d",
    "device_reuse_degree",
    "ip_reuse_degree",
    "sender_fan_out",
    "receiver_fan_in",
    "pass_through_ratio",
    "account_activity_span_hours",
    "rapid_drain_speed_seconds"
]


def c_factor(n: int) -> float:
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2.0 * (np.log(n - 1) + 0.5772156649) - (2.0 * (n - 1) / n)


class EIFNode:
    def __init__(self, left=None, right=None, normal_vector=None, intercept_point=None, size=0):
        self.left = left
        self.right = right
        self.normal_vector = normal_vector
        self.intercept_point = intercept_point
        self.size = size
        self.is_leaf = left is None and right is None


class EIFTree:
    def __init__(self, max_depth: int, extension_level: int):
        self.max_depth = max_depth
        self.extension_level = extension_level
        self.root = None

    def fit(self, X: np.ndarray, current_depth: int = 0) -> EIFNode:
        n_samples, n_features = X.shape
        if current_depth >= self.max_depth or n_samples <= 1:
            return EIFNode(size=n_samples)

        dim = min(self.extension_level + 1, n_features)
        chosen_dims = np.random.choice(n_features, size=dim, replace=False)

        normal_vec = np.zeros(n_features)
        normal_vec[chosen_dims] = np.random.normal(0, 1, size=dim)
        norm_mag = np.linalg.norm(normal_vec)
        if norm_mag > 0:
            normal_vec /= norm_mag

        mins = np.min(X, axis=0)
        maxs = np.max(X, axis=0)
        if np.allclose(mins, maxs):
            return EIFNode(size=n_samples)

        intercept = mins + np.random.uniform(0, 1, size=n_features) * (maxs - mins)
        dots = np.dot(X - intercept, normal_vec)
        left_mask = dots <= 0
        right_mask = ~left_mask

        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return EIFNode(size=n_samples)

        left_node = self.fit(X[left_mask], current_depth + 1)
        right_node = self.fit(X[right_mask], current_depth + 1)

        return EIFNode(
            left=left_node,
            right=right_node,
            normal_vector=normal_vec,
            intercept_point=intercept,
            size=n_samples
        )

    def path_length(self, x: np.ndarray, node: EIFNode, current_depth: int = 0) -> float:
        if node.is_leaf:
            return current_depth + c_factor(node.size)
        dot = np.dot(x - node.intercept_point, node.normal_vector)
        if dot <= 0:
            return self.path_length(x, node.left, current_depth + 1)
        else:
            return self.path_length(x, node.right, current_depth + 1)


class ExtendedIsolationForest:
    def __init__(self, n_trees: int = 128, sample_size: int = 256, extension_level: Optional[int] = None):
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.extension_level = extension_level
        self.trees: List[EIFTree] = []
        self.scaler_mean = None
        self.scaler_std = None

    def fit(self, X: np.ndarray):
        # Robust standardization using median & IQR/std to handle outliers during training
        self.scaler_mean = np.mean(X, axis=0)
        self.scaler_std = np.std(X, axis=0)
        self.scaler_std[self.scaler_std == 0] = 1.0
        X_scaled = (X - self.scaler_mean) / self.scaler_std

        n_samples, n_features = X_scaled.shape
        ext_lvl = (n_features - 1) if self.extension_level is None else self.extension_level
        subsample = min(self.sample_size, n_samples)
        max_depth = int(np.ceil(np.log2(max(subsample, 2))))

        self.trees = []
        for _ in range(self.n_trees):
            tree = EIFTree(max_depth=max_depth, extension_level=ext_lvl)
            sample_idx = np.random.choice(n_samples, size=subsample, replace=False)
            tree.root = tree.fit(X_scaled[sample_idx])
            self.trees.append(tree)
        return self

    def raw_scores(self, X: np.ndarray) -> np.ndarray:
        """Unnormalised anomaly score 2^(-E[h(x)]/c(n)); higher = more anomalous.

        Vectorised: each tree routes all samples level by level, so scoring a
        batch costs one numpy pass per node instead of one Python call per sample.
        """
        X = np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0, posinf=1e6, neginf=-1e6)
        if self.scaler_mean is not None and self.scaler_std is not None:
            X = (X - self.scaler_mean) / self.scaler_std
        flat = getattr(self, "_flat", None)
        if flat is None or flat["n_trees"] != len(self.trees):
            flat = self._flat = _flatten_forest(self.trees)
        paths = _forest_path_lengths(flat, X)
        avg = paths / max(len(self.trees), 1)
        c_denom = c_factor(self.sample_size)
        if c_denom == 0:
            return np.zeros(X.shape[0])
        return np.power(2.0, -(avg / c_denom))

    def compute_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        if self.scaler_mean is not None and self.scaler_std is not None:
            X_scaled = (X - self.scaler_mean) / self.scaler_std
        else:
            X_scaled = X

        n_samples = X_scaled.shape[0]
        paths = np.zeros(n_samples)
        for tree in self.trees:
            for i in range(n_samples):
                paths[i] += tree.path_length(X_scaled[i], tree.root)
        avg_paths = paths / self.n_trees
        c_denom = c_factor(min(self.sample_size, n_samples))
        if c_denom == 0:
            return np.zeros(n_samples)

        raw_scores = np.power(2.0, - (avg_paths / c_denom))
        s_min = np.min(raw_scores)
        s_max = np.max(raw_scores)
        if s_max > s_min:
            return (raw_scores - s_min) / (s_max - s_min)
        return raw_scores


def _flatten_forest(trees: List[EIFTree]) -> dict:
    """Pack every tree into shared arrays so all trees can be routed at once."""
    normals, inters, left, right, leaf_c, is_leaf, roots = [], [], [], [], [], [], []
    n_features = None

    def add(node: EIFNode) -> int:
        nonlocal n_features
        idx = len(left)
        left.append(idx), right.append(idx)  # leaves point at themselves
        is_leaf.append(node.is_leaf)
        leaf_c.append(c_factor(node.size) if node.is_leaf else 0.0)
        if node.is_leaf:
            normals.append(None), inters.append(None)
        else:
            normals.append(node.normal_vector), inters.append(node.intercept_point)
            n_features = len(node.normal_vector)
            left[idx] = add(node.left)
            right[idx] = add(node.right)
        return idx

    for tree in trees:
        roots.append(add(tree.root))
    nf = n_features or 1
    zero = np.zeros(nf)
    return {
        "n_trees": len(trees),
        "roots": np.array(roots, dtype=np.int64),
        "left": np.array(left, dtype=np.int64),
        "right": np.array(right, dtype=np.int64),
        "is_leaf": np.array(is_leaf, dtype=bool),
        "leaf_c": np.array(leaf_c, dtype=np.float64),
        "normals": np.array([zero if n is None else n for n in normals]),
        "inters": np.array([zero if i is None else i for i in inters]),
    }


def _forest_path_lengths(flat: dict, X: np.ndarray) -> np.ndarray:
    """Sum over trees of the path length of each row (same values as per-tree routing)."""
    B = X.shape[0]
    cur = np.broadcast_to(flat["roots"], (B, flat["n_trees"])).copy()
    depth = np.zeros(cur.shape)
    while True:
        active = ~flat["is_leaf"][cur]
        if not active.any():
            break
        d = ((X[:, None, :] - flat["inters"][cur]) * flat["normals"][cur]).sum(axis=2)
        nxt = np.where(d <= 0, flat["left"][cur], flat["right"][cur])
        cur = np.where(active, nxt, cur)
        depth += active
    return (depth + flat["leaf_c"][cur]).sum(axis=1)


def _batch_path_lengths(root: EIFNode, X: np.ndarray) -> np.ndarray:
    result = np.zeros(X.shape[0])
    _route(root, X, np.arange(X.shape[0]), 0, result)
    return result


def _route(node: EIFNode, X: np.ndarray, idx: np.ndarray, depth: int, result: np.ndarray) -> None:
    if len(idx) == 0:
        return
    if node.is_leaf:
        result[idx] = depth + c_factor(node.size)
        return
    left = (X[idx] - node.intercept_point) @ node.normal_vector <= 0
    _route(node.left, X, idx[left], depth + 1, result)
    _route(node.right, X, idx[~left], depth + 1, result)


class CalibratedEIF:
    """EIF plus the training-period score distribution.

    ``score`` maps a raw anomaly score to its empirical percentile among
    training payments, so a live payment gets a stable 0-1 value that does
    not depend on which other payments happen to be scored with it.
    """

    def __init__(self, model: ExtendedIsolationForest, reference_raw: np.ndarray):
        self.model = model
        self.reference = np.sort(np.asarray(reference_raw, dtype=np.float64))

    def score(self, X: np.ndarray) -> np.ndarray:
        raw = self.model.raw_scores(X)
        if len(self.reference) == 0:
            return np.clip(raw, 0.0, 1.0)
        return np.searchsorted(self.reference, raw, side="right") / len(self.reference)


def build_point_in_time_behavioral_features(df_payments: pd.DataFrame) -> pd.DataFrame:
    """
    Computes bidirectional point-in-time features strictly as-of transaction T.
    Captures mule layering, device sharing, burst drain speeds, and fan-out/in.
    """
    df = df_payments.sort_values("timestamp").copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    records = []
    
    # State maps tracking cumulative history strictly up to time T
    user_sent = {}       # user -> list of (t, amount, recv, dev, ip)
    user_recv = {}       # user -> list of (t, amount, sender, dev, ip)
    device_users = {}    # dev -> set of users
    ip_users = {}        # ip -> set of users

    for _, row in df.iterrows():
        t = row["timestamp"]
        sender = row["sender"]
        receiver = row["receiver"]
        amount = float(row["amount"])
        dev = str(row["device_id"])
        ip = str(row["ip_address"])

        # Prior history for sender & receiver strictly prior to current tx
        s_out = user_sent.get(sender, [])
        s_in = user_recv.get(sender, [])
        r_out = user_sent.get(receiver, [])
        r_in = user_recv.get(receiver, [])

        # Time windows
        t_1h = t - pd.Timedelta(hours=1)
        t_24h = t - pd.Timedelta(days=1)
        t_7d = t - pd.Timedelta(days=7)

        # 1. Velocities
        s_out_1h = sum(1 for h in s_out if h[0] >= t_1h)
        s_out_24h = sum(1 for h in s_out if h[0] >= t_24h)
        s_out_7d = sum(1 for h in s_out if h[0] >= t_7d)

        # 2. Prior amount baseline
        prior_amounts = [h[1] for h in s_out]
        prior_avg = float(np.mean(prior_amounts)) if prior_amounts else amount
        amount_to_prior_avg = amount / (prior_avg + 1e-2)

        # 3. Device & IP multiplexing
        dev_sharing_deg = len(device_users.get(dev, set()))
        ip_sharing_deg = len(ip_users.get(ip, set()))

        # 4. Fan-out / Fan-in
        unique_recvs = len(set(h[2] for h in s_out))
        sender_fan_out = unique_recvs / (len(s_out) + 1.0)
        
        unique_senders_to_r = len(set(h[2] for h in r_in))
        receiver_fan_in = unique_senders_to_r / (len(r_in) + 1.0)

        # 5. Mule pass-through ratio (total sent out / total received in + 1e-2)
        total_in_amt = sum(h[1] for h in s_in)
        total_out_amt = sum(h[1] for h in s_out) + amount
        pass_through_ratio = min(total_out_amt / (total_in_amt + 10.0), 10.0)

        # 6. Account activity span in hours (short span with high velocity = mule/bot)
        first_seen = s_out[0][0] if s_out else (s_in[0][0] if s_in else t)
        activity_span = max((t - first_seen).total_seconds() / 3600.0, 0.01)

        # 7. Rapid drain speed (seconds elapsed since last inbound money received)
        if s_in:
            last_in_time = s_in[-1][0]
            drain_seconds = max((t - last_in_time).total_seconds(), 0.0)
        else:
            drain_seconds = 86400.0  # default 24h baseline

        records.append({
            "transaction_id": row["transaction_id"],
            "sender": sender,
            "receiver": receiver,
            "timestamp": t,
            "tx_amount": amount,
            "amount_to_prior_avg": amount_to_prior_avg,
            "tx_velocity_1h": s_out_1h,
            "tx_velocity_24h": s_out_24h,
            "tx_velocity_7d": s_out_7d,
            "device_reuse_degree": dev_sharing_deg,
            "ip_reuse_degree": ip_sharing_deg,
            "sender_fan_out": sender_fan_out,
            "receiver_fan_in": receiver_fan_in,
            "pass_through_ratio": pass_through_ratio,
            "account_activity_span_hours": activity_span,
            "rapid_drain_speed_seconds": min(drain_seconds, 86400.0)
        })

        # Update historical state
        if sender not in user_sent:
            user_sent[sender] = []
        user_sent[sender].append((t, amount, receiver, dev, ip))

        if receiver not in user_recv:
            user_recv[receiver] = []
        user_recv[receiver].append((t, amount, sender, dev, ip))

        if dev not in device_users:
            device_users[dev] = set()
        device_users[dev].add(sender)

        if ip not in ip_users:
            ip_users[ip] = set()
        ip_users[ip].add(sender)

    return pd.DataFrame(records)