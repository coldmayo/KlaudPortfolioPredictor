import numpy as np
from tqdm import tqdm
from mpi4py import MPI
from collections import Counter

# Using MPI4PY for MPI implimentation in Python

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, gain=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.gain = gain
        self.value = value


class DTree:
    def __init__(self, min_samples=2, max_depth=10, max_feats=None):
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.max_features = max_feats
        self.root = None

    def entropy(self, y):
        if len(y) == 0:
            return 0

        entropy = 0
        labels = np.unique(y)

        for label in labels:
            p = np.sum(y == label) / len(y)
            entropy -= p * np.log2(p)

        return entropy

    def get_feature_importance(self, num_features):
        importances = np.zeros(num_features)
    
        def _collect_importance(node):
            if node is None or node.feature is None:
                return
        
            importances[node.feature] += node.gain
        
            _collect_importance(node.left)
            _collect_importance(node.right)

        _collect_importance(self.root)
        return importances

    def info_gain(self, parent, left, right):

        w_l = len(left) / len(parent)
        w_r = len(right) / len(parent)

        return (self.entropy(parent) - w_l * self.entropy(left) - w_r * self.entropy(right))


    def best_split(self, dataset, num_features, n_thresholds=20):
        k = self.max_features if self.max_features else num_features
        k = min(k, num_features)
        feature_indices = np.random.choice(num_features, k, replace=False)

        best = {}
        max_gain = -float("inf")

        for feature_idx in feature_indices:
            unique_vals= np.unique(dataset[:, feature_idx])
            if len(unique_vals) > n_thresholds:
                percentiles = np.linspace(5, 95, n_thresholds)
                thresholds = np.unique(np.percentile(dataset[:, feature_idx], percentiles))
            else:
                thresholds = unique_vals

            for threshold in thresholds:
                left  = dataset[dataset[:, feature_idx] <= threshold]
                right = dataset[dataset[:, feature_idx] >  threshold]

                if len(left) == 0 or len(right) == 0:
                    continue

                gain = self.info_gain(dataset[:, -1], left[:, -1], right[:, -1])
                if gain > max_gain:
                    best = dict(
                        feature=feature_idx,
                        threshold=threshold,
                        left=left,
                        right=right,
                        gain=gain,
                    )
                    max_gain = gain

        return best


    def build_tree(self, dataset, depth=0):
        X, y = dataset[:, :-1], dataset[:, -1]
        n_samples, n_features = X.shape
 
        if n_samples >= self.min_samples and depth <= self.max_depth:
            split = self.best_split(dataset, n_features)
            if split and split["gain"] > 0:
                return Node(
                    feature=split["feature"],
                    threshold=split["threshold"],
                    left=self.build_tree(split["left"],  depth + 1),
                    right=self.build_tree(split["right"], depth + 1),
                    gain=split["gain"],
                )
 
        return Node(value=self._majority_vote(y))

    def _majority_vote(self, y):
        return Counter(y).most_common(1)[0][0]

    def fit(self, X, y):
        dataset = np.concatenate((X, y.reshape(-1, 1)), axis=1)
        self.root = self.build_tree(dataset)

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

class RForest_MPI:
    def __init__(self, num_trees=10, max_depth=10, min_samples=2, max_features=None, comm=None):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.max_features = max_features
        self.comm = comm or MPI.COMM_WORLD
        self.trees = []
        self.cached_trees = None

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def get_feature_importances(self, X):
        all_trees = self._gather_trees()
        if self.comm.Get_rank() == 0:
            n_features = X.shape[1]
            total = np.zeros(n_features)
            for t in all_trees:
                total += t.get_feature_importance(n_features)
            s = total.sum()
            return total / s if s > 0 else total
        return None


    def _local_tree_count(self):
        size = self.comm.Get_size()
        rank = self.comm.Get_rank()
        base, remainder = divmod(self.num_trees, size)
        return base + (1 if rank < remainder else 0)


    def fit(self, X, y):

        comm = self.comm
        rank = comm.Get_rank()

        self.cached_trees = None
        self.trees = []
        
        n_features = X.shape[1]
        max_features = self.max_features or max(1, int(np.sqrt(n_features)))
        
        np.random.seed(42 + rank)

        n_local = self._local_tree_count()

        if rank == 0:
            size = comm.Get_size()
            print(f"[MPI] {size} ranks, {self.num_trees} total trees "
                  f"(~{self.num_trees // size} per rank), "
                  f"max_features={max_features}/{n_features}")
            
        for i in range(n_local):
            tree = DTree(min_samples=self.min_samples,
                         max_depth=self.max_depth,
                         max_feats=max_features)
            X_s, y_s = self._bootstrap_sample(X, y)
            tree.fit(X_s, y_s)
            self.trees.append(tree)
            if rank == 0:
                print(f"  rank 0: trained tree {i + 1}/{n_local}", flush=True)
        
        comm.Barrier()

    def _gather_trees(self):

        if self.cached_trees is not None:
            return self.cached_trees
            
        all_local = self.comm.gather(self.trees, root=0)
        if self.comm.Get_rank() == 0:
            return [tree for sublist in all_local for tree in sublist]
        return None

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.comm = MPI.COMM_WORLD

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["comm"]
        return state

    def predict(self, X):
        all_trees = self._gather_trees()
        if self.comm.Get_rank() == 0:
            preds = np.array([tree.predict(X) for tree in all_trees])
            return np.array([max(set(row), key=list(row).count) for row in preds.T])
        return None

    def predict_probs(self, X):
        all_trees = self._gather_trees()
        if self.comm.Get_rank() == 0:
            tree_preds = np.array([tree.predict(X) for tree in all_trees])

            self.classes_ = np.unique(tree_preds)
            n_samples = X.shape[0]
            n_classes = len(self.classes_)

            proba = np.zeros((n_samples, n_classes))

            for i, cls in enumerate(self.classes_):
                proba[:, i] = np.sum(tree_preds == cls, axis=0) / len(all_trees)

            return proba
        return None
