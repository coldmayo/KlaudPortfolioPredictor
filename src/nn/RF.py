import numpy as np
from tqdm import tqdm

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, gain=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.gain = gain
        self.value = value


class DTree:
    def __init__(self, min_samples=2, max_depth=10):
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.root = None

    def entropy(self, y):
        if len(y) == 0:
            return 0

        entropy = 0
        labels = np.unique(y)

        for label in labels:
            p = len(y[y == label]) / len(y)
            entropy += -p * np.log2(p)

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
        parent_e = self.entropy(parent)

        w_l = len(left) / len(parent)
        w_r = len(right) / len(parent)

        e_l = self.entropy(left)
        e_r = self.entropy(right)

        weighted_e = (w_l * e_l) + (w_r * e_r)

        return parent_e - weighted_e

    def best_split(self, dataset, num_features):
        best_split = {}
        max_gain = -float("inf")

        for feature_idx in range(num_features):
            feature_values = dataset[:, feature_idx]
            thresholds = np.unique(feature_values)

            for threshold in thresholds:
                left = dataset[dataset[:, feature_idx] <= threshold]
                right = dataset[dataset[:, feature_idx] > threshold]

                if len(left) == 0 or len(right) == 0:
                    continue

                y = dataset[:, -1]
                y_left = left[:, -1]
                y_right = right[:, -1]

                gain = self.info_gain(y, y_left, y_right)

                if gain > max_gain:
                    best_split = {
                        "feature": feature_idx,
                        "threshold": threshold,
                        "left": left,
                        "right": right,
                        "gain": gain
                    }
                    max_gain = gain

        return best_split

    def build_tree(self, dataset, depth=0):
        X = dataset[:, :-1]
        y = dataset[:, -1]
        n_samples, n_features = X.shape

        if (
            n_samples >= self.min_samples
            and depth <= self.max_depth
        ):
            split = self.best_split(dataset, n_features)

            if split and split["gain"] > 0:
                left_subtree = self.build_tree(split["left"], depth + 1)
                right_subtree = self.build_tree(split["right"], depth + 1)

                return Node(
                    feature=split["feature"],
                    threshold=split["threshold"],
                    left=left_subtree,
                    right=right_subtree,
                    gain=split["gain"]
                )

        # Leaf node
        leaf_value = self._majority_vote(y)
        return Node(value=leaf_value)

    def _majority_vote(self, y):
        return max(set(y), key=list(y).count)

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

class RForest:
    def __init__(self, num_trees=10, max_depth=10, min_samples=2, max_features=None):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.max_features = max_features
        self.trees = []

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def get_feature_importances(self, X):
        n_features = X.shape[1]
        total_importances = np.zeros(n_features)

        for tree in self.trees:
            total_importances += tree.get_feature_importance(n_features)

        if np.sum(total_importances) > 0:
            total_importances /= np.sum(total_importances)

        return total_importances

    def fit(self, X, y):
        self.trees = []

        for _ in tqdm(range(self.num_trees)):
            tree = DTree(
                min_samples=self.min_samples,
                max_depth=self.max_depth
            )

            X_sample, y_sample = self._bootstrap_sample(X, y)

            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])

        # majority vote across trees
        return np.array([
            max(set(row), key=list(row).count)
            for row in tree_preds.T
        ])

    def predict_probs(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])

        self.classes_ = np.unique(tree_preds)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)

        proba = np.zeros((n_samples, n_classes))

        for i, cls in enumerate(self.classes_):
            proba[:, i] = np.sum(tree_preds == cls, axis=0) / self.num_trees

        return proba
