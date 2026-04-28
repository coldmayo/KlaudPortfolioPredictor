import numpy as np
from tqdm import tqdm

class Node():
    def __init__(self, feature=None, threshold=None, left=None, right=None, gain=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.gain = gain
        self.value = value

class DTree():
    def __init__(self, min_samples = 2, max_depth = 3, gamma = 0.00):
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.root = None
        self.gamma = gamma

    def mse(self, y):
        if len(y) == 0:
            return 0
        return np.mean((y - np.mean(y))**2)

    def gain(self, parent, left, right):
        parent_e = self.mse(parent)

        w_l = len(left) / len(parent)
        w_r = len(right) / len(parent)

        e_l = self.mse(left)
        e_r = self.mse(right)

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

                gain = self.gain(y, y_left, y_right)

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

            if split and split["gain"] > self.gamma:
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
        return np.mean(y)

    def fit(self, X, y):
        data = np.concatenate((X, y.reshape(-1, 1)), axis=1)
        self.root = self.build_tree(data)

    def _traverse(self, x, node):
        if node.value is not None:
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])

class XGBoost():
    def __init__(self, n_estimators=50, learning_rate=0.1, max_depth=3):
        self.trees = []
        self.base_pred = None
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth

    def sigmoid(self, x):
        x = np.clip(x, -15, 15)
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        y_mapped = np.where(y == -1, 0, 1)
        
        prob_mean = np.mean(y_mapped)
        self.base_pred = np.log(prob_mean / (1 - prob_mean + 1e-9))
        
        f_t = np.full_like(y_mapped, self.base_pred, dtype=float)

        #print(np.unique(y, return_counts=True))

        for i in tqdm(range(self.n_estimators)):
            p = self.sigmoid(f_t)
            residuals = y_mapped - p

            tree = DTree(max_depth=self.max_depth)
            tree.fit(X, residuals)
    
            tree_preds = tree.predict(X)
            #print(f"  tree pred range: [{tree_preds.min():.4f}, {tree_preds.max():.4f}], unique vals: {len(np.unique(tree_preds))}")
    
            f_t += self.learning_rate * tree_preds
            self.trees.append(tree)

    def predict(self, X):
        # Start with base log(odds)
        f_t = np.full(X.shape[0], self.base_pred)

        for tree in self.trees:
            f_t += self.learning_rate * tree.predict(X)
        
        # Convert log(odds) to probability
        probs = self.sigmoid(f_t)
        
        # Convert probabilities back to your original labels {-1, 1}
        # Using 0.5 as the threshold
        return np.where(probs >= 0.5, 1, -1)
