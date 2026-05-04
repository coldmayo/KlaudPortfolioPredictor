import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight

class RForest_Sklearn:

    def __init__(
        self,
        num_trees: int = 100,
        max_depth: int = 10,
        min_samples: int = 2,
        max_features: int | str | None = "sqrt",
        n_jobs: int = -1,
        class_weight: str | dict | None = "balanced",
        random_state: int = 42,
        comm=None,
    ):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.class_weight = class_weight
        self.random_state = random_state

        self._model = RandomForestClassifier(
            n_estimators=num_trees,
            max_depth=max_depth,
            min_samples_leaf=min_samples,
            max_features=max_features,
            n_jobs=n_jobs,
            class_weight=class_weight,
            random_state=random_state,
        )

        self.classes_ = None

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight=None) -> None:

        print(
            f"[sklearn RF] fitting {self.num_trees} trees "
            f"(max_depth={self.max_depth}, max_features={self.max_features}, "
            f"class_weight={self.class_weight}) ..."
        )
        self._model.fit(X, y, sample_weight=sample_weight)
        self.classes_ = self._model.classes_
        print(f"[sklearn RF] done — {len(self._model.estimators_)} estimators built.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    # same as predict_proba
    def predict_probs(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)

    def get_feature_importances(self, X: np.ndarray) -> np.ndarray:
        return self._model.feature_importances_

    def _gather_trees(self):
        return self._model.estimators_