import numpy as np
from tqdm import tqdm

class SVM:
    def __init__(self, C=1.0, tol=1e-3, max_passes=5, kernel="linear", degree=3, sigma=1.0):
        self.C = C
        self.tol = tol
        self.max_passes = max_passes
        self.kernel_type = kernel
        self.degree = degree
        self.sigma = sigma

        self.alpha = None
        self.b = 0
        self.X = None
        self.y = None
        self.K = None

    def _kernel(self, x1, x2):
        if self.kernel_type == "linear":
            return np.dot(x1, x2)
        elif self.kernel_type == "poly":
            return (np.dot(x1, x2) + 1) ** self.degree
        elif self.kernel_type == "rbf":
            return np.exp(-np.linalg.norm(x1 - x2)**2 / (2 * self.sigma**2))
        else:
            raise ValueError("Unknown kernel")

    def _compute_kernel_matrix(self):
        n = self.X.shape[0]
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = self._kernel(self.X[i], self.X[j])
        return K

    def _decision_function(self, i):
        return np.sum(self.alpha * self.y * self.K[:, i]) + self.b

    def predict(self, X):
        preds = []
        for x in X:
            s = 0
            for i in range(len(self.alpha)):
                if self.alpha[i] > 0:
                    s += self.alpha[i] * self.y[i] * self._kernel(self.X[i], x)
            preds.append(np.sign(s + self.b))
        return np.array(preds)

    # SMO training

    def fit(self, X, y):
        self.X = X
        self.y = np.where(y <= 0, -1, 1)

        n_samples = X.shape[0]
        self.alpha = np.zeros(n_samples)
        self.b = 0

        self.K = self._compute_kernel_matrix()

        passes = 0

        while passes < self.max_passes:
            num_changed_alphas = 0

            for i in tqdm(range(n_samples)):
                Ei = self._decision_function(i) - self.y[i]

                if ((self.y[i] * Ei < -self.tol and self.alpha[i] < self.C) or
                    (self.y[i] * Ei > self.tol and self.alpha[i] > 0)):

                    j = np.random.randint(0, n_samples)
                    while j == i:
                        j = np.random.randint(0, n_samples)

                    Ej = self._decision_function(j) - self.y[j]

                    alpha_i_old = self.alpha[i]
                    alpha_j_old = self.alpha[j]

                    # Compute bounds L, H
                    if self.y[i] != self.y[j]:
                        L = max(0, self.alpha[j] - self.alpha[i])
                        H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                    else:
                        L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                        H = min(self.C, self.alpha[i] + self.alpha[j])

                    if L == H:
                        continue

                    eta = 2 * self.K[i, j] - self.K[i, i] - self.K[j, j]
                    if eta >= 0:
                        continue

                    self.alpha[j] -= self.y[j] * (Ei - Ej) / eta
                    self.alpha[j] = np.clip(self.alpha[j], L, H)

                    if abs(self.alpha[j] - alpha_j_old) < 1e-5:
                        continue

                    # Update alpha_i
                    self.alpha[i] += self.y[i] * self.y[j] * (alpha_j_old - self.alpha[j])

                    # Compute b1, b2
                    b1 = (self.b
                          - Ei
                          - self.y[i] * (self.alpha[i] - alpha_i_old) * self.K[i, i]
                          - self.y[j] * (self.alpha[j] - alpha_j_old) * self.K[i, j])

                    b2 = (self.b
                          - Ej
                          - self.y[i] * (self.alpha[i] - alpha_i_old) * self.K[i, j]
                          - self.y[j] * (self.alpha[j] - alpha_j_old) * self.K[j, j])

                    # Update bias
                    if 0 < self.alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2

                    num_changed_alphas += 1

            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0

            print(f"Pass: {passes}, changed alphas: {num_changed_alphas}")
