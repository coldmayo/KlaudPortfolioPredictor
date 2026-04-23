import numpy as np

class SVM():
    def __init__(self, lr, lambda_p, iters):
        self.lr = lr
        self.lambda_p = lambda_p
        self.iters = iters
        self.w = None
        self.b = None

    def predict(self, x):
        approx = np.dot(x, self.w) - self.b
        return np.sign(approx)

    def fit(self, x, y):
        n_samples, n_feats = x.shape
        y_ = np.where(y <= 0, -1, 1)

        self.w = np.zeros(n_feats)
        self.b = 0

        for i in range(self.iters):
            for j, x_i in enumerate(x):
                condition = y_[j] * (np.dot(x_i, self.w) - self.b) >= 1

                if condition:
                    self.w -= self.lr * (2 * self.lambda_p * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_p * self.w - y_[j] * x_i)
                    self.b -= self.lr * y_[j]
