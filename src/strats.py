import numpy as np

class Strategy:
    def __init__(self, alpha):
        self.alpha = alpha

    def get_positions(self):
        raise NotImplementedError

class SignStrategy(Strategy):
    def get_positions(self):
        return np.sign(self.alpha)

class ThresholdStrategy(Strategy):
    def __init__(self, alpha, threshold=0.5):
        super().__init__(alpha)
        self.threshold = threshold

    def get_positions(self):
        pos = np.zeros_like(self.alpha)
        pos[self.alpha > self.threshold] = 1
        pos[self.alpha < -self.threshold] = -1
        return pos

class TopKStrategy(Strategy):
    def __init__(self, alpha, k=0.1, mode="timeseries", n_assets=None, dates=None):
        super().__init__(alpha)
        self.k = k
        self.mode = mode
        self.n_assets = n_assets
        self.dates = dates

    def get_positions(self):
        if self.mode == "timeseries":
            return self._timeseries_positions()
        elif self.mode == "crosssectional":
            return self._crosssectional_positions()
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Use 'timeseries' or 'crosssectional'")

    def _timeseries_positions(self):

        n = len(self.alpha)
        k = max(1, int(n * self.k))

        ranks = np.argsort(self.alpha)
        pos = np.zeros(n)
        pos[ranks[-k:]] = 1
        pos[ranks[:k]]  = -1
        return pos

    def _crosssectional_positions(self):
        if self.alpha.ndim != 2:
            raise ValueError(
                "Cross-sectional TopK requires alpha shape (T, n_assets). "
                f"Got shape {self.alpha.shape}"
            )

        T, n_assets = self.alpha.shape
        k = max(1, int(n_assets * self.k))
        pos = np.zeros_like(self.alpha)

        for t in range(T):
            row = self.alpha[t]
            if np.all(np.isnan(row)):
                continue
            ranks = np.argsort(row)
            pos[t, ranks[-k:]] = 1
            pos[t, ranks[:k]]  = -1

        return pos

class VolScaledStrategy(Strategy):
    def __init__(self, alpha, vol):
        super().__init__(alpha)
        self.vol = vol

    def get_positions(self):
        scaled = self.alpha / (self.vol + 1e-8)
        return np.clip(scaled, -1, 1)
