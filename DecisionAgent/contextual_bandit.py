import numpy as np

class ContextualBanditAgent:
    def __init__(self, lr: float = 1e-3, epsilon: float = 0.1, seed: int = 0):
        self.lr = lr
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

        # We use 2 transformed features + bias
        self.w = np.zeros(3, dtype=np.float32)  # [w_progress, w_distance, bias]

    def _phi_from_row(self, disaster_row: np.ndarray) -> np.ndarray:
        """
        disaster_row = [type, completion_progress, distance]
        - ignore type
        completion_progress: smaller => more completed (better)
        distance: smaller => better
        """
        p = float(disaster_row[1])
        d = float(disaster_row[2])

        # transform: larger is better
        f_progress = -p
        f_distance = -np.log1p(d)  # log(1+d)

        return np.array([f_progress, f_distance, 1.0], dtype=np.float32)

    def predict(self, disaster_row: np.ndarray) -> float:
        phi = self._phi_from_row(disaster_row)
        return float(np.dot(self.w, phi))

    def select_action(self, obs: dict) -> int:
        mat = obs["visible_disasters"]  # [N, 3]
        mask = obs["valid_actions"].astype(np.int32)
        valid = np.where(mask == 1)[0]

        if valid.size == 0:
            return 0  # ideally env should provide WAIT/NOOP

        # explore
        if self.rng.random() < self.epsilon:
            return int(self.rng.choice(valid))

        # exploit: pick max predicted reward
        scores = [self.predict(mat[a]) for a in valid]
        return int(valid[int(np.argmax(scores))])

    def update(self, obs: dict, action: int, reward: float):
        mat = obs["visible_disasters"]
        phi = self._phi_from_row(mat[action])

        r = float(reward)
        pred = float(np.dot(self.w, phi))
        err = pred - r
        self.w -= self.lr * err * phi
