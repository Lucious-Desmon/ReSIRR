import numpy as np


class StateReflect(object):
    def __init__(self, size, n_actions, move_step):
        self.image = np.zeros(size, dtype=np.float32)
        self.n_actions = n_actions
        self.move_step = move_step
        if n_actions % 2 == 0:
            raise RuntimeError("n_actions must be odd for a neutral action")

    def reset(self, x):
        self.image = x

    def step(self, act):
        neutral = (self.n_actions - 1) / 2
        move = act.astype(np.float32)
        delta = (move - neutral) * self.move_step
        self.image = np.clip(self.image + delta, 0.0, 1.0)
