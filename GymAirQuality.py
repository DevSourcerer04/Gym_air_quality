import numpy as np
import gymnasium as gym
from gymnasium import spaces

class SensorTransmissionEnv(gym.Env):
    def __init__(self):
        self.lam = 0.1
        self.B = 10
        self.eta = 2
        self.Delta = 3
        self.air = np.load("air.npy")
        self.solar = np.load("solar.npy")
        self.observation_space = spaces.MultiDiscrete([51, 11, 51, 51])
        self.action_space = spaces.Discrete(3)
        self.t = 0
        self.state = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        th = np.random.randint(51)
        b = np.random.randint(11)
        h = np.random.randint(51)
        m = th
        self.t = 0
        self.state = np.array([th, b, h, m], dtype=np.int32)
        return self.state, {}

    def step(self, action):
        th, b, h, m = self.state
        if b < self.eta:
            action = 0
        success = action > 0 and np.random.rand() < self.lam
        nh = h
        if success:
            nh = th if action == 1 else m
        x = th / 50
        y = nh / 50
        loss = abs(x - y) if x <= y else 1.5 * abs(x - y)
        r = -loss
        d = np.random.choice(4, p=self.solar)
        nb = min(self.B, b + d - (self.eta if action > 0 else 0))
        nth = np.random.choice(51, p=self.air[th])
        nm = nth if success else max(m, nth)
        self.t += 1
        self.state = np.array([nth, nb, nh, nm], dtype=np.int32)
        return self.state, r, False, self.t >= 288, {}

    def render(self):
        pass