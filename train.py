import numpy as np
import gymnasium as gym
from GymAirQuality import SensorTransmissionEnv

def valid(s):
    return [0, 1, 2] if s[1] >= 2 else [0]

def idx(s):
    return tuple(s)

def act(q, s, eps):
    a = valid(s)
    if np.random.rand() < eps:
        return np.random.choice(a)
    v = q[idx(s)][a]
    return a[np.random.choice(np.flatnonzero(v == v.max()))]

def policy_from_q(q):
    p = np.zeros(q.shape[:-1], dtype=np.int8)
    for th in range(51):
        for b in range(11):
            for h in range(51):
                for m in range(51):
                    a = [0, 1, 2] if b >= 2 else [0]
                    p[th, b, h, m] = a[np.argmax(q[th, b, h, m, a])]
    return p

def test(env, p, n=20):
    z = 0
    for _ in range(n):
        s, _ = env.reset()
        done = False
        while not done:
            s, r, term, trunc, _ = env.step(p[idx(s)])
            z += r
            done = term or trunc
    return z / n

def QLearning(env, beta, Nepisodes, alpha):
    q = np.zeros((51, 11, 51, 51, 3))
    scores = []
    for e in range(Nepisodes):
        s, _ = env.reset()
        done = False
        eps = max(0.02, 1 - e / (0.7 * Nepisodes))
        while not done:
            a = act(q, s, eps)
            ns, r, term, trunc, _ = env.step(a)
            na = valid(ns)
            q[idx(s)][a] += alpha * (r + beta * np.max(q[idx(ns)][na]) - q[idx(s)][a])
            s = ns
            done = term or trunc
        if e % 100 == 0:
            scores.append(test(env, policy_from_q(q)))
        if (e + 1) % 1000 == 0 or e == 0:
            print(f"QLearning: completed episode {e + 1}/{Nepisodes}")
    np.save("qlearning_scores.npy", np.array(scores))
    return policy_from_q(q)

def QLearning_StructuralKnowledge(env, beta, Nepisodes, alpha):
    q = np.zeros((51, 11, 51, 51, 3))
    scores = []
    for e in range(Nepisodes):
        s, _ = env.reset()
        done = False
        eps = max(0.02, 1 - e / (0.7 * Nepisodes))
        while not done:
            a = act(q, s, eps)
            ns, r, term, trunc, _ = env.step(a)
            na = valid(ns)
            best = na[np.argmax(q[idx(ns)][na])]
            p = np.ones(len(na)) * eps / len(na)
            p[na.index(best)] += 1 - eps
            target = r + beta * np.sum(p * q[idx(ns)][na])
            q[idx(s)][a] += alpha * (target - q[idx(s)][a])
            s = ns
            done = term or trunc
        if e % 100 == 0:
            scores.append(test(env, policy_from_q(q)))
        if (e + 1) % 1000 == 0 or e == 0:
            print(f"StructuralKnowledge: completed episode {e + 1}/{Nepisodes}")
    np.save("structural_scores.npy", np.array(scores))
    return policy_from_q(q)

env = SensorTransmissionEnv()

Nepisodes = 10000
alpha = 0.1
beta = 0.98

print("Starting Q-learning training...")
policy1 = QLearning(env, beta, Nepisodes, alpha)
print("Starting structural-knowledge training...")
policy2 = QLearning_StructuralKnowledge(env, beta, Nepisodes, alpha)

np.save('policy1.npy', policy1)
np.save('policy2.npy', policy2)

print("Saved policy1.npy, policy2.npy, qlearning_scores.npy, and structural_scores.npy")

env.close()
