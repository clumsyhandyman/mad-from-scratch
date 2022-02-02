import numpy as np


class QLearner:
    def __init__(self, num_states, num_actions, alpha=0.2, gamma=0.9, rar=0.9, radr=0.99):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.cur_policy = np.random.randint(num_actions, size=num_states)
        self.q_table = np.zeros((num_states, num_actions))

    def update_step(self, s, a, s_prime, r):
        q_prime = np.max(self.q_table[s_prime])
        old_q_value = self.q_table[s, a]
        learned_value = r + self.gamma * q_prime - old_q_value
        self.q_table[s, a] += self.alpha * learned_value
        self.cur_policy[s] = np.argmax(self.q_table[s])
        if np.random.uniform() <= self.rar:
            return np.random.randint(self.num_actions)
        else:
            return self.cur_policy[s_prime]

    def update_episode(self):
        self.rar *= self.radr












