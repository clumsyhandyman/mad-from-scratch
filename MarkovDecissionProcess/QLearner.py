import numpy as np


class QLearner:
    def __init__(self, num_states, num_actions, alpha=0.2, gamma=0.9, epsilon=0.9, xi=0.99):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.xi = xi
        self.cur_policy = np.random.randint(num_actions, size=num_states)
        self.q_table = np.zeros((num_states, num_actions))

    def percept(self, s, a, s_prime, r):
        q_prime = np.max(self.q_table[s_prime])
        old_q_value = self.q_table[s, a]
        learned_value = r + self.gamma * q_prime - old_q_value
        self.q_table[s, a] += self.alpha * learned_value
        self.cur_policy[s] = np.argmax(self.q_table[s])

    def actuate(self, s):
        if np.random.uniform() <= self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            return self.cur_policy[s]

    def update_episode(self):
        self.epsilon *= self.xi












