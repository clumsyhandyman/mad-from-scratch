import numpy as np


class ModelBasedLearner:
    def __init__(self, num_states, num_actions, alpha=0.1):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.learned_transition_model = np.ones((self.num_states, self.num_actions, self.num_states)) * 0.0001
        self.learned_reward_function = np.zeros(self.num_states)

    def get_action(self):
        return np.random.choice(self.num_actions)

    def update(self, s, a, s_prime, r):
        self.learned_transition_model[s, a, s_prime] += 1
        old_r = self.learned_reward_function[s_prime]
        self.learned_reward_function[s_prime] += self.alpha * (r - old_r)

    def generate_transition_model_for_solver(self):
        return self.learned_transition_model / np.sum(self.learned_transition_model, axis=2, keepdims=1)









