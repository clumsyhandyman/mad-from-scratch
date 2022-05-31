import numpy as np


class MCLearner:
    def __init__(self, num_states, num_actions, epsilon=0.9, xi=0.99, initial_q=0.0):
        self.num_states = num_states
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.xi = xi

        self.q_table = np.ones((num_states, num_actions)) * initial_q
        self.g_table = np.zeros((num_states, num_actions))
        self.count_table = np.zeros((num_states, num_actions))
        self.cur_policy = np.random.randint(num_actions, size=num_states)
        self.visited = np.zeros((num_states, num_actions))

    def percept(self, s, a, s_prime, r):
        if self.visited[s, a] == 0:
            self.visited[s, a] = 1
        self.g_table[self.visited == 1] += r

    def actuate(self, s_prime):
        if np.random.uniform() <= self.epsilon:
            # temp = self.count_table[s_prime]
            # return np.random.choice(np.flatnonzero(temp == temp.min()))
            return np.random.randint(self.num_actions)
        else:
            return self.cur_policy[s_prime]

    def policy_update(self):
        # print(f'visited = {self.visited}')
        # print(f'r_table = {self.r_table}')
        # print(f'q_table = {self.q_table}')
        # print(f'count_table = {self.count_table}')
        # print(f'policy = {self.cur_policy}')
        self.count_table[self.visited == 1] += 1
        self.q_table[self.visited == 1] += (self.g_table[self.visited == 1] - self.q_table[self.visited == 1]
                                            ) / self.count_table[self.visited == 1]

        for s in range(self.num_states):
            self.cur_policy[s] = np.argmax(self.q_table[s])

        self.epsilon *= self.xi
        self.g_table = np.zeros((self.num_states, self.num_actions))
        self.visited = np.zeros((self.num_states, self.num_actions))

        # print(f'visited = {self.visited}')
        # print(f'r_table = {self.r_table}')
        # print(f'q_table = {self.q_table}')
        # print(f'count_table = {self.count_table}')
        # print(f'policy = {self.cur_policy}')












