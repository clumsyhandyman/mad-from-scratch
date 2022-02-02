import numpy as np


class MCLearner:
    def __init__(self, num_states, num_actions, gamma=0.9, rar=0.9, radr=0.99):
        self.num_states = num_states
        self.num_actions = num_actions
        self.gamma = gamma
        self.rar = rar
        self.radr = radr

        self.q_table = np.zeros((num_states, num_actions))
        self.r_table = np.zeros((num_states, num_actions))
        self.count_table = np.zeros((num_states, num_actions))
        self.cur_policy = np.random.randint(num_actions, size=num_states)
        self.visited_flag = np.zeros((num_states, num_actions))

    def update_step(self, s, a, s_prime, r):
        if self.visited_flag[s, a] == 0:
            self.visited_flag[s, a] = 1
        self.r_table[self.visited_flag == 1] += r
        if np.random.uniform() < self.rar:
            # temp = self.count_table[s_prime]
            # return np.random.choice(np.flatnonzero(temp == temp.min()))
            return np.random.randint(self.num_actions)
        else:
            return self.cur_policy[s_prime]

    def update_episode(self):
        # print(f'visited = {self.visited_flag}')
        # print(f'r_table = {self.r_table}')
        # print(f'q_table = {self.q_table}')
        # print(f'count_table = {self.count_table}')
        # print(f'policy = {self.cur_policy}')

        self.q_table[self.visited_flag == 1] = (self.q_table[self.visited_flag == 1] * self.count_table[
            self.visited_flag == 1] + self.r_table[self.visited_flag == 1]) / (
                                                           self.count_table[self.visited_flag == 1] + 1)
        self.count_table[self.visited_flag == 1] += 1

        for s in range(self.num_states):
            if np.sum(self.visited_flag[s]) > 0:
                self.cur_policy[s] = np.argmax(self.q_table[s])

        self.rar *= self.radr
        self.r_table = np.zeros((self.num_states, self.num_actions))
        self.visited_flag = np.zeros((self.num_states, self.num_actions))

        # print(f'visited = {self.visited_flag}')
        # print(f'r_table = {self.r_table}')
        # print(f'q_table = {self.q_table}')
        # print(f'count_table = {self.count_table}')
        # print(f'policy = {self.cur_policy}')












