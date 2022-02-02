from GridWorld import GridWorld
from QLearner import QLearner

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


class GridWorldMBSolver:
    def __init__(self, problem, learner_class, rar=0.9, radr=0.99):
        self.problem = problem
        self.learner = learner_class(num_states=problem.num_states, num_actions=problem.num_actions, rar=rar, radr=radr)

    def train_one_epoch(self, start_pos):
        s = self.problem.get_state_from_pos(start_pos)
        a = self.learner.cur_policy[s]
        reward_game = 0
        while True:
            s_prime, r = self.problem.blackbox_move(s, a)
            a = self.learner.update_step(s, a, s_prime, r)
            reward_game += r
            if r == self.problem.reward[1] or r == self.problem.reward[2]:
                break
            else:
                s = s_prime
        self.learner.update_episode()
        return reward_game

    def train(self, epochs, start_pos, plot=True):
        reward_history = np.zeros(epochs)
        total_reward_history = np.zeros(epochs)
        total_reward = 0

        for i in range(epochs):
            print(f'Training epoch {i + 1}')
            reward_episode = self.train_one_epoch(start_pos=start_pos)
            total_reward += reward_episode
            reward_history[i] = reward_episode
            total_reward_history[i] = total_reward

        print('FINAL POLICY')
        print(self.learner.cur_policy)
        print('FINAL Q')
        print(self.learner.q_table)
        print('FINAL VALUE')
        print(np.max(self.learner.q_table, axis=1))

        if plot:
            fig, axes = plt.subplots(2, 1, figsize=(6, 6), dpi=200, sharex='all')
            axes[0].plot(np.arange(len(total_reward_history)), total_reward_history,
                         alpha=0.7, color='#d62728')
            axes[0].set_ylabel('Total rewards')
            axes[1].plot(np.arange(len(reward_history)), reward_history, marker='o', markersize=2,
                         alpha=0.7, color='#2ca02c', linestyle='none')
            axes[1].set_xlabel('Episode')
            axes[1].set_ylabel('Reward from\na single game')
            # axes[1].set_ylim(-1000, 100)
            axes[1].xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
            axes[0].grid(axis='x')
            axes[1].grid(axis='x')
            plt.tight_layout()
            plt.show()


# problem = GridWorld('data/world00.csv', reward={0: -0.04, 1: 1.0, 2: -1.0, 3: np.NaN}, random_rate=0.2)
# problem_solver = GridWorldMBSolver(problem, QLearner, rar=0.5, radr=0.99)
# problem_solver.train(200, start_pos=(2, 0))
#
problem = GridWorld('data/world02.csv', reward={0: -0.04, 1: 10.0, 2: -2.5, 3: np.NaN}, random_rate=0.2)
problem_solver = GridWorldMBSolver(problem, QLearner, rar=0.99, radr=0.99)
problem_solver.train(1000, start_pos=(5, 3), plot=True)


