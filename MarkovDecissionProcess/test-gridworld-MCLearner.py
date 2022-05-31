from GridWorld import GridWorld
from MCLearner import MCLearner

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


class GridWorldMCSolver:
    def __init__(self, problem, learner_class, epsilon=0.9, xi=0.99, initial_q=0.0):
        self.problem = problem
        self.learner = learner_class(num_states=problem.num_states, num_actions=problem.num_actions,
                                     epsilon=epsilon, xi=xi, initial_q=initial_q)

    def train_one_epoch(self, start_pos):
        s = self.problem.get_state_from_pos(start_pos)
        if_win = 0
        reward_game = 0
        while True:
            a = self.learner.actuate(s)
            s_prime, r = self.problem.blackbox_move(s, a)
            self.learner.percept(s, a, s_prime, r)
            reward_game += r
            if r == self.problem.reward[1]:
                if_win = 1
                break
            elif r == self.problem.reward[2]:
                break
            else:
                s = s_prime
        self.learner.policy_update()
        return reward_game, if_win

    def train(self, epochs, start_pos, plot=True):
        reward_history = np.zeros(epochs)
        total_reward_history = np.zeros(epochs)
        total_reward = 0
        game_win = np.zeros(epochs)

        for i in range(epochs):
            print(f'Training epoch {i + 1}')
            reward_episode, win_episode = self.train_one_epoch(start_pos=start_pos)
            total_reward += reward_episode
            game_win[i] = win_episode
            reward_history[i] = reward_episode
            total_reward_history[i] = total_reward

        # print('FINAL POLICY')
        # print(self.learner.cur_policy)
        # print('FINAL Q')
        # print(self.learner.q_table)
        # print('FINAL VALUE')
        # print(np.max(self.learner.q_table, axis=1))
        # print(f'game_win = {game_win}')
        segment = 10
        game_win = game_win.reshape((segment, epochs // segment))
        # print(f'game_win = {game_win}')
        game_win = np.sum(game_win, axis=1)
        # print(f'game_win = {game_win}')

        print(f'winning percentage = {game_win / (epochs // segment)}')

        if plot:
            fig, axes = plt.subplots(2, 1, figsize=(5, 4), dpi=200, sharex='all')
            axes[0].plot(np.arange(len(total_reward_history)), total_reward_history,
                         alpha=0.7, color='#d62728', label=r'$\xi$ = ' + f'{self.learner.xi}')
            axes[0].set_ylabel('Total rewards')
            axes[0].legend(loc='best')
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

            fig, ax = plt.subplots(1, 1, figsize=(6, 3), dpi=200)
            ax.plot(np.arange(1, segment + 1) * (epochs // segment), game_win / (epochs // segment), marker='o', markersize=2,
                    alpha=0.7, color='#2ca02c',
                    label=r'$\xi$ = ' + f'{self.learner.xi}')
            ax.set_ylabel('Winning percentage')
            ax.set_xlabel('Episode')
            plt.legend(loc='best')
            plt.tight_layout()
            plt.show()


        return total_reward


# problem = GridWorld('data/world00.csv', reward={0: -0.04, 1: 1.0, 2: -1.0, 3: np.NaN}, random_rate=0.2)
# n = 10
# random_start = np.zeros(n)
# for i in range(n):
#     print(f'Epoch: {i + 1}')
#     problem_solver = GridWorldMCSolver(problem, MCLearner, epsilon=1.0, xi=0.9, initial_q=2)
#     random_start[i] = problem_solver.train(100, start_pos=(2, 0), plot=False)
# print(f'average = {np.mean(random_start)}')
# print(f'min = {np.min(random_start)}')
# print(f'max = {np.max(random_start)}')
#
# bins = 100
# fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=300)
# ax.set_xlabel('Total rewards in a single game')
# ax.set_ylabel('Frequency')
# ax.hist(random_start, bins=bins, color='#1f77b4', edgecolor='black')
# plt.show()


np.random.seed(42)
problem = GridWorld('data/world02.csv', reward={0: -0.04, 1: 10.0, 2: -2.5, 3: np.NaN}, random_rate=0.2)
problem_solver = GridWorldMCSolver(problem, MCLearner, epsilon=1.0, xi=0.99, initial_q=0.0)
problem_solver.train(1000, start_pos=(5, 3), plot=True)


