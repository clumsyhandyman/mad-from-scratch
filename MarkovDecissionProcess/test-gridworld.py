import numpy as np
import matplotlib.pyplot as plt

from GridWorld import GridWorld

problem = GridWorld('data/world00.csv')
# problem.plot_reward()
# problem.plot_transition_model()
# print(problem.reward_function)
# # policy = problem.generate_random_policy()
# policy = [1, 1, 3, 1, 0, 0, 2, 0, 1, 2, 1, 0]
# policy = [0, 0, 1, 1, 1, 0, 3, 0, 0, 3, 1, 2]
policy = [1, 1, 1, 1, 0, 0, 0, 0, 0, 3, 3, 3]
print(f'random policy = {policy}')
problem.plot_policy(policy)
#
n = 10000
scores = np.zeros(n)
for i in range(n):
    print(f'i = {i}')
    scores[i] = problem.execute_policy(policy)

print(f'max = {np.max(scores)}')
print(f'min = {np.min(scores)}')
print(f'mean = {np.mean(scores)}')
print(f'std = {np.std(scores)}')

bins = 100
fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=300)
ax.set_xlabel('Total rewards in a single game')
ax.set_ylabel('Frequency')
ax.hist(scores, bins=bins, color='#1f77b4', edgecolor='black')
plt.tight_layout()
plt.show()


