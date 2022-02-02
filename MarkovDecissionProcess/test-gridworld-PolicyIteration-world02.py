from GridWorld import GridWorld
from PolicyIteration import PolicyIteration

import numpy as np

problem = GridWorld('data/world02.csv', reward={0: -0.04, 1: 10.0, 2: -2.5, 3: np.NaN}, random_rate=0.2)
problem.plot_map(fig_size=(15, 8))
solver = PolicyIteration(problem.reward_function, problem.transition_model, gamma=0.9, init_policy=None)
solver.train()
problem.plot_policy(policy=solver.policy, fig_size=(10, 8))
problem.random_start_policy(policy=solver.policy, start_pos=(5, 3), n=1000)


