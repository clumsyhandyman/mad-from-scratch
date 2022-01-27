import numpy as np

from GridWorld import GridWorld
from PolicyIteration import PolicyIteration

problem = GridWorld('data/world00.csv', reward={0: -0.04, 1: 1.0, 2: -1.0, 3: np.NaN}, random_rate=0.2)
policy = [1, 1, 3, 1, 0, 0, 2, 0, 1, 2, 1, 0]

solver = PolicyIteration(problem.reward_function, problem.transition_model, gamma=0.9, init_policy=policy)
solver.train()

problem.visualize_value_policy(policy=solver.policy, values=solver.values)
problem.random_start_policy(policy=solver.policy, start_pos=(2, 0), n=1000)
