from GridWorld import GridWorld
from PolicyIteration import PolicyIteration

problem = GridWorld('data/world00.csv')
policy = [1, 1, 3, 1, 0, 0, 2, 0, 1, 2, 1, 0]

solver = PolicyIteration(problem.reward_function, problem.transition_model, gamma=0.9, init_policy=policy)
solver.train()

problem.visualize_value_policy(policy=solver.policy, values=solver.values)
problem.random_start_policy(policy=solver.policy)
