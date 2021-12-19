import numpy as np
import matplotlib.pyplot as plt

from GridWorld import GridWorld
from PolicyIteration import PolicyIteration

problem = GridWorld('data/world00.csv')
policy = [1, 1, 3, 1, 0, 0, 2, 0, 1, 2, 1, 0]
# policy = [0, 0, 1, 1, 1, 0, 3, 0, 0, 3, 1, 2]
# policy = [1, 1, 1, 1, 0, 0, 0, 0, 0, 3, 3, 3]

solver = PolicyIteration(problem.reward_function, problem.transition_model, gamma=0.9, init_policy=policy)

for d in range(100):
    # problem.plot_policy_iteration_values(policy=solver.policy, values=solver.values)
    for dummy in range(100):
        delta, total_delta = solver.one_policy_evaluation()
        # problem.plot_policy_iteration_values(policy=solver.policy, values=solver.values)
        if delta < 1e-3:
            print('Evaluation converge!')
            break
    # problem.plot_policy_iteration_values(policy=solver.policy, values=solver.values)
    update_policy_count = solver.run_policy_improvement()
    if update_policy_count == 0:
        print('Policy converge!')
        break

problem.plot_policy_iteration_values(policy=solver.policy, values=solver.values)
problem.random_start_policy(policy=solver.policy)


