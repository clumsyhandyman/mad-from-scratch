import numpy as np

from GridWorld import GridWorld


problem = GridWorld('data/world00.csv', reward={0: -0.04, 1: 1.0, 2: -1.0, 3: np.NaN}, random_rate=0.2)
# problem = GridWorld('data/world02.csv', reward={0: -0.04, 1: 10.0, 2: -10.0, 3: np.NaN}, random_rate=0.2)

problem.plot_map(fig_size=(10, 8))

init_policy = problem.generate_random_policy()
problem.plot_policy(init_policy, fig_size=(20, 10))
problem.visualize_value_policy(init_policy, np.zeros(problem.num_states), fig_size=(10, 8))

reward_function = problem.reward_function
print(f'reward function =')
for s in range(len(reward_function)):
    print(f'State s = {s}, Reward R({s}) = {reward_function[s]}')

transition_model = problem.transition_model
print(f'transition model =')
for s in range(transition_model.shape[0]):
    print('======================================')
    for a in range(transition_model.shape[1]):
        print('--------------------------------------')
        for s_prime in range(transition_model.shape[2]):
            print(f's = {s}, a = {a}, s\' = {s_prime}, p = {transition_model[s, a, s_prime]}')



