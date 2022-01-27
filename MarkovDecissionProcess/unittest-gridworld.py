import numpy as np
import unittest

from GridWorld import GridWorld


class GridWorldTest(unittest.TestCase):
    def runTest(self):
        pass

    def test_reward_function(self):
        problem = GridWorld('data/world00.csv', reward={0: -0.04, 1: 1.0, 2: -1.0, 3: np.nan}, random_rate=0.2)
        my_rewards = problem.reward_function
        expected_rewards = np.array([-0.04, -0.04, -0.04, 1.0,
                                     -0.04, None, -0.04, -1.0,
                                     -0.04, -0.04, -0.04, -0.04])
        for s in range(len(expected_rewards)):
            if expected_rewards[s] is not None:
                self.assertEqual(expected_rewards[s], my_rewards[s],
                                 msg=f'Reward function of state {s} is NOT correct')
        print('UnitTest for reward function passed successfully!')

    def test_transition_model_size(self):
        problem = GridWorld('data/world00.csv', reward={0: -0.04, 1: 1.0, 2: -1.0, 3: np.nan}, random_rate=0.2)
        transition_model = problem.transition_model
        self.assertEqual(transition_model.shape, (12, 4, 12),
                         msg=f'Size of transition model is NOT correct')
        print('UnitTest for size of transition model passed successfully!')

    def test_transition_model(self):
        problem = GridWorld('data/world02.csv', reward={0: -0.04, 1: 1.0, 2: -1.0, 3: np.nan}, random_rate=0.2)
        transition_model = problem.transition_model
        self.assertTrue(np.all(np.sum(transition_model, axis=2) == 1.0),
                        msg=f'Sum of transition model for a certain s,a should be 1.0')
        print('UnitTest for sum of probability of transition model passed successfully!')


#
# problem = GridWorld('data/world00.csv', reward={0: -0.04, 1: 1.0, 2: -1.0, 3: np.NaN})
# # problem = GridWorld('data/world02.csv', reward={0: -0.04, 1: 10.0, 2: -10.0, 3: np.NaN})
#
# # problem.plot_map(fig_size=(10, 8))
#
#
# reward_function = problem.reward_function
# print(f'reward function =')
# for s in range(len(reward_function)):
#     print(f'State s = {s}, Reward R({s}) = {reward_function[s]}')
#
# transition_model = problem.transition_model
# print(f'transition model =')
# for s in range(transition_model.shape[0]):
#     print('======================================')
#     for a in range(transition_model.shape[1]):
#         print('--------------------------------------')
#         for s_prime in range(transition_model.shape[2]):
#             print(f's = {s}, a = {a}, s\' = {s_prime}, p = {transition_model[s, a, s_prime]}')
#
if __name__ == '__main__':
    unittest.main()

