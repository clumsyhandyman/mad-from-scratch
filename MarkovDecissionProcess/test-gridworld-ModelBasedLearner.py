from GridWorld import GridWorld
from ValueIteration import ValueIteration
from ModelBasedLearner import ModelBasedLearner

import numpy as np

problem = GridWorld('data/world00.csv')
learner = ModelBasedLearner(num_states=problem.num_states, num_actions=problem.num_actions)


def train_one_game_play():
    start_pos = (2, 0)
    s = problem.get_state_from_pos(start_pos)
    while True:
        a = learner.get_action()
        s_prime, r = problem.blackbox_move(s, a)
        learner.update(s, a, s_prime, r)
        if r != 1 and r != -1:
            s = s_prime
        else:
            break


def train(epochs, games_per_epoch):
    best_score = np.zeros(epochs)
    worst_score = np.zeros(epochs)
    average_score = np.zeros(epochs)

    for epoch in range(epochs):
        for game in range(games_per_epoch):
            print(f'Epoch {epoch + 1} Game {game + 1}')
            train_one_game_play()

        assumed_reward_function = learner.learned_reward_function
        assumed_transition_model = learner.generate_transition_model_for_solver()

        solver = ValueIteration(assumed_reward_function, assumed_transition_model, gamma=0.9)
        solver.train(plot=False)

        best_score[epoch], worst_score[epoch], average_score[epoch] = problem.random_start_policy(policy=solver.policy, n=1000)

    print(f'best score = {best_score}')
    print(f'worst score = {worst_score}')
    print(f'average score = {average_score}')


train(epochs=20, games_per_epoch=1)
