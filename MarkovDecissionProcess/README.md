# Markov Desion Process
### GridWorld.py
A grid world problem from the book *Artificial Intelligence A Modern Approach* by Stuart Russell and Peter Norvig is implemented.

More details: https://medium.com/@ngao7/markov-decision-process-basics-3da5144d3348

Under the class **GridWorld**, the following functions are provided:
- *get_state_from_pos(pos)*: discretize position in the grid world into an integer representing state.
- *get_pos_from_state(state)*: transfer the integer representing state back into a position in the grid world.
- *get_reward_function()*: calculate the reward function r(s) of MDP.
- *get_transition_model(random_rate)*: calculate the transitional model p(s'|s, a) of MDP.
- *generate_random_policy()*: initialize a policy by random actions.
- *execute_policy(policy, start_pos)*: get the total reward starting from the start_pos following the given policy.
- *random_start_policy(policy, n)*: repeatedly execute the given policy for n times.
- *plot_reward()*: visualize the reward function r(s).
- *plot_transition_model()*: visualize the transitional model p(s'|s, a).
- *plot_policy(policy)*: visualize the given policy.
- *visualize_value_policy*: visualize the utility and policy during policy iteration or value iteration.

### PolicyIteration.py
Implement policy iteration to solve a MDP.

More details: https://medium.com/@ngao7/markov-decision-process-policy-iteration-42d35ee87c82

Under the class **PolicyIteration**, the following functions are provided:
- *one_policy_evaluation()*: perform one sweep of policy evaluation.
- *run_policy_evaluation(tol)*: perform sweeps of policy evaluation iteratively with a stop criterion of the given tol.
- *run_policy_improvement()*: perform one policy improvement.
- *train(tol, plot)*: perform policy iteration by iteratively alternates policy evaluation and policy improvement. 

### ValueIteration.py
Implement value iteration to solve a MDP.

More details: https://medium.com/@ngao7/markov-decision-process-value-iteration-2d161d50a6ff

Under the class **ValueIteration**, the following functions are provided:
- *one_iteration()*: perform one iteration of value evaluation.
- *get_policy()*: determine policy based on current utility.
- *train(tol, plot)*: perform value iteration with a stop criterion of the given tol.

