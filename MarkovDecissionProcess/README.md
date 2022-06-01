# Markov Decision Process
### GridWorld.py
A grid world problem from the book *Artificial Intelligence A Modern Approach* by Stuart Russell and Peter Norvig is implemented.

More details: https://medium.com/@ngao7/markov-decision-process-basics-3da5144d3348

Under the class **GridWorld**, the following functions are provided:
- *get_state_from_pos(pos)*: transfer a position in the grid world into an integer representing state.
- *get_pos_from_state(state)*: transfer an integer representing state back into a position in the grid world.
- *get_reward_function()*: calculate the reward function r(s) of MDP.
- *get_transition_model()*: calculate the transitional model p(s'|s, a) of MDP.
- *generate_random_policy()*: initialize a policy of random actions.
- *execute_policy(policy, start_pos)*: get the total reward starting from the start_pos following the given policy.
- *random_start_policy(policy, n)*: repeatedly execute the given policy for n times.
- *blackbox_move(s, a)*: simulate an environment where the agent can not access the reward function and transition model. The agent provides the current state s and an action a, this function returns the next state s' of the agent and the reward assigned to the agent through this move.
- *plot_map()*: visualize the map of the grid world.
- *plot_policy(policy)*: visualize the given policy.
- *visualize_value_policy(policy, values)*: visualize the given policy and utility values

### PolicyIteration.py
Implement policy iteration to solve a MDP.

More details: https://medium.com/@ngao7/markov-decision-process-policy-iteration-42d35ee87c82

Under the class **PolicyIteration**, the following functions are provided:
- *one_policy_evaluation()*: perform one sweep of policy evaluation.
- *run_policy_evaluation(tol)*: perform sweeps of policy evaluation iteratively with a stop criterion of the given tol.
- *run_policy_improvement()*: perform one policy improvement.
- *train(tol, plot)*: perform policy iteration by iteratively alternates policy evaluation and policy improvement. If plot is true, the function plots learning curves showing number of sweeps in each iteration and number of policy updates in each iteration.

### ValueIteration.py
Implement value iteration to solve a MDP.

More details: https://medium.com/@ngao7/markov-decision-process-value-iteration-2d161d50a6ff

Under the class **ValueIteration**, the following functions are provided:
- *one_iteration()*: perform one iteration of value evaluation.
- *get_policy()*: determine policy based on current utility.
- *train(tol, plot)*: perform value iteration with a stop criterion of the given tol. If plot is true, the function plots learning curves showing maximum value change in each iteration.

### ADPLearner.py
Implement a model-based adaptive dynamic programming (ADP) agent to learn a MDP.

More details: https://medium.com/@ngao7/reinforcement-learning-model-based-adp-learner-with-code-implementation-6ad73867fb1e

Under the class **ADPLearner**, the following functions are provided:
- *percept(s, a, s', r)*: update the learned reward and transition model after each step moved in MDP from the given set of (s, a, s', r) associated with that step. 
- *actuate(s')*: return the next action for the agent based on currently learned policy for state s'.
- *policy_update*: update the learned policy after each episode (an episode here is defined as a series of steps from a starting state to an ending state).

### MCLearner.py
Implement a model-free Monte Carlo (MC) agent to learn a MDP.

More details: https://medium.com/@ngao7/reinforcement-learning-model-free-mc-learner-with-code-implementation-f9f475296dcb

Under the class **MCLearner**, similar API is provided as the ADP learner:
- *percept(s, a, s', r)*: update the G values of the current episode after each step moved in MDP from the given set of (s, a, s', r) associated with that step. 
- *actuate(s')*: return the next action for the agent based on currently learned policy for state s'.
- *policy_update*: update the learned policy after each episode.





