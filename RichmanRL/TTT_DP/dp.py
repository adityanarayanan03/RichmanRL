from typing import Tuple

import numpy as np
from env import EnvWithModel
from policy import Policy


class GreedyPolicy(Policy):
    def __init__(self, Q):
        #####################
        # TODO: Implement the methods in this class.
        # You may add any arguments to the constructor as you see fit.
        #####################
        self.actions = np.argmax(Q, axis=1)

    def action_prob(self, state: int, action: int) -> float:
        """
        input:
            state, action
        return:
            \pi(a|s)
        """
        return 1 if self.actions[state] == action else 0

    def action(self, state: int) -> int:
        """
        input:
            state
        return:
            action
        """
        return self.actions[state]


def value_prediction(
    env: EnvWithModel, pi: Policy, initV: np.array, theta: float
) -> Tuple[np.array, np.array]:
    """
    inp:
        env: environment with model information, i.e. you know transition dynamics and reward function
        pi: policy
        initV: initial V(s); numpy array shape of [nS,]
        theta: exit criteria
    return:
        V: $v_\pi$ function; numpy array shape of [nS]
        Q: $q_\pi$ function; numpy array shape of [nS,nA]
    """

    def expected_update(state: int, action: int, values: np.array):
        accum = 0
        gamma = env.spec.gamma
        for new_state in range(env.spec.nS):
            prob_transition = env.TD[state][action][new_state]
            reward = env.R[state][action][new_state]
            accum += prob_transition * (reward + gamma * values[new_state])
        return accum

    num_states = env.spec.nS
    num_actions = env.spec.nA
    Q = np.zeros((num_states, num_actions))
    prev_values = initV
    while True:
        new_values = np.zeros(num_states)
        delta = 0
        for state in range(num_states):
            for action in range(num_actions):
                Q[state][action] = expected_update(state, action, prev_values)
                new_values[state] += pi.action_prob(state, action) * Q[state][action]
            delta = max(delta, abs(prev_values[state] - new_values[state]))
        prev_values = new_values
        if delta < theta:
            break
    #####################
    # TODO: Implement Value Prediction Algorithm (Hint: Sutton Book p.75)
    #####################
    return prev_values, Q


def value_iteration(
    env: EnvWithModel, initV: np.array, theta: float
) -> Tuple[np.array, Policy]:
    """
    inp:
        env: environment with model information, i.e. you know transition dynamics and reward function
        initV: initial V(s); numpy array shape of [nS,]
        theta: exit criteria
    return:
        value: optimal value function; numpy array shape of [nS]
        policy: optimal deterministic policy; instance of Policy class
    """

    #####################
    # TODO: Implement Value Iteration Algorithm (Hint: Sutton Book p.83)
    #####################

    def expected_update(state: int, action: int, values: np.array):
        accum = 0
        gamma = env.spec.gamma
        for new_state in range(env.spec.nS):
            prob_transition = env.TD[state][action][new_state]
            reward = env.R[state][action][new_state]
            accum += prob_transition * (reward + gamma * values[new_state])
        return accum

    num_states = env.spec.nS
    num_actions = env.spec.nA
    Q = np.zeros((num_states, num_actions))
    prev_values = initV
    while True:
        delta = 0
        new_values = np.ones(num_states) * -np.inf
        for state in range(num_states):
            for action in range(num_actions):
                Q[state][action] = expected_update(state, action, prev_values)
                new_values[state] = max(new_values[state], Q[state][action])
            delta = max(delta, abs(prev_values[state] - new_values[state]))
        prev_values = new_values
        if delta < theta:
            break
    return prev_values, GreedyPolicy(Q)
