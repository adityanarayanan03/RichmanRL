from typing import Iterable, Tuple

import numpy as np
from .env import EnvSpec
from .policy import Policy


def off_policy_mc_prediction_ordinary_importance_sampling(
    env_spec: EnvSpec,
    trajs: Iterable[Iterable[Tuple[int, int, int, int]]],
    bpi: Policy,
    pi: Policy,
    initQ: np.array,
) -> np.array:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        bpi: behavior policy used to generate trajectories
        pi: evaluation target policy
        initQ: initial Q values; np array shape of [nS,nA]
    ret:
        Q: $q_pi$ function; numpy array shape of [nS,nA]
    """

    #####################
    # TODO: Implement Off Policy Monte-Carlo prediction algorithm using ordinary importance
    # sampling (Hint: Sutton Book p. 109, every-visit implementation is fine)
    #####################
    num_states, num_actions = env_spec.nS, env_spec.nA
    gamma = env_spec.gamma
    counts = np.zeros((num_states, num_actions))
    for traj in trajs:
        G = 0
        W = 1
        for i in range(len(traj) - 1, -1, -1):
            state, action, reward, next_state = traj[i]
            G = gamma * G + reward
            counts[state][action] += 1
            initQ[state][action] += (
                W / counts[state][action] * (G - initQ[state][action])
            )
            W *= pi.action_prob(state, action) / bpi.action_prob(state, action)

    return initQ


def off_policy_mc_prediction_weighted_importance_sampling(
    env_spec: EnvSpec,
    trajs: Iterable[Iterable[Tuple[int, int, int, int]]],
    bpi: Policy,
    pi: Policy,
    initQ: np.array,
) -> np.array:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using behavior policy bpi
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        bpi: behavior policy used to generate trajectories
        pi: evaluation target policy
        initQ: initial Q values; np array shape of [nS,nA]
    ret:
        Q: $q_pi$ function; numpy array shape of [nS,nA]
    """

    #####################
    # TODO: Implement Off Policy Monte-Carlo prediction algorithm using weighted importance
    # sampling (Hint: Sutton Book p. 110, every-visit implementation is fine)
    #####################
    num_states, num_actions = env_spec.nS, env_spec.nA
    gamma = env_spec.gamma
    counts = np.zeros((num_states, num_actions))
    for traj in trajs:
        G = 0
        W = 1
        for i in range(len(traj) - 1, -1, -1):
            state, action, reward, next_state = traj[i]
            G = gamma * G + reward
            counts[state][action] += W
            initQ[state][action] += (
                W / counts[state][action] * (G - initQ[state][action])
            )
            W *= pi.action_prob(state, action) / bpi.action_prob(state, action)

    return initQ
