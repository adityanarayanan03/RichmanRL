from typing import Iterable, Tuple

import numpy as np
from .env import EnvSpec
from .policy import Policy
import math
from tqdm import tqdm


class QPolicy(Policy):
    def __init__(self, Q):
        #####################
        # TODO: Implement the methods in this class.
        # You may add any arguments to the constructor as you see fit.
        # "QPolicy" here refers to a policy that takes
        #    greedy actions w.r.t. Q values
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


def on_policy_n_step_td(
    env_spec: EnvSpec,
    trajs: Iterable[Iterable[Tuple[int, int, int, int]]],
    n: int,
    alpha: float,
    initV: np.array,
) -> Tuple[np.array]:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        n: how many steps?
        alpha: learning rate
        initV: initial V values; np array shape of [nS]
    ret:
        V: $v_pi$ function; numpy array shape of [nS]
    """

    #####################
    # TODO: Implement On Policy n-Step TD algorithm
    # sampling (Hint: Sutton Book p. 144)
    #####################
    gamma = env_spec.gamma
    V = initV

    for traj in tqdm(trajs):
        T = len(traj)
        for t in range(T):
            tau = t - n + 1
            if tau >= 0:
                G = sum(
                    [
                        np.power(gamma, i - tau) * traj[i][2]
                        for i in range(tau, min(tau + n, T))
                    ]
                )
                if tau + n < T:
                    G += np.power(gamma, n) * V[traj[tau + n][0]]
                V[traj[tau][0]] += alpha * (G - V[traj[tau][0]])

    return V


def off_policy_n_step_sarsa(
    env_spec: EnvSpec,
    trajs: Iterable[Iterable[Tuple[int, int, int, int]]],
    bpi: Policy,
    n: int,
    alpha: float,
    initQ: np.array,
) -> Tuple[np.array, Policy]:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        bpi: behavior policy used to generate trajectories
        n: how many steps?
        alpha: learning rate
        initQ: initial Q values; np array shape of [nS,nA]
    ret:
        Q: $q_star$ function; numpy array shape of [nS,nA]
        policy: $pi_star$; instance of policy class
    """

    #####################
    # TODO: Implement Off Policy n-Step SARSA algorithm
    # sampling (Hint: Sutton Book p. 149)
    #####################
    gamma = env_spec.gamma
    Q = initQ
    # pi = GreedyPolicy(Q)
    policy = QPolicy(Q)

    for traj in trajs:
        T = len(traj)
        total_reward = 0
        for t in range(T):
            tau = t - n + 1
            total_reward += traj[t][2]
            if tau >= 0:
                rho = 1
                # for i in range(tau, min(tau+n-1, T-1)):
                rho = np.prod(
                    [
                        (
                            policy.action_prob(traj[i][0], traj[i][1])
                            / bpi.action_prob(traj[i][0], traj[i][1])
                        )
                        for i in range(tau + 1, min(tau + n, T))
                    ]
                )
                G = sum(
                    [
                        np.power(gamma, i - tau) * traj[i][2]
                        for i in range(tau, min(tau + n, T))
                    ]
                )
                if tau + n < T:
                    G += (
                        np.power(gamma, n)
                        * Q[traj[int(tau + n)][0], traj[int(tau + n)][1]]
                    )
                Q[traj[int(tau)][0], traj[int(tau)][1]] += (
                    alpha * rho * (G - Q[traj[int(tau)][0], traj[int(tau)][1]])
                )
                policy = QPolicy(Q)
    return Q, policy
