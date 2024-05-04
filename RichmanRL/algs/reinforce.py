"""Implements the REINFORCE algorithm."""

from RichmanRL.envs import RichmanEnv
from RichmanRL.envs import Hex
from pettingzoo.classic import tictactoe_v3
from RichmanRL.envs.typing_utils import RichmanAction, RichmanObservation
from RichmanRL.utils import Policy, ValueFunction
from itertools import count
from .typing_utils import AgentTrajectory
from typing import Union, Literal, Tuple
from tqdm import tqdm
import torch
import sys
from .theoretical_Hex import HexBiddingPolicy, HexGamePolicy, HexPolicy

from RichmanRL.utils import (
    RandomBiddingPolicy,
    RandomGamePolicy,
    BiddingNNPolicy,
    InGameNNPolicy,
    ConstantBaseline,
)
from RichmanRL.utils import get_pickled_policy

import logging

logger = logging.getLogger("reinforce.py")
logger.setLevel(logging.DEBUG)


class REINFORCE:
    """Implements the REINFORCE policy gradient algorithm."""

    def __init__(
        self,
        env: RichmanEnv,
        agent_1_bid_pi: Policy,
        agent_1_game_pi: Policy,
        agent_2_bid_pi: Policy,
        agent_2_game_pi: Policy,
        gamma: float,
        num_episodes: int,
        V: ValueFunction,
        verbose: bool = True,
    ) -> None:
        """Constructor for REINFORCE.

        Args:
            env : RichmanEnv - Environment to learn
            agent_1_bid_pi : Policy - Agent 1's bidding policy
            agent_1_game_pi : Policy - Agent 1's in-game policy
            agent_2_bid_pi : Policy - Agent 2's bidding policy
            agent_2_game_pi : Policy - Agent 2's in-game policy
            gamma : float - discount rate
            num_episodes : int - number of episodes to train for
            V : ValueFunction - Value function for baseline (optional)
            verbose: Whether or not to enable logging.
        """
        self.logger = logging.getLogger("REINFORCE")
        if verbose:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.ERROR)

        self.env = env
        self.agent_1_bid_pi = agent_1_bid_pi
        self.agent_1_game_pi = agent_1_game_pi
        self.agent_2_bid_pi = agent_2_bid_pi
        self.agent_2_game_pi = agent_2_game_pi
        self.gamma = gamma
        self.num_episodes = num_episodes
        self.V = V

    def _sample_actions(
        self, S_1: RichmanObservation, S_2: RichmanObservation
    ) -> RichmanAction:
        """Uses the instantiated policies to sample actions for both agents.

        Args:
            S_1 : Observation of agent 1
            S_2 : Observation of agent 2
            agent_1_mask : the legal mask returned from observe() on the env.
            agent_2_mask : the same as agent_1_mask from agent_2's perspective.

        Returns:
            RichmanAction describing both agents' actions.
        """
        player_1_bid = self.agent_1_bid_pi(S_1)
        player_1_move = self.agent_1_game_pi(S_1)

        if player_1_bid > S_1["action_mask"][0]:
            print(f"[ERROR] Player 1 sampled illegal action {player_1_bid}")
            sys.exit(0)

        player_2_bid = self.agent_2_bid_pi(S_2)
        player_2_move = self.agent_2_game_pi(S_2)

        if player_2_bid > S_2["action_mask"][0]:
            print(f"[ERROR] Player 2 sampled illegal action {player_2_bid}")
            sys.exit(0)

        return RichmanAction(
            player_1=(player_1_bid, player_1_move),
            player_2=(player_2_bid, player_2_move),
        )

    def _generate_trajectory(self) -> AgentTrajectory:
        """Generates a single trajectory."""
        self.env.reset()

        traj = AgentTrajectory(player_1=[], player_2=[])

        S_1, R_1, done1, _, _ = self.env.last("player_1")
        S_2, R_2, done2, _, _ = self.env.last("player_2")

        for t in count():
            A: RichmanAction = self._sample_actions(S_1, S_2)

            traj["player_1"].append((R_1, S_1, A))
            traj["player_2"].append((R_2, S_2, A))

            self.env.step(A)

            S_1, R_1, done1, _, _ = self.env.last("player_1")
            S_2, R_2, done2, _, _ = self.env.last("player_2")

            if done1 or done2:
                traj["player_1"].append((R_1, S_1, None))
                traj["player_2"].append((R_2, S_2, None))
                break

        return traj

    @torch.no_grad()
    def _update_from_traj(
        self, traj: AgentTrajectory, agent: Union[Literal["player_1", "player_2"]]
    ):
        """Update policies and value functions for a single player."""
        player_traj = traj[agent]

        # print("Updating a whole trajectory")
        for t in range(0, len(player_traj)):
            # print("Updating from a step of a trajectory")
            gamma_t = self.gamma**t

            # Compute the monte carlo return
            G = 0
            for k in range(t + 1, len(player_traj)):
                G += self.gamma ** (k - t - 1) * player_traj[k][0]

            delta = G - self.V(player_traj[t][1])

            self.V.update(player_traj[t][1], G)

            if agent == "player_1":
                self.agent_1_bid_pi.update(
                    state=player_traj[t][1],
                    action=player_traj[t][2],
                    gamma_t=gamma_t,
                    delta=delta,
                    agent=agent,
                )

                self.agent_1_game_pi.update(
                    state=player_traj[t][1],
                    action=player_traj[t][2],
                    gamma_t=gamma_t,
                    delta=delta,
                    agent=agent,
                )
            else:
                self.agent_2_bid_pi.update(
                    state=player_traj[t][1],
                    action=player_traj[t][2],
                    gamma_t=gamma_t,
                    delta=delta,
                    agent=agent,
                )

                self.agent_2_game_pi.update(
                    state=player_traj[t][1],
                    action=player_traj[t][2],
                    gamma_t=gamma_t,
                    delta=delta,
                    agent=agent,
                )

    def __call__(self):
        """Runs the actual learning."""
        for episode_idx in tqdm(range(self.num_episodes)):
            # sample a trajectory for both agents
            traj = self._generate_trajectory()

            # Inspect the trajectory for illegal bids
            for step in traj["player_2"]:
                action = step[2]

                if not action:
                    continue

                legal_bid = step[1]["action_mask"][0]
                if action["player_2"][0] > legal_bid:
                    self.logger.error("Found an inconsistency in player_2's trajectory")
                    self.logger.debug(
                        f"legal bid was {legal_bid} and action was {action}"
                    )  # noqa: E501

            self._update_from_traj(traj, "player_1")
            self._update_from_traj(traj, "player_2")

    def get_policies(self):
        """Returns all the policies."""
        return (
            self.agent_1_bid_pi,
            self.agent_1_game_pi,
            self.agent_2_bid_pi,
            self.agent_2_game_pi,
        )


def _train_ttt(training_steps: int) -> Tuple[Policy, Policy]:
    r = RichmanEnv(
        env=tictactoe_v3.raw_env(render_mode=None), capital=100, verbose=True
    )

    reinforce = REINFORCE(
        r,
        RandomBiddingPolicy(None, 201, 0),
        RandomGamePolicy(None, 9, 0),
        BiddingNNPolicy(19, 201, 0.0003),
        InGameNNPolicy(18, 9, 0.0003),
        0.99,
        training_steps,
        ConstantBaseline(),
    )

    reinforce()

    return reinforce.get_policies()[2:4]

def _train_hex(training_steps: int) -> Tuple[Policy, Policy]:
    r = RichmanEnv(
        env=Hex(render_mode=None), capital=100, verbose=True
    )
    
    reinforce = REINFORCE(
        r,
        RandomBiddingPolicy(None, 201, 0),
        RandomGamePolicy(None, 121, 0),
        BiddingNNPolicy(243, 201, 0.0003),
        InGameNNPolicy(242, 121, 0.0003),
        0.99,
        training_steps,
        ConstantBaseline(),
    )
    
    reinforce()
    
    return reinforce.get_policies()[2:4]


def train_reinforce_agent(
    game: Literal["ttt", "hex"], training_steps: int = 100_000
) -> Tuple[Policy, Policy]:
    """Trains a reinforce agent.

    Args:
        game: Literal["ttt", "hex"] - which game to train an agent for.
        training_steps: int - Number of steps to train for.
    """
    if game == "ttt":
        return _train_ttt(training_steps)
    elif game == "hex":
        return _train_hex(training_steps)
    else:
        logger.error(f"Game {game} not implemented.")
        raise NotImplementedError(f"Game {game} not implemented.")
