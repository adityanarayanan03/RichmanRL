"""Implements the REINFORCE algorithm."""

from RichmanRL.envs import RichmanEnv
from RichmanRL.envs.typing_utils import RichmanAction
from RichmanRL.utils import Policy, ValueFunction
from itertools import count
from .typing_utils import AgentTrajectory


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
        """
        self.env = env
        self.agent_1_bid_pi = agent_1_bid_pi
        self.agent_1_game_pi = agent_1_game_pi
        self.agent_2_bid_pi = agent_2_bid_pi
        self.agent_2_game_pi = agent_2_game_pi
        self.gamma = gamma
        self.num_episodes = num_episodes
        self.V = V

    def _sample_actions(self, S_1, S_2, agent_1_mask, agent_2_mask) -> RichmanAction:
        """Uses the instantiated policies to sample actions for both agents.

        Args:
            S_1 : Observation of agent 1
            S_2 : Observation of agent 2
            agent_1_mask : the legal mask returned from observe() on the env.
            agent_2_mask : the same as agent_1_mask from agent_2's perspective.

        Returns:
            RichmanAction describing both agents' actions.
        """
        highest_bid_1 = agent_1_mask[0]
        legal_bids_1 = [1 if i <= highest_bid_1 else 0 for i in range(201)]
        legal_moves_1 = agent_1_mask[1]

        highest_bid_2 = agent_2_mask[0]
        legal_bids_2 = [1 if i <= highest_bid_2 else 0 for i in range(201)]
        legal_moves_2 = agent_2_mask[1]

        player_1_bid = self.agent_1_bid_pi(S_1, legal_bids_1)
        player_1_move = self.agent_1_game_pi(S_1, legal_moves_1)

        player_2_bid = self.agent_2_bid_pi(S_2, legal_bids_2)
        player_2_move = self.agent_2_game_pi(S_2, legal_moves_2)

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
            A: RichmanAction = self._sample_actions(
                S_1["observation"],
                S_2["observation"],
                S_1["action_mask"],
                S_2["action_mask"],
            )

            traj["player_1"].append((R_1, S_1, A["player_1"]))
            traj["player_2"].append((R_2, S_2, A["player_2"]))

            self.env.step(A)

            S_1, R_1, done1, _, _ = self.env.last("player_1")
            S_2, R_2, done2, _, _ = self.env.last("player_2")

            if done1 or done2:
                traj["player_1"].append((R_1, S_1, A["player_1"]))
                traj["player_2"].append((R_2, S_2, A["player_2"]))
                break

        return traj
