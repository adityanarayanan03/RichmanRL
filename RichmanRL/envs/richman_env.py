"""A generic wrapper that turns any pettingzoo environment into a Richman game."""
from pettingzoo import AECEnv
from typing import Union, Literal, Tuple, Any
from gymnasium import spaces
from .typing_utils import RichmanAction, RichmanObservation
import logging
import numpy as np


class RichmanEnv:
    """Wrapper for any pettingzoo env."""

    def __init__(self, env: AECEnv, capital: int = 100, verbose: bool = True) -> None:
        """Constructs RichmanEnv.

        Args:
            env: AECEnv - PettingZoo AEC Environment to wrap with bidding. This should
                be a raw_env, not an environment, since we will violate some of the
                checks not done in raw_env.
            capital: int - Starting capital of players.
            verbose: bool - Whether or not to have logging output
        """
        # Setup logging
        self.logger = logging.getLogger("RichmanEnv")
        if verbose:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.ERROR)

        self.env = env
        self.start_capital = capital
        self.stacks = {agent: capital for agent in self.env.agents}

    def observe(
        self, agent: Union[Literal["player_1", "player_2"]]
    ) -> RichmanObservation:
        """Returns an observation.

        Args:
            agent: Union[Literal["player_1", "player_2]] - Agent to observe for

        Returns:
            Tuple[int, int, Any] - representing: own_stack, opp_stack, game_obs
        """
        sub_observation = self.env.observe(agent)
        if agent == "player_1":
            output = RichmanObservation(
                observation = (self.stacks["player_1"], 
                               self.stacks["player_2"], 
                               sub_observation["observation"]), 
                action_mask= (self.stacks["player_1"], 
                              sub_observation["action_mask"])
            )
            return output

        elif agent == "player_2":
            output = RichmanObservation(
                observation = (self.stacks["player_2"], 
                               self.stacks["player_1"], 
                               sub_observation["observation"]), 
                action_mask= (self.stacks["player_2"], 
                              sub_observation["action_mask"])
            )
            return output

        else:
            raise ValueError(f"Agent: {agent} not in set of players.")

    def observation_space(
        self, agent: Union[Literal["player_1", "player_2"]]
    ) -> spaces.Tuple:
        """Returns the observation space for this game.

        Args:
            agent: Union[Literal["player_1", "player_2]] - Agent to observe for

        The observation space will be returned as a 3-tuple of
        gymnasium spaces.Space types. The first two will correspond to the
        stacks of the players, the third to the observation space of the game.
        """
        return spaces.Tuple(
            (
                spaces.Discrete(self.start_capital * 2 + 1),
                spaces.Discrete(self.start_capital * 2 + 1),
                self.env.observation_space(agent),
            )
        )

    def action_space(
        self, agent: Union[Literal["player_1", "player_2"]]
    ) -> spaces.Tuple:
        """Returns the action sapce for a given agent.

        Args:
            agent: Union[Literal["player_1", "player_2]] - Agent to observe for
        """
        return spaces.Tuple(
            (
                spaces.Discrete(self.start_capital * 2 + 1),
                self.env.action_space(agent),
            )
        )

    def step(self, action: RichmanAction) -> None:
        """Implements bidding selection and then defaults to environment step.

        When learning a sub-game policy or value function, pass bids equal to 
        each other. This will result in random player selection with equal probability.

        Args:
            action: RichmanAction - Actions from BOTH agents, as a tuple of
            bidding and game action.
        """
        # Validate both actions-- log a warning and return for now if invalid action
        if (
            action["player_1"][0] > self.stacks["player_1"]
            or action["player_2"][0] > self.stacks["player_2"]
        ):
            self.logger.error(
                f"Invalid stack size in action {action}, available stacks are {self.stacks}"  # noqa: E501
            )
            raise ValueError(
                f"Invalid stack size in action {action}, available stacks are {self.stacks}"  # noqa: E501
            )

        bet_1 = action["player_1"][0]
        bet_2 = action["player_2"][0]

        if bet_1 > bet_2:
            # Player 1 wins the bid, and transfer stacks to player2
            self.stacks["player_2"] += bet_1
            self.stacks["player_1"] -= bet_1

            self.env.agent_selection = "player_1"
            self.env.step(action["player_1"][1])
            return

        elif bet_2 > bet_1:
            # Player 2 wins the bid, and transfers stacks to player1
            self.stacks["player_1"] += bet_2
            self.stacks["player_2"] -= bet_2

            self.env.agent_selection = "player_2"
            self.env.step(action["player_2"][1])
            return

        else:
            self.logger.error(f'''Bid sizes are equal, with action was {action}. 
                              Make sure this was intentional, such as when learning a 
                              sub-game policy or value function. 
                              This will result in random player selection.''')
        
            if np.random.random() > 0.5:
                 # Player 1 wins the bid, and transfer stacks to player2
                self.stacks["player_2"] += bet_1
                self.stacks["player_1"] -= bet_1

                self.env.agent_selection = "player_1"
                self.env.step(action["player_1"][1])
                return
            
            else:
                # Player 2 wins the bid, and transfers stacks to player1
                self.stacks["player_1"] += bet_2
                self.stacks["player_2"] -= bet_2

                self.env.agent_selection = "player_2"
                self.env.step(action["player_2"][1])
                return

    def reset(self, *args, **kwargs) -> None:
        """Resets the environment.

        *args and **kwargs will be passed to the self.env environment
        for reset parameters. Captial for agents will be reset to
        instantiated value.
        """
        self.stacks = self.stacks = {
            agent: self.start_capital for agent in self.env.agents
        }

        self.env.reset(*args, **kwargs)
    
    def close(self, *args, **kwargs) -> None:
        """Mostly just a wrapper around game env close."""
        self.env.close(*args, **kwargs)

    def render(self, *args, **kwargs) -> None:
        """Just a wrapper around game env render."""
        self.env.render(*args, **kwargs)
