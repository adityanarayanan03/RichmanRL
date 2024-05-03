"""Contains utilities for evalutation of policies."""
from RichmanRL.envs import RichmanEnv
from RichmanRL.envs import Hex
from typing import Literal
from tqdm import tqdm
from pettingzoo.classic import tictactoe_v3

import logging
logger = logging.getLogger("evaluation.py")
logger.setLevel(logging.DEBUG)


def evaluate_policies_hex(
    agent_1_bid_pi,
    agent_1_game_pi,
    agent_2_bid_pi,
    agent_2_game_pi,
    num_samples: int = 10_000,
    mode = None
):
    r = RichmanEnv(
            env=Hex(render_mode=mode), capital=100, verbose=True
        )
    wins = 0
    losses = 0
    ties = 0
    for _ in tqdm(range(num_samples)):
        traj = r.generate_trajectory(
            agent_1_bid_pi, agent_1_game_pi, agent_2_bid_pi, agent_2_game_pi
        )

        last_reward = traj["player_1"][-1][0]

        if last_reward == 1:
            wins += 1
        elif last_reward == -1:
            losses += 1
        else:
            ties += 1

    return wins / num_samples, losses / num_samples, ties / num_samples
    
    

def evaluate_policies_ttt(
    agent_1_bid_pi,
    agent_1_game_pi,
    agent_2_bid_pi,
    agent_2_game_pi,
    num_samples: int = 10_000,
    mode = None
):
    """Evaluate how good a policy is.

    Args:
        game: ttt or hex
        agent_1_bid_pi: Policy,
        agent_1_game_pi: Policy,
        agent_2_bid_pi: Policy,
        agent_2_game_pi: Policy,
        num_samples: int - Number of trial games

    Returns:
        (float, float, float) - player_1, player_2, tie
    """

    r = RichmanEnv(
            env=tictactoe_v3.raw_env(render_mode=mode), capital=100, verbose=True
        )
    wins = 0
    losses = 0
    ties = 0
    for _ in tqdm(range(num_samples)):
        traj = r.generate_trajectory(
            agent_1_bid_pi, agent_1_game_pi, agent_2_bid_pi, agent_2_game_pi
        )

        last_reward = traj["player_1"][-1][0]

        if last_reward == 1:
            wins += 1
        elif last_reward == -1:
            losses += 1
        else:
            ties += 1

    return wins / num_samples, losses / num_samples, ties / num_samples

def evaluate_policies(
    game: Literal["ttt", "hex"],
    agent_1_bid_pi,
    agent_1_game_pi,
    agent_2_bid_pi,
    agent_2_game_pi,
    num_samples: int = 10_000,
    mode = None
):
    if(game == "ttt"):
        evaluate_policies_ttt(agent_1_bid_pi,
                            agent_1_game_pi,
                            agent_2_bid_pi,
                            agent_2_game_pi,
                            num_samples, mode)
    else:
        evaluate_policies_hex(agent_1_bid_pi,
                            agent_1_game_pi,
                            agent_2_bid_pi,
                            agent_2_game_pi,
                            num_samples, mode)
