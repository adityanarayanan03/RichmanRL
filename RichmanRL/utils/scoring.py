"""Score policies against optimal.

We can only do this for Hex.
"""

from __future__ import annotations
from RichmanRL.algs import HexPolicy, HexGamePolicy, HexBiddingPolicy
from RichmanRL.envs import RichmanEnv, Hex
import numpy as np
from tqdm import tqdm


import logging

logging.basicConfig()
logger = logging.getLogger("scoring.py")
logger.setLevel(logging.DEBUG)


def score_nn_bids(
    nn_bid_pi: Policy,
    nn_game_pi: Policy,
    num_evals: int = 100,
):
    """Score the bids given by a bidding policy.

    The algorithm for this is as follows:

    We have 4 policies - enough to generate complete trajectories.
    The metric we use to score nn bids is the average mass under
    the optimal bid in the output softmax'ed activation map from the bidding
    neural netowrk. The optimal bid is given with the MC solution
    to hex.

    Args:
        nn_bid_pi: Policy - TRAINED nn bidding policy
        nn_game_pi: Policy - TRAINED nn game policy
        num_evals: int - Number of evaluations.
    """
    r = RichmanEnv(env=Hex(render_mode=None), capital=100, verbose=True)

    hex_base = HexPolicy()
    hex_game, hex_bidding = HexGamePolicy(hex_base), HexBiddingPolicy(hex_base)

    scores = []
    for eval_idx in tqdm(range(num_evals)):
        traj = r.generate_trajectory(nn_bid_pi, nn_game_pi, hex_bidding, hex_game)

        nn_traj = traj["player_1"]
        theoretical_traj = traj["player_2"]

        traj_score = 0
        for idx in range(len(nn_traj)):
            R1, S1, A1 = nn_traj[idx]
            R2, S2, A2 = theoretical_traj[idx]

            if not A1 or not A2:
                continue

            theoretical_bid = hex_bidding(
                S1
            )  # What would the optimal policy have done in this position
            nn_probs, nn_taken = nn_bid_pi(
                S1, return_probs=True
            )  # What did we do instead?

            score = nn_probs[theoretical_bid]

            if score == 0:
                logger.error("Mass under theoretical bid was 0!")

            traj_score += score

        traj_score /= len(nn_traj)
        scores.append(traj_score)

    return np.mean(scores)
