"""Score policies against optimal.

We can only do this for Hex.
"""

from __future__ import annotations
from RichmanRL.algs import HexPolicy, HexGamePolicy, HexBiddingPolicy
from RichmanRL.utils import RandomBiddingPolicy, RandomGamePolicy
from RichmanRL.envs import RichmanEnv, Hex
import numpy as np
from tqdm import tqdm
from scipy import stats

import logging

logging.basicConfig()
logger = logging.getLogger("scoring.py")
logger.setLevel(logging.DEBUG)


def score_nn(
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
    game_scores = []
    for eval_idx in tqdm(range(num_evals)):
        traj = r.generate_trajectory(
            nn_bid_pi,
            nn_game_pi,
            RandomBiddingPolicy(None, 201, 0),
            RandomGamePolicy(None, 121, 0),
        )

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
            
            #print(f"[DEBG] before game policy")
            theoretical_game_action, _ = hex_game(S1, return_prob=True)
            #print(f"[DEBUG] Theoretical game action vector is {theoretical_game_action}")
            nn_game_action_probs, _ = nn_game_pi(S1, return_probs = True)
            
            #Compare with kendall tau
            #print(f"[DEBUG] Shape of theoretical is {theoretical_game_action.shape}")
            #print(f"[DEBUG] Shape of nn is {nn_game_action_probs.shape}")
            res = stats.kendalltau(theoretical_game_action, nn_game_action_probs)
            game_scores.append(res.statistic)
            #print(f"[DEBUG] kendall tau is {res.statistic}")

            score = nn_probs[theoretical_bid]
            #print(f"[DEBUG] Score is {score}")
            #print(f"[DEBUG] nn_probs is {nn_probs}")
            #print(f"[DEBUG] Theoretical bid is {theoretical_bid}")

            # We have a giga problem if the theoretical bid is not legal
            if not 0 <= theoretical_bid <= S1["action_mask"][0]:
                raise ValueError("Theoretical bid was not legal!")

            traj_score += score

        traj_score /= len(nn_traj)
        scores.append(traj_score)

    return np.mean(scores), np.mean(game_scores)