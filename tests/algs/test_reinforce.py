"""Tests the behavior of the REINFORCE algorithm."""

from RichmanRL.envs import RichmanEnv
from RichmanRL.algs import REINFORCE, train_reinforce_agent, HexPolicy, HexBiddingPolicy, HexGamePolicy
from RichmanRL.utils import (
    RandomGamePolicy,
    ConstantBaseline,
    NoBiddingPolicy,
    InGameNNPolicy,
    BiddingNNPolicy,
    RandomBiddingPolicy,  # noqa: F401
)
from pettingzoo.classic import tictactoe_v3
from tqdm import tqdm
import pdb
from RichmanRL.utils.evaluation import evaluate_policies


def test_trajectory_generation():
    """Makes sure trajectories can be sampled properly."""
    r = RichmanEnv(
        env=tictactoe_v3.raw_env(render_mode=None), capital=100, verbose=True
    )

    r.reset()
    print("\n")

    reinforce = REINFORCE(
        r,
        NoBiddingPolicy(None, 201, 0),
        RandomGamePolicy(None, 9, 0),
        BiddingNNPolicy(19, 201, 0.0003),
        InGameNNPolicy(18, 9, 0.0003),
        0.99,
        10_000,
        ConstantBaseline(),
    )

    for _ in tqdm(range(1_000)):
        traj = reinforce._generate_trajectory()  # noqa: F841

        # Inspect the trajectory for illegal bids
        for step in traj["player_2"]:
            action = step[2]

            if not action:
                continue

            legal_bid = step[1]["action_mask"][0]
            if action["player_2"][0] > legal_bid:
                print("[ERROR] Found an inconsistency in player_2's trajectory")
                print(f"[DEBUG] legal bid was {legal_bid} and action was {action}")

                pdb.set_trace()

                assert False


def test_reinforce():
    """Makes sure reinforce runs without errors."""
    bidding_policy, game_policy = train_reinforce_agent("hex", 10_000)
    hex_base = HexPolicy()
    hex_game, hex_bidding = HexGamePolicy(hex_base), HexBiddingPolicy(hex_base)
    stats = evaluate_policies(
        "hex",
        # RandomBiddingPolicy(None, 201, 0),
        # RandomGamePolicy(None, 9, 0),
        hex_bidding,
        hex_game,
        bidding_policy,
        game_policy,
    )

    print(f"win, loss, tie is {stats}")

test_reinforce()
