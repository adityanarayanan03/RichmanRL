"""Tests for policies."""

from RichmanRL.utils import RandomPolicy
from RichmanRL.envs import RichmanEnv
from pettingzoo.classic import tictactoe_v3


def test_random_policy():
    """Makes sure random policy works normally on the tic-tac-toe env."""
    r = RichmanEnv(
        env=tictactoe_v3.raw_env(render_mode=None), capital=100, verbose=True
    )

    r.reset()
    print("\n")

    random_policy = RandomPolicy(None, 9, 0)

    while True:
        observation1, reward1, done1, _, _ = r.last("player_1")
        observation2, reward2, done2, _, _ = r.last("player_2")

        if done1 or done2:
            break

        legal_mask_1 = observation1["action_mask"][1]
        legal_mask_2 = observation2["action_mask"][1]

        print(f"Player 1 legal mask is {legal_mask_1}")
        print(f"Player 2 legal mask is {legal_mask_2}")

        player_1_action = random_policy(None, legal_mask_1)
        player_2_action = random_policy(None, legal_mask_2)

        print(f"Player 1 taking action {player_1_action}")
        print(f"Player 2 taking action {player_2_action}")

        r.step({"player_1": (5, player_1_action), "player_2": (1, player_2_action)})


def test_random_policy_bidding():
    """Makes sure random policy works normally on the tic-tac-toe env."""
    r = RichmanEnv(
        env=tictactoe_v3.raw_env(render_mode=None), capital=100, verbose=True
    )

    r.reset()
    print("\n")

    random_policy = RandomPolicy(None, 9, 0)
    random_bidding_policy = RandomPolicy(None, 201, 0)

    while True:
        observation1, reward1, done1, _, _ = r.last("player_1")
        observation2, reward2, done2, _, _ = r.last("player_2")

        if done1 or done2:
            break

        legal_mask_1 = observation1["action_mask"][1]
        legal_mask_2 = observation2["action_mask"][1]

        print(f"Player 1 legal mask is {legal_mask_1}")
        print(f"Player 2 legal mask is {legal_mask_2}")

        player_1_action = random_policy(None, legal_mask_1)
        player_2_action = random_policy(None, legal_mask_2)

        print(f"Player 1 taking action {player_1_action}")
        print(f"Player 2 taking action {player_2_action}")

        highest_bid_1 = observation1["action_mask"][0]
        legal_bids_1 = [1 if i <= highest_bid_1 else 0 for i in range(201)]
        highest_bid_2 = observation2["action_mask"][0]
        legal_bids_2 = [1 if i <= highest_bid_2 else 0 for i in range(201)]

        player_1_bid = random_bidding_policy(None, legal_bids_1)
        player_2_bid = random_bidding_policy(None, legal_bids_2)

        print(f"Player 1 random bid is {player_1_bid}")
        print(f"Player 2 random bid is {player_2_bid}")

        r.step(
            {
                "player_1": (player_1_bid, player_1_action),
                "player_2": (player_2_bid, player_2_action),
            }
        )
