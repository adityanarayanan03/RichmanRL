"""Tests for policies."""

from RichmanRL.utils import RandomGamePolicy, RandomBiddingPolicy
from RichmanRL.envs import RichmanEnv
from pettingzoo.classic import tictactoe_v3
from RichmanRL.utils import (
    pickle_policy,
    get_pickled_policy,
    ConstantBaseline,
    InGameNNPolicy,
    NoBiddingPolicy,
)
from RichmanRL.algs import REINFORCE
from tqdm import tqdm


def test_random_policy():
    """Makes sure random policy works normally on the tic-tac-toe env."""
    r = RichmanEnv(
        env=tictactoe_v3.raw_env(render_mode=None), capital=100, verbose=True
    )

    r.reset()
    print("\n")

    random_policy = RandomGamePolicy(None, 9, 0)

    while True:
        observation1, reward1, done1, _, _ = r.last("player_1")
        observation2, reward2, done2, _, _ = r.last("player_2")

        if done1 or done2:
            break

        legal_mask_1 = observation1["action_mask"][1]
        legal_mask_2 = observation2["action_mask"][1]

        print(f"Player 1 legal mask is {legal_mask_1}")
        print(f"Player 2 legal mask is {legal_mask_2}")

        player_1_action = random_policy(observation1)
        player_2_action = random_policy(observation2)

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

    random_policy = RandomGamePolicy(None, 9, 0)
    random_bidding_policy = RandomBiddingPolicy(None, 201, 0)

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


def test_pickle_policy():
    """Ensures that a policy can be pickled and unpickled and be unaffected.."""
    # This is basically the test from test_reinforce with pickling in the middle.
    r = RichmanEnv(
        env=tictactoe_v3.raw_env(render_mode=None), capital=100, verbose=True
    )

    r.reset()
    print("\n")

    reinforce = REINFORCE(
        r,
        NoBiddingPolicy(None, 201, 0),
        RandomGamePolicy(None, 9, 0),
        NoBiddingPolicy(None, 201, 0),
        InGameNNPolicy(18, 9, 0.0003),
        0.99,
        2000,
        ConstantBaseline(),
    )

    reinforce()

    in_game_policy = reinforce.agent_2_game_pi
    r = RichmanEnv(
        env=tictactoe_v3.raw_env(render_mode=None), capital=100, verbose=True
    )

    reinforce = REINFORCE(
        r,
        NoBiddingPolicy(None, 201, 0),
        RandomGamePolicy(None, 9, 0),
        NoBiddingPolicy(None, 201, 0),
        in_game_policy,
        0.99,
        1000,
        ConstantBaseline(),
    )

    total_reward = 0
    for x in tqdm(range(1000)):
        traj = reinforce._generate_trajectory()  # noqa: F841

        # find out the winner
        last_reward = traj["player_2"][-1][0]

        total_reward += last_reward
    print(total_reward)

    print("Pickling policy...")
    pickle_policy(
        in_game_policy,
        "REINFORCE_policy.pkl",
        "/Users/aditya/Documents/repos/ut/RichmanRL/",
    )

    print("Unpickling policy...")
    loaded_policy = get_pickled_policy(
        "REINFORCE_policy.pkl", "/Users/aditya/Documents/repos/ut/RichmanRL/"
    )

    print("Validating total accumulated reward")
    reinforce = REINFORCE(
        r,
        NoBiddingPolicy(None, 201, 0),
        RandomGamePolicy(None, 9, 0),
        NoBiddingPolicy(None, 201, 0),
        loaded_policy,
        0.99,
        1000,
        ConstantBaseline(),
    )

    total_reward = 0
    for x in tqdm(range(1000)):
        traj = reinforce._generate_trajectory()  # noqa: F841

        # find out the winner
        last_reward = traj["player_2"][-1][0]

        total_reward += last_reward
    print(total_reward)
