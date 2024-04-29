"""Tests the behavior of the REINFORCE algorithm."""

from RichmanRL.envs import RichmanEnv
from RichmanRL.algs import REINFORCE
from RichmanRL.utils import (
    RandomGamePolicy,
    ConstantBaseline,
    NoBiddingPolicy,
    InGameNNPolicy,
)
from pettingzoo.classic import tictactoe_v3
from tqdm import tqdm


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
        NoBiddingPolicy(None, 201, 0),
        InGameNNPolicy(18, 9, 0.0003),
        0.99,
        10_000,
        ConstantBaseline(),
    )

    for _ in tqdm(range(10_000)):
        traj = reinforce._generate_trajectory()  # noqa: F841

"""         print("-----Player_1-----")
        for step in traj["player_1"]:
            print(f"{step}\n")

        print("-----Player_2-----")
        for step in traj["player_2"]:
            print(f"{step}\n") """


def test_reinforce():
    """Makes sure reinforce runs without errors."""
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

        #find out the winner
        last_reward = traj["player_2"][-1][0]
        
        total_reward += last_reward
    print(total_reward)
