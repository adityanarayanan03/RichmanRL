"""Tests the behavior of the REINFORCE algorithm."""

from RichmanRL.envs import RichmanEnv
from RichmanRL.algs import REINFORCE
from RichmanRL.utils import RandomGamePolicy, RandomBiddingPolicy, ConstantBaseline
from pettingzoo.classic import tictactoe_v3


def test_trajectory_generation():
    """Makes sure trajectories can be sampled properly."""
    r = RichmanEnv(
        env=tictactoe_v3.raw_env(render_mode="human"), capital=100, verbose=True
    )

    r.reset()
    print("\n")

    reinforce = REINFORCE(
        r,
        RandomBiddingPolicy(None, 201, 0),
        RandomGamePolicy(None, 9, 0),
        RandomBiddingPolicy(None, 201, 0),
        RandomGamePolicy(None, 9, 0),
        0.99,
        10,
        None,
    )

    for _ in range(10):
        traj = reinforce._generate_trajectory()

        print("-----Player_1-----")
        for step in traj["player_1"]:
            print(f"{step}\n")

        print("-----Player_2-----")
        for step in traj["player_2"]:
            print(f"{step}\n")

def test_reinforce():
    """Makes sure trajectories can be sampled properly."""
    r = RichmanEnv(
        env=tictactoe_v3.raw_env(render_mode=None), capital=100, verbose=True
    )

    r.reset()
    print("\n")

    reinforce = REINFORCE(
        r,
        RandomBiddingPolicy(None, 201, 0),
        RandomGamePolicy(None, 9, 0),
        RandomBiddingPolicy(None, 201, 0),
        RandomGamePolicy(None, 9, 0),
        0.99,
        10_000,
        ConstantBaseline(),
    )

    reinforce()