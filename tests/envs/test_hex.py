"""Tests RichmanEnv wrapper using tictactoe_v3."""

from RichmanRL.envs import RichmanEnv
from RichmanRL.envs import Hex
from tqdm import tqdm
import time

from RichmanRL.utils import RandomBiddingPolicy, RandomGamePolicy


def test_instantiation():
    """Test instantiation of RichmanEnv."""
    RichmanEnv(env=Hex(render_mode="human"), capital=100, verbose=True)


def test_observation():
    """Tests making an observation using the tuple action."""
    r = RichmanEnv(
        env=Hex(render_mode="human"), capital=100, verbose=True
    )

    print("\n")
    print(r.observe("player_1"))
    print(r.observe("player_2"))


def test_action():
    """Tests making an action."""
    r = RichmanEnv(
        env=Hex(render_mode="human"), capital=100, verbose=True
    )

    r.reset()

    """
    Both players start with 100 chips. Suppose the following happens. Player 2
    continually bets 10 dollars. Player 1 bets 15. Player 1 wins the tic tac 
    toe directly.
    """

    print("\n")
    print("Taking first round of actions")
    r.step({"player_1": (15, 0), "player_2": (10, 0)})
    print(r.observe("player_1"))
    print(r.env.terminations)
    # r.render()

    print("Taking second round of actions")
    r.step({"player_1": (15, 1), "player_2": (10, 1)})
    print(r.observe("player_1"))
    print(r.env.terminations)
    # r.render()

    print("Taking third round of actions")
    r.step({"player_1": (15, 2), "player_2": (10, 2)})
    print(r.observe("player_1"))
    print(r.env.terminations)
    # r.render()


def test_speed():
    """Tests speed of making plays in the game."""
    r = RichmanEnv(
        env=Hex(render_mode=None), capital=100, verbose=True
    )

    r.reset()

    """
    We run 100 games, each where player 1 wins deterministically. Then report the time.
    """

    print("\n")

    start_time = time.time()
    for idx in tqdm(range(10_000)):
        r.reset()

        r.step({"player_1": (1, 0), "player_2": (0, 0)})
        # print(r.env.terminations)
        r.step({"player_1": (1, 4), "player_2": (0, 0)})
        # print(r.env.terminations)
        r.step({"player_1": (1, 8), "player_2": (0, 0)})
        # print(r.env.terminations)

    print(f"Finished {10_000} iterations in {time.time() - start_time} seconds.")


def test_action_space():
    """Tests the action space function for an agent.

    The action space includes the legal mask, so we test to see and make sure
    that only legal actions are returned by the action space function.
    """
    r = RichmanEnv(
        env=Hex(render_mode=None), capital=100, verbose=True
    )
    r.reset()

    print("\n")
    print(r.action_space("player_1"))
    print(r.action_space("player_1").sample())
    print(r.observe("player_1"))


def test_tiebreak():
    """Ensures that ties on bidding assign moves randomly."""
    r = RichmanEnv(
        env=Hex(render_mode="human"), capital=100, verbose=True
    )

    r.reset()

    print("\n")
    print("Taking first round of actions")
    r.step({"player_1": (10, 0), "player_2": (10, 0)})
    print(r.observe("player_1"))
    print(r.env.terminations)
    r.render()

    print("Taking second round of actions")
    r.step({"player_1": (10, 1), "player_2": (10, 1)})
    print(r.observe("player_1"))
    print(r.env.terminations)
    r.render()

    print("Taking third round of actions")
    r.step({"player_1": (10, 2), "player_2": (10, 2)})
    print(r.observe("player_1"))
    print(r.env.terminations)
    r.render()

def test_trajectory_generation():
    """Makes sure trajectories can be sampled properly."""
    r = RichmanEnv(
        env=Hex(render_mode=None), capital=100, verbose=True
    )

    r.reset()
    print("\n")

    a = RandomBiddingPolicy(None, 201, 0)
    b = RandomGamePolicy(None, 121, 0)
    c = RandomBiddingPolicy(None, 201, 0)
    d = RandomGamePolicy(None, 121, 0)

    total_reward = 0
    for x in tqdm(range(1_000)):
        traj = r.generate_trajectory(a, b, c, d)  # noqa: F841

        #find out the winner
        last_reward = traj["player_2"][-1][0]
        
        total_reward += last_reward
    print(f"Total observed reward was {total_reward}")