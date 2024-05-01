"""Unsimplified Policy Gradient."""

import omegaconf
from RichmanRL.envs import RichmanEnv
from pettingzoo.classic import tictactoe_v3
from RichmanRL.algs import REINFORCE
from RichmanRL.utils import (
    RandomGamePolicy,
    ConstantBaseline,
    BiddingNNPolicy,
    InGameNNPolicy,
    RandomBiddingPolicy
)


def reinforce_against_random(
    env: str = "ttt",
    num_evals: int = 1,
    training_trajs: int = 5_000,
    testing_trajs: int = 1_000,
):
    """Train a policy gradient algorithm against random play.

    Args:
        env: str - default ttt
        num_evals: Number of rounds to evaluate over
        training_trajs: int - Number of traning iterations
        testing_trajs: int - Number of testing trajectories
    """
    if env == "ttt":
        r = RichmanEnv(
            env=tictactoe_v3.raw_env(render_mode=None), capital=100, verbose=True
        )

    for i in range(num_evals):
        reinforce = REINFORCE(
            r,
            RandomBiddingPolicy(None, 201, 0),
            RandomGamePolicy(None, 9, 0),
            BiddingNNPolicy(19, 201, 0.0003),
            InGameNNPolicy(18, 9, 0.0003),
            0.99,
            training_trajs,
            ConstantBaseline(),
        )

        reinforce()

        performance = r.evaluate_policies(*reinforce.get_policies(), testing_trajs)

        print(
            f"""Performance of reinforce on evaluation {i+1}/{num_evals} against random is {performance}"""  # noqa: E501
        )


if __name__ == "__main__":
    config = omegaconf.OmegaConf.load("config.yaml")
    reinforce_against_random(**config)
