"""Implements the main functionality."""

from __future__ import annotations
from typing import Literal

from RichmanRL.utils import (
    RandomGamePolicy,
    RandomBiddingPolicy,
    InGameNNPolicy,
    BiddingNNPolicy,
    NoBiddingPolicy,
    get_pickled_policy
)

from RichmanRL.algs import train_reinforce_agent
from RichmanRL.utils.evaluation import evaluate_policies

from RichmanRL.TTT_DP import TTTPolicy, get_policy

import numpy as np

import logging

logging.basicConfig()
logger = logging.getLogger("main.py")
logger.setLevel(logging.DEBUG)


def compare_methods_ttt(
    method_1: Literal["random"]
    | Literal["policy_gradient"]
    | Literal["policy_gradient_simplified"]
    | Literal["Tabular"],
    method_2: Literal["random"]
    | Literal["policy_gradient"]
    | Literal["policy_gradient_simplified"]
    | Literal["Tabular"],
) -> tuple[float, float, float]:
    """Compares given methods."""
    # Instantiate policies for this method
    if method_1 == "random":
        agent_1_bidding = RandomBiddingPolicy(None, 201, 0)
        agent_1_game = RandomGamePolicy(None, 9, 0)
    elif method_1 == "policy_gradient":
        agent_1_bidding, agent_1_game = train_reinforce_agent(
            "ttt", training_steps=1_000
        )
    elif method_1 == "policy_gradient_simplified":
        agent_1_bidding, agent_1_game = train_reinforce_agent(
            "ttt_simplified", training_steps=1_000
        )
    elif method_1 == "Tabular":
        agent_1_bidding, _ = train_reinforce_agent("ttt", training_steps=1_000)
        agent_1_game = get_pickled_policy("TTT_n_step_policy.pkl", "/Users/aditya/Documents/repos/ut/RichmanRL/")
    else:
        logger.error("Didn't understand method 1")
        raise ValueError("Didn't understand method 1 parameter")

    if method_2 == "random":
        agent_2_bidding = RandomBiddingPolicy(None, 201, 0)
        agent_2_game = RandomGamePolicy(None, 9, 0)
    elif method_2 == "policy_gradient":
        agent_2_bidding, agent_2_game = train_reinforce_agent(
            "ttt", training_steps=1_000
        )
    elif method_2 == "policy_gradient_simplified":
        agent_2_bidding, agent_2_game = train_reinforce_agent(
            "ttt_simplified", training_steps=1_000
        )
    elif method_2 == "Tabular":
        agent_2_bidding, _ = train_reinforce_agent("ttt", training_steps=1_000)
        agent_2_game = get_pickled_policy("TTT_n_step_policy.pkl", "/Users/aditya/Documents/repos/ut/RichmanRL/")
    else:
        logger.error("Didn't understand method 2")
        raise ValueError("Didn't understand method 2 parameter")

    # Play the policies against each other
    stats = evaluate_policies(
        "ttt",
        agent_1_bidding,
        agent_1_game,
        agent_2_bidding,
        agent_2_game,
        num_samples=1_000,
    )
    logger.info(f"win, loss, tie for {method_1} against {method_2} is {stats}")

    return stats


def gladiator_ring_ttt(
    methods: list = ["random", "policy_gradient", "policy_gradient_simplified", "Tabular"],
    runs: int = 10,
) -> list[list[tuple]]:
    """Plays all the methods against each other."""
    num_methods = len(methods)
    results = [[None for _ in range(num_methods)] for _ in range(num_methods)]

    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods):
            if j <= i:
                continue

            compare_stats = [0, 0, 0]
            for k in range(runs):
                logger.info(
                    f"\n Comparing {method1} and {method2} on run {k+1}/{runs}\n"
                )
                stats = compare_methods_ttt(method1, method2)

                for p in range(3):
                    compare_stats[p] += stats[p]

            results[i][j] = tuple([x / runs for x in compare_stats])

    return results


if __name__ == "__main__":
    # stats = compare_methods("ttt", "random", "policy_gradient")
    # print(f"[INFO] win, loss, tie for random against policy_gradient is {stats}")
    # logger.info(f"win, loss, tie for random against policy_gradient is {stats}")
    results = gladiator_ring_ttt()
    logger.info(results)
