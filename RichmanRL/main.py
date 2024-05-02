"""Implements the main functionality."""

from __future__ import annotations
from typing import Literal

from RichmanRL.utils import (
    RandomGamePolicy,
    RandomBiddingPolicy,
    InGameNNPolicy,
    BiddingNNPolicy,
    NoBiddingPolicy,
)

from RichmanRL.algs import train_reinforce_agent
from RichmanRL.utils.evaluation import evaluate_policies
import numpy as np

import logging

logging.basicConfig()
logger = logging.getLogger("main.py")
logger.setLevel(logging.DEBUG)


def compare_methods(
    game: Literal["ttt"] | Literal["hex"],
    method_1: Literal["random"] | Literal["policy_gradient"],
    method_2: Literal["random"] | Literal["policy_gradient"],
) -> tuple[float, float, float]:
    """Compares given methods."""
    if game == "hex":
        logger.error("Hex hasn't been implemented yet in compare methods.")
        raise ValueError("Hex hasn't been implemented yet in compare.")

    # Instantiate policies for this method
    if method_1 == "random":
        agent_1_bidding = RandomBiddingPolicy(None, 201, 0)
        agent_1_game = RandomGamePolicy(None, 9, 0)
    elif method_1 == "policy_gradient":
        agent_1_bidding, agent_1_game = train_reinforce_agent(
            "ttt", training_steps=1_000
        )
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


def gladiator_ring(
    methods: list = ["random", "policy_gradient"], runs: int = 5
) -> list[list[tuple]]:
    """Plays all the methods against each other."""
    num_methods = len(methods)
    results = [[None for _ in range(num_methods)] for _ in range(num_methods)]

    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods):
            if j < i:
                continue
            
            for k in range(runs):
                logger.info(
                    f"\n Comparing {method1} and {method2} on run {k+1}/{runs}\n"
                )
                stats = compare_methods("ttt", method1, method2)
                results[i][j] = stats

    return results


if __name__ == "__main__":
    # stats = compare_methods("ttt", "random", "policy_gradient")
    # print(f"[INFO] win, loss, tie for random against policy_gradient is {stats}")
    # logger.info(f"win, loss, tie for random against policy_gradient is {stats}")
    results = gladiator_ring()
    logger.info(results)
