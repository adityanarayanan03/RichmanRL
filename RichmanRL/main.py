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

def compare_methods_hex(
    method_1: Literal["random"]
    | Literal["policy_gradient"]
    | Literal["policy_gradient_simplified"],
    method_2: Literal["random"]
    | Literal["policy_gradient"]
    | Literal["policy_gradient_simplified"],
) -> tuple[float, float, float]:
    """Compares given methods."""
    # Instantiate policies for this method
    if method_1 == "random":
        agent_1_bidding = RandomBiddingPolicy(None, 201, 0)
        agent_1_game = RandomGamePolicy(None, 121, 0)
    elif method_1 == "policy_gradient":
        agent_1_bidding, agent_1_game = train_reinforce_agent(
            "hex", training_steps=1_000
        )
    elif method_1 == "policy_gradient_simplified":
        agent_1_bidding, agent_1_game = train_reinforce_agent(
            "hex_simplified", training_steps=1_000
        )
    else:
        logger.error("Didn't understand method 1")
        raise ValueError("Didn't understand method 1 parameter")

    if method_2 == "random":
        agent_2_bidding = RandomBiddingPolicy(None, 201, 0)
        agent_2_game = RandomGamePolicy(None, 121, 0)
    elif method_2 == "policy_gradient":
        agent_2_bidding, agent_2_game = train_reinforce_agent(
            "hex", training_steps=1_000
        )
    elif method_2 == "policy_gradient_simplified":
        agent_2_bidding, agent_2_game = train_reinforce_agent(
            "hex_simplified", training_steps=1_000
        )
    else:
        logger.error("Didn't understand method 2")
        raise ValueError("Didn't understand method 2 parameter")

    # Play the policies against each other
    stats = evaluate_policies(
        "hex",
        agent_1_bidding,
        agent_1_game,
        agent_2_bidding,
        agent_2_game,
        num_samples=1_000,
    )
    logger.info(f"win, loss, tie for {method_1} against {method_2} is {stats}")

    return stats


def gladiator_ring(
    game: Literal["ttt"] | Literal["hex"],
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
                
                if game == "ttt":
                    stats = compare_methods_ttt(method1, method2)
                elif game == "hex":
                    stats = compare_methods_hex(method1, method2)
                else:
                    raise ValueError("you didn't specify game parameter correctly.")

                for p in range(3):
                    compare_stats[p] += stats[p]

            results[i][j] = tuple([x / runs for x in compare_stats])

    return results


if __name__ == "__main__":
    results = gladiator_ring("hex", methods=["random", "policy_gradient", "policy_gradient_simplified"], runs=3)
    logger.info(results)
