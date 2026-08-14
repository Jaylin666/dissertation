"""Player orientation helpers."""

from __future__ import annotations

from typing import Literal, Tuple

from code.models.glicko import expected_score as glicko_expected_score


Side = Literal["a", "b"]


def canonical_players(player_one: int, player_two: int) -> Tuple[int, int]:
    """Return Player A and Player B in ascending database-ID order."""

    first = int(player_one)
    second = int(player_two)
    if first == second:
        raise ValueError("A match must contain two distinct players")
    return (first, second) if first < second else (second, first)


def canonical_outcome(winner_id: int, player_a_id: int, player_b_id: int) -> int:
    """Return one for a Player A win and zero for a Player B win."""

    winner = int(winner_id)
    player_a = int(player_a_id)
    player_b = int(player_b_id)
    if winner == player_a:
        return 1
    if winner == player_b:
        return 0
    raise ValueError("winner_id must equal player_a_id or player_b_id")


def focal_outcome(outcome_a: int, focal_side: Side) -> int:
    """Return the focal-player outcome."""

    outcome = int(outcome_a)
    if outcome not in {0, 1}:
        raise ValueError("outcome_a must be 0 or 1")
    if focal_side == "a":
        return outcome
    if focal_side == "b":
        return 1 - outcome
    raise ValueError(f"Unsupported focal side: {focal_side}")


def focal_probability(probability_a: float, focal_side: Side) -> float:
    """Orient player-A probability to the focal player."""

    probability = float(probability_a)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability_a must be in [0, 1]")
    if focal_side == "a":
        return probability
    if focal_side == "b":
        return 1.0 - probability
    raise ValueError(f"Unsupported focal side: {focal_side}")


def direct_player_a_glicko_probability(
    rating_a: float,
    rating_b: float,
    rd_b: float,
) -> float:
    """Return direct Player A probability using the opponent RD."""

    return glicko_expected_score(rating_a, rating_b, rd_b)
