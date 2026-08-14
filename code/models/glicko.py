"""Glicko-1 equations."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


DEFAULT_RATING = 1500.0
DEFAULT_RD = 350.0
MIN_RD = 30.0
MAX_RD = 350.0
Q = math.log(10) / 400
C = 0.0


@dataclass(frozen=True)
class SingleGameUpdate:
    """Result of one Glicko-1 update."""

    player1_rating_before: float
    player1_rd_before: float
    player2_rating_before: float
    player2_rd_before: float
    score1: float
    predicted_player1_win: float
    player1_rating_after: float
    player1_rd_after: float
    player2_rating_after: float
    player2_rd_after: float


def clamp_rd(rd: float) -> float:
    """Clamp RD to the configured range."""

    return min(MAX_RD, max(MIN_RD, float(rd)))


def g_function(rd: float) -> float:
    """Return the Glicko scaling factor."""

    rd = float(rd)
    return 1.0 / math.sqrt(1.0 + (3.0 * (Q**2) * (rd**2)) / (math.pi**2))


def expected_score(rating: float, opponent_rating: float, opponent_rd: float) -> float:
    """Return expected score using the opponent's RD."""

    g_rd = g_function(opponent_rd)
    exponent = -g_rd * (float(rating) - float(opponent_rating)) / 400.0
    return 1.0 / (1.0 + 10.0**exponent)


def _as_float_list(values: Iterable[float]) -> list[float]:
    return [float(value) for value in values]


def update_player_glicko(
    rating: float,
    rd: float,
    opponent_ratings: Iterable[float],
    opponent_rds: Iterable[float],
    scores: Iterable[float],
) -> tuple[float, float]:
    """Update one player over a Glicko-1 rating period."""

    opponent_ratings = _as_float_list(opponent_ratings)
    opponent_rds = _as_float_list(opponent_rds)
    scores = _as_float_list(scores)

    if not (len(opponent_ratings) == len(opponent_rds) == len(scores)):
        raise ValueError("opponent_ratings, opponent_rds, and scores must have the same length")

    if len(scores) == 0:
        return float(rating), float(rd)

    rating = float(rating)
    rd = clamp_rd(rd)

    g_values = [g_function(opponent_rd) for opponent_rd in opponent_rds]
    expected_values = [
        expected_score(rating, opponent_rating, opponent_rd)
        for opponent_rating, opponent_rd in zip(opponent_ratings, opponent_rds)
    ]

    information_sum = sum(
        (g_j**2) * expected_j * (1.0 - expected_j)
        for g_j, expected_j in zip(g_values, expected_values)
    )
    if information_sum <= 0.0:
        return rating, rd

    d2 = 1.0 / ((Q**2) * information_sum)
    denominator = (1.0 / (rd**2)) + (1.0 / d2)
    rating_delta_sum = sum(
        g_j * (score_j - expected_j)
        for g_j, score_j, expected_j in zip(g_values, scores, expected_values)
    )

    new_rating = rating + (Q / denominator) * rating_delta_sum
    new_rd = math.sqrt(1.0 / denominator)

    return float(new_rating), clamp_rd(new_rd)


def update_two_players_single_game(
    player1_rating: float,
    player1_rd: float,
    player2_rating: float,
    player2_rd: float,
    score1: float,
) -> SingleGameUpdate:
    """Update both players for one game."""

    score1 = float(score1)
    score2 = 1.0 - score1
    predicted_player1_win = expected_score(player1_rating, player2_rating, player2_rd)

    player1_rating_after, player1_rd_after = update_player_glicko(
        player1_rating,
        player1_rd,
        [player2_rating],
        [player2_rd],
        [score1],
    )
    player2_rating_after, player2_rd_after = update_player_glicko(
        player2_rating,
        player2_rd,
        [player1_rating],
        [player1_rd],
        [score2],
    )

    return SingleGameUpdate(
        player1_rating_before=float(player1_rating),
        player1_rd_before=float(player1_rd),
        player2_rating_before=float(player2_rating),
        player2_rd_before=float(player2_rd),
        score1=score1,
        predicted_player1_win=predicted_player1_win,
        player1_rating_after=player1_rating_after,
        player1_rd_after=player1_rd_after,
        player2_rating_after=player2_rating_after,
        player2_rd_after=player2_rd_after,
    )
