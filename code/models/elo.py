"""Elo equations and adaptive update rules."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Union

from code.config import EloConfig


Number = Union[int, float]


@dataclass(frozen=True)
class EloUpdate:
    """Result of one Elo update."""

    winner_rating_before: float
    loser_rating_before: float
    predicted_winner_win: float
    rating_change: float
    winner_rating_after: float
    loser_rating_after: float


def _finite(value: Number, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def expected_score(rating_a: Number, rating_b: Number, scale: Number) -> float:
    """Return player A's expected score."""

    rating_a_value = _finite(rating_a, "rating_a")
    rating_b_value = _finite(rating_b, "rating_b")
    scale_value = _finite(scale, "scale")
    if scale_value <= 0.0:
        raise ValueError("scale must be positive")
    probability = 1.0 / (
        1.0 + 10.0 ** ((rating_b_value - rating_a_value) / scale_value)
    )
    return float(probability)


def update_winner_loser(
    winner_rating: Number,
    loser_rating: Number,
    k_factor: Number,
    scale: Number,
) -> EloUpdate:
    """Update one winner and loser."""

    winner_before = _finite(winner_rating, "winner_rating")
    loser_before = _finite(loser_rating, "loser_rating")
    k_value = _finite(k_factor, "k_factor")
    if k_value <= 0.0:
        raise ValueError("k_factor must be positive")
    probability = expected_score(winner_before, loser_before, scale)
    change = k_value * (1.0 - probability)
    return EloUpdate(
        winner_rating_before=winner_before,
        loser_rating_before=loser_before,
        predicted_winner_win=probability,
        rating_change=float(change),
        winner_rating_after=float(winner_before + change),
        loser_rating_after=float(loser_before - change),
    )


def update_with_config(
    winner_rating: Number,
    loser_rating: Number,
    config: EloConfig,
) -> EloUpdate:
    """Update one game with the given configuration."""

    return update_winner_loser(
        winner_rating=winner_rating,
        loser_rating=loser_rating,
        k_factor=config.k_factor,
        scale=config.scale,
    )


def adaptive_k_total(previous_total_games: Number) -> float:
    """Return adaptive K from total prior games."""

    games = int(previous_total_games)
    if games < 0:
        raise ValueError("previous_total_games cannot be negative")
    if games < 20:
        return 30.0
    if games < 100:
        return 20.0
    return 10.0


def adaptive_k_previous_year(previous_year_games: Number) -> float:
    """Return adaptive K from previous-year games."""

    games = int(previous_year_games)
    if games < 0:
        raise ValueError("previous_year_games cannot be negative")
    if games <= 5:
        return 30.0
    if games <= 30:
        return 20.0
    return 10.0
