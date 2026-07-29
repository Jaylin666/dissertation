"""Frozen scientific configuration for the dissertation analyses.

Every value in this module is taken from the validated historical scripts or
their tracked compact outputs. Changing a value here changes the scientific
specification and therefore requires a new analysis, not a maintenance
refactor.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple


FULL_HISTORY_START_YEAR = 1985
FULL_HISTORY_END_YEAR = 2025
TEST_YEAR = 2025
VALIDATION_YEARS: Tuple[int, ...] = (2023, 2024)

EXPECTED_FULL_HISTORY_MATCHES = 456_382
EXPECTED_UNIQUE_PLAYERS = 5_143
EXPECTED_TEST_MATCHES = 11_379
EXPECTED_VALIDATION_MATCHES = 23_888
EXPECTED_APPEARANCE_ROWS = 22_758

EXPECTED_SYSTEM_START_PLAYERS = 314
EXPECTED_WITHIN_BURN_IN_PLAYERS = 456
EXPECTED_POST_BURN_IN_PLAYERS_1990_2024 = 4_297
EXPECTED_TEST_YEAR_ENTRANTS = 76

EXPECTED_DEBUT_APPEARANCES = 76
EXPECTED_DEBUT_PLAYERS = 76
EXPECTED_DEBUT_MATCHES = 74
EXPECTED_EXACTLY_ONE_DEBUT_MATCHES = 72
EXPECTED_BOTH_DEBUT_MATCHES = 2

PRIMARY_BURN_IN_YEARS = 5
POST_BURN_IN_START_YEAR = 1990

BOOTSTRAP_REPETITIONS = 2_000
STEP_29_RANDOM_SEED = 20260713
STEP_31_RANDOM_SEED = 20260714
STEP_33_RANDOM_SEED = 20260715
STEP_34_RANDOM_SEED = 20260715
STEP_35_RANDOM_SEED = 20260716
STEP_37_RANDOM_SEED = 20260717


@dataclass(frozen=True)
class EloConfig:
    """Immutable Elo parameterisation."""

    name: str
    initial_rating: float
    k_factor: float
    scale: float

    def __post_init__(self) -> None:
        if self.k_factor <= 0.0:
            raise ValueError("Elo k_factor must be positive")
        if self.scale <= 0.0:
            raise ValueError("Elo scale must be positive")


CONSERVATIVE_ELO = EloConfig(
    name="Conservative_Elo",
    initial_rating=1500.0,
    k_factor=10.0,
    scale=500.0,
)
DEFAULT_ELO = EloConfig(
    name="Default_Elo",
    initial_rating=1500.0,
    k_factor=20.0,
    scale=500.0,
)
VALIDATION_BEST_ELO = EloConfig(
    name="Validation_best_Elo",
    initial_rating=1500.0,
    k_factor=30.0,
    scale=300.0,
)
ELO_CONFIGURATIONS: Tuple[EloConfig, ...] = (
    CONSERVATIVE_ELO,
    DEFAULT_ELO,
    VALIDATION_BEST_ELO,
)


@dataclass(frozen=True)
class GlickoConfig:
    """Immutable Glicko-1 and inactivity-inflation parameterisation."""

    name: str = "Glicko_low_inflation_match_by_match"
    initial_rating: float = 1500.0
    initial_rd: float = 350.0
    minimum_rd: float = 30.0
    maximum_rd: float = 350.0
    inactivity_unit: str = "month"
    inactivity_target_periods: int = 240
    rating_period: str = "match_by_match"

    def __post_init__(self) -> None:
        if self.minimum_rd <= 0.0:
            raise ValueError("minimum_rd must be positive")
        if self.maximum_rd < self.minimum_rd:
            raise ValueError("maximum_rd must be at least minimum_rd")
        if self.inactivity_target_periods <= 0:
            raise ValueError("inactivity_target_periods must be positive")
        if self.inactivity_unit != "month":
            raise ValueError("The frozen low-inflation configuration uses months")
        if self.rating_period != "match_by_match":
            raise ValueError("The frozen primary Glicko model is match-by-match")

    @property
    def inactivity_c(self) -> float:
        """Return the frozen Step 24 low-inflation C value."""

        numerator = (self.maximum_rd**2) - (self.minimum_rd**2)
        return math.sqrt(numerator / self.inactivity_target_periods)


GLICKO_LOW_INFLATION = GlickoConfig()


@dataclass(frozen=True)
class FirstAppearanceGolden:
    """Frozen 2025 first-recorded-appearance headline values."""

    mean_current_probability: float = 0.743448336412
    empirical_win_rate: float = 0.407894736842
    brier_score: float = 0.322316
    mean_opponent_rating: float = 1180.755015
    mean_initial_minus_opponent_gap: float = 319.244985
    mean_initial_minus_contemporaneous_median_gap: float = 262.658851


FIRST_APPEARANCE_GOLDEN = FirstAppearanceGolden()
