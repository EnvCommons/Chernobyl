"""Reward calculation for nuclear power plant management.

Evaluates agent performance across multiple safety dimensions:
core integrity, radiation containment, containment integrity,
fuel integrity, and hydrogen safety.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from physics import REACTOR_PARAMS, ReactorParams
from reactor import ReactorState


@dataclass(frozen=True)
class SafetyLimits:
    """Safety limits for a reactor type."""
    clad_temp_limit: float          # Cladding temperature limit (C)
    containment_pressure_limit: float  # Containment design pressure (MPa)
    rated_power: float              # Rated thermal power (MWt)
    max_coolant_temp: float         # Maximum coolant outlet temperature (C)
    normal_pressure: float          # Normal operating pressure (MPa)


SAFETY_LIMITS: dict[str, SafetyLimits] = {
    "rbmk": SafetyLimits(
        clad_temp_limit=700.0,
        containment_pressure_limit=0.45,
        rated_power=3200.0,
        max_coolant_temp=284.0,
        normal_pressure=7.0,
    ),
    "pwr": SafetyLimits(
        clad_temp_limit=1204.0,
        containment_pressure_limit=0.41,
        rated_power=2772.0,
        max_coolant_temp=327.0,
        normal_pressure=15.5,
    ),
    "bwr": SafetyLimits(
        clad_temp_limit=1204.0,
        containment_pressure_limit=0.53,
        rated_power=2381.0,
        max_coolant_temp=286.0,
        normal_pressure=7.0,
    ),
    "windscale": SafetyLimits(
        clad_temp_limit=400.0,
        containment_pressure_limit=0.0,
        rated_power=180.0,
        max_coolant_temp=350.0,
        normal_pressure=0.101,
    ),
}


class RewardCalculator:
    """Calculates rewards based on reactor state and safety criteria."""

    def __init__(self, reactor_type: str, target_outcome: str) -> None:
        self.reactor_type = reactor_type
        self.target_outcome = target_outcome
        self.limits = SAFETY_LIMITS[reactor_type]
        self.stability_counter: int = 0
        self.stability_threshold: int = 6  # Consecutive stable steps for success

    def step_reward(
        self,
        state: ReactorState,
        prev_state: ReactorState,
    ) -> float:
        """Calculate dense per-step reward.

        Returns reward in range [-1.0, 1.0].
        """
        reward = 0.0

        # 1. Core integrity (weight: 0.30)
        if self.limits.clad_temp_limit > 0:
            clad_margin = max(0.0, self.limits.clad_temp_limit - state.cladding_temp_c)
            clad_score = min(1.0, clad_margin / self.limits.clad_temp_limit)
        else:
            clad_score = 1.0 if state.cladding_temp_c < 400 else 0.0
        reward += 0.30 * clad_score

        # 2. Radiation containment (weight: 0.30)
        if state.environmental_release_tbq < 0.001:
            rad_score = 1.0
        else:
            rad_score = max(0.0, 1.0 - state.environmental_release_tbq / 100.0)
        reward += 0.30 * rad_score

        # 3. Containment integrity (weight: 0.20)
        if self.limits.containment_pressure_limit > 0:
            press_margin = max(
                0.0,
                self.limits.containment_pressure_limit - state.containment_pressure_mpa,
            )
            press_score = min(1.0, press_margin / self.limits.containment_pressure_limit)
        else:
            # No containment (Windscale): penalize any release
            press_score = max(0.0, 1.0 - state.environmental_release_tbq / 10.0)
        reward += 0.20 * press_score

        # 4. Fuel integrity (weight: 0.15)
        fuel_score = 1.0 - state.fuel_damage_fraction
        reward += 0.15 * fuel_score

        # 5. Hydrogen safety (weight: 0.05)
        if state.containment_hydrogen_pct < 4.0:
            h2_score = 1.0
        elif state.containment_hydrogen_pct < 13.0:
            h2_score = max(0.0, 1.0 - (state.containment_hydrogen_pct - 4.0) / 9.0)
        else:
            h2_score = 0.0
        reward += 0.05 * h2_score

        # === Penalties ===

        # Exceeding cladding temperature limit
        if self.limits.clad_temp_limit > 0 and state.cladding_temp_c > self.limits.clad_temp_limit:
            excess = state.cladding_temp_c - self.limits.clad_temp_limit
            reward -= 0.3 * min(1.0, excess / 500.0)

        # Fuel damage worsening
        damage_delta = state.fuel_damage_fraction - prev_state.fuel_damage_fraction
        if damage_delta > 0:
            reward -= 0.5 * damage_delta

        # Unnecessary scram in normal operation scenarios
        if self.target_outcome == "maintain_power":
            if state.thermal_power_mw < 0.05 * self.limits.rated_power:
                if prev_state.thermal_power_mw > 0.5 * self.limits.rated_power:
                    reward -= 0.05

        return max(-1.0, min(1.0, reward))

    def terminal_reward(self, state: ReactorState) -> float:
        """Calculate terminal reward at episode end."""
        if state.fuel_damage_fraction >= 0.95:
            return -1.0  # Complete meltdown

        if state.environmental_release_tbq > 50.0:
            return -0.8  # Major radiation release

        if state.containment_hydrogen_pct > 13.0:
            return -0.5  # Hydrogen detonation risk

        if self.target_outcome == "stabilize":
            if (
                state.fuel_damage_fraction < 0.1
                and state.cladding_temp_c < self.limits.clad_temp_limit
                and state.environmental_release_tbq < 1.0
            ):
                return 1.0  # Full success
            elif state.fuel_damage_fraction < 0.3:
                return 0.5  # Partial success
            elif state.fuel_damage_fraction < 0.5:
                return 0.2  # Marginal
            else:
                return -0.3  # Severe damage

        elif self.target_outcome == "maintain_power":
            power_fraction = state.thermal_power_mw / self.limits.rated_power
            if 0.9 <= power_fraction <= 1.1 and state.fuel_damage_fraction < 0.01:
                return 1.0  # Maintained power successfully
            elif state.fuel_damage_fraction < 0.01:
                return 0.5  # No damage but power not maintained
            else:
                return -0.3

        return 0.0

    def is_terminal(
        self,
        state: ReactorState,
        step_count: int,
        max_steps: int,
    ) -> tuple[bool, str]:
        """Check if episode should end.

        Returns (is_terminal, reason).
        """
        if state.fuel_damage_fraction >= 0.95:
            return True, "core_meltdown"

        if state.containment_hydrogen_pct > 18.0:
            return True, "hydrogen_detonation"

        if state.environmental_release_tbq > 200.0:
            return True, "catastrophic_release"

        if step_count >= max_steps:
            return True, "max_steps_reached"

        # Check for stability (success condition for stabilize scenarios)
        if self.target_outcome == "stabilize":
            is_stable = (
                state.cladding_temp_c < self.limits.clad_temp_limit
                and state.fuel_damage_fraction < 0.1
                and state.containment_hydrogen_pct < 4.0
                and (
                    self.limits.containment_pressure_limit <= 0
                    or state.containment_pressure_mpa < self.limits.containment_pressure_limit
                )
            )
            if is_stable:
                self.stability_counter += 1
                if self.stability_counter >= self.stability_threshold:
                    return True, "stabilized"
            else:
                self.stability_counter = 0

        return False, ""

    def get_reward_breakdown(self, state: ReactorState, prev_state: ReactorState) -> dict[str, float]:
        """Return detailed reward breakdown for diagnostics."""
        if self.limits.clad_temp_limit > 0:
            clad_margin = max(0.0, self.limits.clad_temp_limit - state.cladding_temp_c)
            clad_score = min(1.0, clad_margin / self.limits.clad_temp_limit)
        else:
            clad_score = 1.0 if state.cladding_temp_c < 400 else 0.0

        if state.environmental_release_tbq < 0.001:
            rad_score = 1.0
        else:
            rad_score = max(0.0, 1.0 - state.environmental_release_tbq / 100.0)

        if self.limits.containment_pressure_limit > 0:
            press_margin = max(0.0, self.limits.containment_pressure_limit - state.containment_pressure_mpa)
            press_score = min(1.0, press_margin / self.limits.containment_pressure_limit)
        else:
            press_score = max(0.0, 1.0 - state.environmental_release_tbq / 10.0)

        fuel_score = 1.0 - state.fuel_damage_fraction

        if state.containment_hydrogen_pct < 4.0:
            h2_score = 1.0
        elif state.containment_hydrogen_pct < 13.0:
            h2_score = max(0.0, 1.0 - (state.containment_hydrogen_pct - 4.0) / 9.0)
        else:
            h2_score = 0.0

        return {
            "core_integrity": round(clad_score, 3),
            "radiation_containment": round(rad_score, 3),
            "containment_integrity": round(press_score, 3),
            "fuel_integrity": round(fuel_score, 3),
            "hydrogen_safety": round(h2_score, 3),
            "total_step_reward": round(self.step_reward(state, prev_state), 3),
        }
