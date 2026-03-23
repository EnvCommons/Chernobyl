"""Equipment state machines for nuclear power plant components.

Models pumps, valves, generators, batteries, and instruments with
realistic failure modes and state transitions.
"""

from __future__ import annotations

import math
import random as _random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EquipmentStatus(str, Enum):
    RUNNING = "running"
    STANDBY = "standby"
    TRIPPED = "tripped"
    FAILED = "failed"
    DESTROYED = "destroyed"
    UNAVAILABLE = "unavailable"
    DEPLETED = "depleted"
    STUCK_OPEN = "stuck_open"
    STUCK_CLOSED = "stuck_closed"
    THROTTLED = "throttled"
    ALARM = "alarm"
    ARMED = "armed"
    DISABLED = "disabled"


# States that cannot be operated
INOPERABLE_STATES = {
    EquipmentStatus.FAILED,
    EquipmentStatus.DESTROYED,
    EquipmentStatus.DEPLETED,
}

# States that cannot transition back to running without repair
PERMANENT_FAILURE_STATES = {
    EquipmentStatus.DESTROYED,
    EquipmentStatus.DEPLETED,
}


class Equipment:
    """Base class for reactor equipment."""

    def __init__(self, eq_id: str, config: dict[str, Any]) -> None:
        self.id = eq_id
        self.status = EquipmentStatus(config.get("status", "standby"))
        self.speed_pct: float = config.get("speed_pct", 0.0 if self.status != EquipmentStatus.RUNNING else 100.0)
        self.flow_pct: float = config.get("flow_pct", self.speed_pct)
        self.flow_kg_s: float = config.get("flow_kg_s", 0.0)
        # What the control room indicator shows (may differ from reality)
        self.visible_indicator: str | None = config.get("visible_indicator", None)
        self.temp_c: float = config.get("temp_c", 0.0)
        self.level_pct: float = config.get("level_pct", 0.0)

    def can_operate(self) -> bool:
        return self.status not in INOPERABLE_STATES

    def can_restart(self) -> bool:
        return self.status not in PERMANENT_FAILURE_STATES

    def get_apparent_status(self) -> str:
        """What the instrument panel shows (may differ from reality)."""
        if self.visible_indicator is not None:
            return self.visible_indicator
        return self.status.value

    def start(self) -> tuple[bool, str]:
        if not self.can_restart():
            return False, f"{self.id} is {self.status.value} and cannot be restarted"
        if self.status == EquipmentStatus.RUNNING:
            return True, f"{self.id} is already running"
        self.status = EquipmentStatus.RUNNING
        self.speed_pct = 100.0
        return True, f"{self.id} started"

    def stop(self) -> tuple[bool, str]:
        if self.status in PERMANENT_FAILURE_STATES:
            return False, f"{self.id} is {self.status.value}"
        self.status = EquipmentStatus.STANDBY
        self.speed_pct = 0.0
        return True, f"{self.id} stopped"

    def set_speed(self, speed_pct: float) -> tuple[bool, str]:
        if not self.can_operate():
            return False, f"{self.id} is {self.status.value} and cannot be adjusted"
        self.speed_pct = max(0.0, min(100.0, speed_pct))
        if self.speed_pct > 0:
            self.status = EquipmentStatus.RUNNING
        else:
            self.status = EquipmentStatus.STANDBY
        return True, f"{self.id} speed set to {self.speed_pct:.1f}%"

    def update(self, dt: float, state: Any = None) -> None:
        """Update equipment state based on time passage."""
        pass


class Valve(Equipment):
    """Valve with open/close/stuck behavior."""

    def __init__(self, eq_id: str, config: dict[str, Any]) -> None:
        super().__init__(eq_id, config)
        self.position_pct: float = config.get("position_pct", 0.0)
        if self.status == EquipmentStatus.STUCK_OPEN:
            self.position_pct = 100.0
        elif self.status == EquipmentStatus.STUCK_CLOSED:
            self.position_pct = 0.0
        elif self.status == EquipmentStatus.RUNNING:
            # "running" for a valve means "open"
            self.position_pct = config.get("position_pct", 100.0)

    def open(self) -> tuple[bool, str]:
        if self.status == EquipmentStatus.STUCK_CLOSED:
            return False, f"{self.id} is stuck closed"
        if self.status in PERMANENT_FAILURE_STATES:
            return False, f"{self.id} is {self.status.value}"
        self.position_pct = 100.0
        self.status = EquipmentStatus.RUNNING
        return True, f"{self.id} opened"

    def close(self) -> tuple[bool, str]:
        if self.status == EquipmentStatus.STUCK_OPEN:
            return False, f"{self.id} is stuck open and cannot be closed directly"
        if self.status in PERMANENT_FAILURE_STATES:
            return False, f"{self.id} is {self.status.value}"
        self.position_pct = 0.0
        self.status = EquipmentStatus.STANDBY
        return True, f"{self.id} closed"

    def set_position(self, position_pct: float) -> tuple[bool, str]:
        if self.status in (EquipmentStatus.STUCK_OPEN, EquipmentStatus.STUCK_CLOSED):
            return False, f"{self.id} is {self.status.value}"
        if self.status in PERMANENT_FAILURE_STATES:
            return False, f"{self.id} is {self.status.value}"
        self.position_pct = max(0.0, min(100.0, position_pct))
        self.status = EquipmentStatus.RUNNING if self.position_pct > 0 else EquipmentStatus.STANDBY
        return True, f"{self.id} set to {self.position_pct:.1f}% open"

    def is_open(self) -> bool:
        return self.position_pct > 0.0

    def effective_flow_fraction(self) -> float:
        """Fraction of maximum flow through this valve (0-1)."""
        return self.position_pct / 100.0


class Battery(Equipment):
    """DC battery with charge depletion."""

    def __init__(self, eq_id: str, config: dict[str, Any]) -> None:
        super().__init__(eq_id, config)
        self.charge_pct: float = config.get("charge_pct", 100.0)
        self.hours_remaining: float = config.get("hours_remaining", 8.0)
        if self.hours_remaining > 0 and self.charge_pct > 0:
            self.drain_rate_per_s = self.charge_pct / (self.hours_remaining * 3600.0)
        else:
            self.drain_rate_per_s = 0.0

    def update(self, dt: float, state: Any = None) -> None:
        if self.status == EquipmentStatus.RUNNING and self.charge_pct > 0:
            self.charge_pct -= self.drain_rate_per_s * dt
            self.hours_remaining -= dt / 3600.0
            if self.charge_pct <= 0:
                self.charge_pct = 0.0
                self.hours_remaining = 0.0
                self.status = EquipmentStatus.DEPLETED

    def has_power(self) -> bool:
        return self.status == EquipmentStatus.RUNNING and self.charge_pct > 0


class SuppressionPool(Equipment):
    """BWR suppression pool (torus/wetwell)."""

    def __init__(self, eq_id: str, config: dict[str, Any]) -> None:
        super().__init__(eq_id, config)
        self.temp_c = config.get("temp_c", 30.0)
        self.level_pct = config.get("level_pct", 100.0)

    def update(self, dt: float, state: Any = None) -> None:
        # Pool heats up from steam condensation (simplified)
        pass


@dataclass
class InstrumentFailure:
    """Describes how an instrument deviates from truth."""
    instrument: str
    failure: str  # e.g., "reads_closed_when_open", "reads_low", "offscale_high", "destroyed", "unreliable"

    @classmethod
    def from_dict(cls, d: dict[str, str]) -> InstrumentFailure:
        return cls(instrument=d["instrument"], failure=d["failure"])


# Rated pump flow rates by pump type (kg/s)
PUMP_RATED_FLOWS: dict[str, float] = {
    # RBMK MCPs
    "mcp_1": 1562.5, "mcp_2": 1562.5, "mcp_3": 1562.5, "mcp_4": 1562.5,
    "mcp_5": 1562.5, "mcp_6": 1562.5, "mcp_7": 1562.5, "mcp_8": 1562.5,
    # PWR RCPs
    "rcp_1": 4800.0, "rcp_2": 4800.0, "rcp_3": 4800.0, "rcp_4": 4800.0,
    # BWR
    "rcp_a": 3400.0, "rcp_b": 3400.0,
    # Emergency/auxiliary
    "rcic": 50.0,
    "hpci": 100.0,
    "eccs_lp": 500.0,
    "hpi_1": 30.0, "hpi_2": 30.0,
    # Windscale blowers (kg/s of air)
    "blower_1": 200.0, "blower_2": 200.0,
}


class EquipmentManager:
    """Manages all equipment for a reactor scenario."""

    def __init__(self, reactor_type: str, equipment_config: dict[str, dict],
                 instrument_failures: list[dict] | None = None) -> None:
        self.reactor_type = reactor_type
        self.equipment: dict[str, Equipment] = {}
        self.instrument_failures: list[InstrumentFailure] = []
        self.containment_venting = False

        for eq_id, config in equipment_config.items():
            if "batteries" in eq_id or "battery" in eq_id:
                self.equipment[eq_id] = Battery(eq_id, config)
            elif "valve" in eq_id or "porv" in eq_id or "srv" in eq_id or "msiv" in eq_id:
                self.equipment[eq_id] = Valve(eq_id, config)
            elif "block_valve" in eq_id:
                self.equipment[eq_id] = Valve(eq_id, config)
            elif "containment_vent" in eq_id:
                self.equipment[eq_id] = Valve(eq_id, config)
            elif "suppression_pool" in eq_id:
                self.equipment[eq_id] = SuppressionPool(eq_id, config)
            else:
                self.equipment[eq_id] = Equipment(eq_id, config)

        if instrument_failures:
            self.instrument_failures = [
                InstrumentFailure.from_dict(f) for f in instrument_failures
            ]

    def update(self, dt: float, state: Any = None) -> None:
        """Update all equipment states."""
        for eq in self.equipment.values():
            eq.update(dt, state)

    def get(self, eq_id: str) -> Equipment | None:
        return self.equipment.get(eq_id)

    def get_effective_coolant_flow(self) -> float:
        """Calculate total effective coolant flow from all running pumps (kg/s)."""
        total_flow = 0.0
        for eq_id, eq in self.equipment.items():
            if eq.status == EquipmentStatus.RUNNING:
                rated = PUMP_RATED_FLOWS.get(eq_id, 0.0)
                if rated > 0:
                    total_flow += rated * (eq.speed_pct / 100.0)
                elif eq.flow_kg_s > 0:
                    total_flow += eq.flow_kg_s
            elif eq.status == EquipmentStatus.THROTTLED:
                rated = PUMP_RATED_FLOWS.get(eq_id, 0.0)
                if rated > 0:
                    total_flow += rated * (eq.flow_pct / 100.0)
        return total_flow

    def get_fire_truck_flow(self) -> float:
        """Get flow from fire truck injection (kg/s)."""
        ft = self.equipment.get("fire_truck")
        if ft and ft.status == EquipmentStatus.RUNNING:
            return ft.flow_kg_s if ft.flow_kg_s > 0 else 5.0
        return 0.0

    def has_ac_power(self) -> bool:
        """Check if AC power is available from any source."""
        for eq_id, eq in self.equipment.items():
            if "diesel" in eq_id or "turbine" in eq_id:
                if eq.status == EquipmentStatus.RUNNING:
                    return True
        return False

    def has_dc_power(self) -> bool:
        """Check if DC power is available."""
        for eq_id, eq in self.equipment.items():
            if isinstance(eq, Battery):
                if eq.has_power():
                    return True
        # AC power can also provide DC through chargers
        return self.has_ac_power()

    def get_instrument_reading(
        self, instrument: str, true_value: float, rng: _random.Random
    ) -> float | str:
        """Apply instrument failures to return what operator sees."""
        for failure in self.instrument_failures:
            if failure.instrument == instrument:
                match failure.failure:
                    case "reads_closed_when_open":
                        return "CLOSED"
                    case "reads_low":
                        return true_value * 0.6
                    case "reads_high":
                        return true_value * 1.5
                    case "reads_high_intermittent":
                        return true_value * 1.5 if rng.random() > 0.5 else true_value
                    case "offscale_high":
                        return "OFFSCALE HIGH"
                    case "offscale_high_intermittent":
                        return "OFFSCALE HIGH" if rng.random() > 0.3 else true_value
                    case "destroyed":
                        return "N/A"
                    case "unreliable":
                        return true_value * (0.5 + rng.random())
                    case _:
                        return true_value
        return true_value

    def get_porv_leak_rate(self, system_pressure_mpa: float) -> float:
        """Calculate coolant leak rate through stuck-open PORV (kg/s)."""
        porv = self.equipment.get("porv")
        if porv is None:
            return 0.0
        block = self.equipment.get("block_valve")
        # If block valve is closed, no flow regardless of PORV state
        if block and isinstance(block, Valve) and not block.is_open():
            return 0.0
        if isinstance(porv, Valve) and porv.is_open():
            # Simplified: critical flow through PORV
            # TMI PORV: ~20 kg/s at normal pressure
            p_ratio = max(0.0, system_pressure_mpa / 15.5)
            return 20.0 * math.sqrt(p_ratio)
        if porv.status == EquipmentStatus.STUCK_OPEN:
            p_ratio = max(0.0, system_pressure_mpa / 15.5)
            return 20.0 * math.sqrt(p_ratio)
        return 0.0

    def get_srv_flow(self, system_pressure_mpa: float) -> float:
        """Calculate steam flow through open SRVs (kg/s, for BWR depressurization)."""
        total = 0.0
        for eq_id, eq in self.equipment.items():
            if "srv" in eq_id and isinstance(eq, Valve) and eq.is_open():
                # Each SRV passes ~50 kg/s at rated pressure
                p_ratio = max(0.0, system_pressure_mpa / 7.0)
                total += 50.0 * math.sqrt(p_ratio)
        return total

    def describe_equipment(self) -> dict[str, dict[str, Any]]:
        """Return a description of all equipment suitable for display."""
        result = {}
        for eq_id, eq in self.equipment.items():
            info: dict[str, Any] = {
                "apparent_status": eq.get_apparent_status(),
            }
            if eq.status == EquipmentStatus.RUNNING and eq.speed_pct < 100:
                info["speed_pct"] = eq.speed_pct
            if isinstance(eq, Valve):
                info["position_pct"] = eq.position_pct
            if isinstance(eq, Battery):
                info["charge_pct"] = round(eq.charge_pct, 1)
                info["hours_remaining"] = round(eq.hours_remaining, 1)
            if isinstance(eq, SuppressionPool):
                info["temp_c"] = round(eq.temp_c, 1)
                info["level_pct"] = round(eq.level_pct, 1)
            result[eq_id] = info
        return result
