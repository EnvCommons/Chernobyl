"""Scenario definitions with historically accurate initial conditions.

Each scenario represents a specific moment during a real nuclear disaster,
with reactor state, equipment conditions, and instrument failures set to
match historical records as closely as possible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Scenario:
    """Definition of a scenario with initial conditions."""
    id: str
    scenario: str
    reactor_type: str
    difficulty: str
    time_step_minutes: float
    max_steps: int
    target_outcome: str
    description: str
    initial_conditions: dict[str, Any]


# =============================================================================
# CHERNOBYL SCENARIOS (RBMK-1000)
# =============================================================================

CHERNOBYL_XENON_PIT = Scenario(
    id="chernobyl_xenon_pit",
    scenario="chernobyl_xenon_pit",
    reactor_type="rbmk",
    difficulty="hard",
    time_step_minutes=1.0,
    max_steps=300,
    target_outcome="stabilize",
    description=(
        "April 26, 1986, 00:28 — Power has collapsed to 30 MWt from 1600 MWt due to "
        "operator error during transfer to automatic control. Xenon-135 is building rapidly "
        "from the massive iodine-135 inventory accumulated during 9 hours at 50% power. "
        "Most control rods have been withdrawn to fight the xenon. ORM is critically low. "
        "The reactor is entering the 'iodine pit'. You must decide: attempt recovery "
        "(risking xenon-induced instability) or shut down safely and wait 24-48 hours "
        "for xenon to decay."
    ),
    initial_conditions={
        "thermal_power_mw": 30.0,
        "neutron_population": 30.0 / 3200.0,
        # Core excess reactivity from fuel loading, balanced against rods + xenon
        # + temperature feedback to achieve the desired initial reactivity.
        # Ref: Duderstadt & Hamilton, Nuclear Reactor Analysis (1976), Ch. 8.
        "core_excess_reactivity": 0.055,
        "reactivity_total": -0.001,
        "reactivity_rods": 0.015,
        "reactivity_xenon": -0.035,
        "reactivity_void": 0.0,
        "reactivity_doppler": 0.012,
        "reactivity_moderator_temp": 0.007,
        "xenon_concentration": 2.5,
        "iodine_concentration": 1.8,
        "fuel_temp_c": 350.0,
        "fuel_centerline_temp_c": 400.0,
        "cladding_temp_c": 290.0,
        "coolant_inlet_temp_c": 270.0,
        "coolant_outlet_temp_c": 280.0,
        "coolant_pressure_mpa": 7.0,
        "coolant_flow_rate_kg_s": 8000.0,
        "void_fraction": 0.02,
        "steam_quality": 0.0,
        "pressurizer_level_pct": 0.0,
        "pressurizer_pressure_mpa": 0.0,
        "decay_heat_fraction": 0.0,
        "time_since_shutdown_s": 0.0,
        "fuel_damage_fraction": 0.0,
        "cladding_oxidation_pct": 0.0,
        "hydrogen_generated_kg": 0.0,
        "containment_pressure_mpa": 0.1,
        "containment_temp_c": 30.0,
        "containment_hydrogen_pct": 0.0,
        "radiation_level_sv_hr": 0.0001,
        "environmental_release_tbq": 0.0,
        "control_rod_position_pct": 85.0,
        "orm_count": 8,
        "equipment": {
            "mcp_1": {"status": "running", "speed_pct": 100},
            "mcp_2": {"status": "running", "speed_pct": 100},
            "mcp_3": {"status": "running", "speed_pct": 100},
            "mcp_4": {"status": "running", "speed_pct": 100},
            "mcp_5": {"status": "running", "speed_pct": 40},
            "mcp_6": {"status": "running", "speed_pct": 40},
            "mcp_7": {"status": "standby"},
            "mcp_8": {"status": "standby"},
            "turbine_1": {"status": "running"},
            "turbine_2": {"status": "tripped"},
            "diesel_1": {"status": "standby"},
            "diesel_2": {"status": "standby"},
            "eccs": {"status": "disabled"},
        },
        "instrument_failures": [],
    },
)

CHERNOBYL_TEST_START = Scenario(
    id="chernobyl_test_start",
    scenario="chernobyl_test_start",
    reactor_type="rbmk",
    difficulty="expert",
    time_step_minutes=0.25,
    max_steps=200,
    target_outcome="stabilize",
    description=(
        "April 26, 1986, 01:23:04 — The turbine test is about to begin. Reactor at "
        "200 MWt (far below the 700-1000 MWt test requirement). ORM is only 6 equivalent "
        "rods (mandatory shutdown threshold is 15, authorization required below 26). ECCS "
        "disabled. All 8 MCPs running. Multiple automatic safety systems have been disabled. "
        "When the turbine trips, coolant flow will decrease, voids will form, and the "
        "positive void coefficient will drive power up. WARNING: Pressing AZ-5 (scram) "
        "with rods this far withdrawn will trigger the graphite-tip positive reactivity "
        "insertion — the very mechanism that destroyed the real reactor."
    ),
    initial_conditions={
        "thermal_power_mw": 200.0,
        "neutron_population": 200.0 / 3200.0,
        "core_excess_reactivity": 0.062,
        "reactivity_total": 0.0,
        "reactivity_rods": 0.035,
        "reactivity_xenon": -0.032,
        "reactivity_void": -0.003,
        "reactivity_doppler": 0.0,
        "reactivity_moderator_temp": 0.0,
        "xenon_concentration": 2.8,
        "iodine_concentration": 1.5,
        "fuel_temp_c": 500.0,
        "fuel_centerline_temp_c": 800.0,
        "cladding_temp_c": 300.0,
        "coolant_inlet_temp_c": 270.0,
        "coolant_outlet_temp_c": 284.0,
        "coolant_pressure_mpa": 6.8,
        "coolant_flow_rate_kg_s": 12000.0,
        "void_fraction": 0.05,
        "steam_quality": 0.01,
        "pressurizer_level_pct": 0.0,
        "pressurizer_pressure_mpa": 0.0,
        "decay_heat_fraction": 0.0,
        "time_since_shutdown_s": 0.0,
        "fuel_damage_fraction": 0.0,
        "cladding_oxidation_pct": 0.0,
        "hydrogen_generated_kg": 0.0,
        "containment_pressure_mpa": 0.1,
        "containment_temp_c": 30.0,
        "containment_hydrogen_pct": 0.0,
        "radiation_level_sv_hr": 0.0001,
        "environmental_release_tbq": 0.0,
        "control_rod_position_pct": 92.0,
        "orm_count": 6,
        "equipment": {
            "mcp_1": {"status": "running", "speed_pct": 100},
            "mcp_2": {"status": "running", "speed_pct": 100},
            "mcp_3": {"status": "running", "speed_pct": 100},
            "mcp_4": {"status": "running", "speed_pct": 100},
            "mcp_5": {"status": "running", "speed_pct": 80},
            "mcp_6": {"status": "running", "speed_pct": 80},
            "mcp_7": {"status": "running", "speed_pct": 80},
            "mcp_8": {"status": "running", "speed_pct": 80},
            "turbine_1": {"status": "running"},
            "turbine_2": {"status": "tripped"},
            "diesel_1": {"status": "standby"},
            "diesel_2": {"status": "standby"},
            "eccs": {"status": "disabled"},
        },
        "instrument_failures": [],
    },
)

CHERNOBYL_NORMAL_OPS = Scenario(
    id="chernobyl_normal_ops",
    scenario="chernobyl_normal_ops",
    reactor_type="rbmk",
    difficulty="normal",
    time_step_minutes=5.0,
    max_steps=200,
    target_outcome="maintain_power",
    description=(
        "RBMK-1000 at normal 3200 MWt operation. Maintain stable power output "
        "despite periodic perturbations (pump speed fluctuations, load following "
        "requests, instrument noise). This scenario tests basic reactor operation "
        "knowledge for the RBMK design."
    ),
    initial_conditions={
        "thermal_power_mw": 3200.0,
        "neutron_population": 1.0,
        # Core excess reactivity: balances rods (-0.075 at 50%), xenon (-0.025),
        # and feedback terms to maintain criticality at rated power.
        # Typical RBMK excess reactivity ~0.10 dk/k at mid-cycle.
        # Ref: INSAG-7 §5.2 — RBMK core excess reactivity management.
        "core_excess_reactivity": 0.100,
        "reactivity_total": 0.0,
        "reactivity_rods": -0.002,
        "reactivity_xenon": -0.025,
        "reactivity_void": 0.015,
        "reactivity_doppler": 0.012,
        "reactivity_moderator_temp": 0.0,
        "xenon_concentration": 1.0,
        "iodine_concentration": 1.0,
        "fuel_temp_c": 1200.0,
        "fuel_centerline_temp_c": 1800.0,
        "cladding_temp_c": 350.0,
        "coolant_inlet_temp_c": 270.0,
        "coolant_outlet_temp_c": 284.0,
        "coolant_pressure_mpa": 7.0,
        # RBMK-1000 normal operation: 8 MCPs × 1562.5 kg/s = 12500 kg/s
        # Ref: INSAG-7 §2.1 — 8 main coolant pumps in normal operation
        "coolant_flow_rate_kg_s": 12500.0,
        "void_fraction": 0.15,
        "steam_quality": 0.14,
        "pressurizer_level_pct": 0.0,
        "pressurizer_pressure_mpa": 0.0,
        "decay_heat_fraction": 0.0,
        "time_since_shutdown_s": 0.0,
        "fuel_damage_fraction": 0.0,
        "cladding_oxidation_pct": 0.0,
        "hydrogen_generated_kg": 0.0,
        "containment_pressure_mpa": 0.1,
        "containment_temp_c": 30.0,
        "containment_hydrogen_pct": 0.0,
        "radiation_level_sv_hr": 0.0001,
        "environmental_release_tbq": 0.0,
        "control_rod_position_pct": 50.0,
        "orm_count": 30,
        "equipment": {
            # Ref: INSAG-7 §2.1 — RBMK-1000 operates with 8 MCPs
            # (4 per turbogenerator), all running in normal operation.
            "mcp_1": {"status": "running", "speed_pct": 100},
            "mcp_2": {"status": "running", "speed_pct": 100},
            "mcp_3": {"status": "running", "speed_pct": 100},
            "mcp_4": {"status": "running", "speed_pct": 100},
            "mcp_5": {"status": "running", "speed_pct": 100},
            "mcp_6": {"status": "running", "speed_pct": 100},
            "mcp_7": {"status": "running", "speed_pct": 100},
            "mcp_8": {"status": "running", "speed_pct": 100},
            "turbine_1": {"status": "running"},
            "turbine_2": {"status": "running"},
            "diesel_1": {"status": "standby"},
            "diesel_2": {"status": "standby"},
            "eccs": {"status": "armed"},
        },
        "instrument_failures": [],
    },
)


# =============================================================================
# THREE MILE ISLAND SCENARIOS (PWR)
# =============================================================================

TMI_PORV_STUCK = Scenario(
    id="tmi_porv_stuck",
    scenario="tmi_porv_stuck",
    reactor_type="pwr",
    difficulty="hard",
    time_step_minutes=1.0,
    max_steps=200,
    target_outcome="stabilize",
    description=(
        "March 28, 1979, 04:01 — Turbine trip, reactor automatically scrams. "
        "Pressure spike caused PORV to open. PORV should have reclosed at 2205 psi "
        "but is STUCK OPEN. The control room indicator shows the PORV solenoid is "
        "de-energized (reading 'CLOSED') but the valve is physically stuck open. "
        "Coolant is streaming out through the stuck PORV at ~20 kg/s. HPI pumps "
        "are running but operators historically throttled them because the "
        "pressurizer level was RISING (misleadingly, due to void formation). "
        "You must diagnose the stuck PORV — look for indirect evidence like PORV "
        "tailpipe temperature, falling pressure despite 'closed' PORV — and close "
        "the block valve upstream."
    ),
    initial_conditions={
        "thermal_power_mw": 0.0,
        "neutron_population": 0.0,
        "core_excess_reactivity": 0.025,
        "reactivity_total": -0.10,
        "reactivity_rods": -0.10,
        "reactivity_xenon": -0.02,
        "reactivity_void": 0.0,
        "reactivity_doppler": 0.0,
        "reactivity_moderator_temp": 0.0,
        "xenon_concentration": 1.0,
        "iodine_concentration": 1.0,
        "fuel_temp_c": 600.0,
        "fuel_centerline_temp_c": 800.0,
        "cladding_temp_c": 350.0,
        "coolant_inlet_temp_c": 285.0,
        "coolant_outlet_temp_c": 315.0,
        # TMI-2 PORV opened at 2205 psi (15.2 MPa), pressure dropping.
        # Ref: NRC NUREG-0600, Kemeny Commission Report (1979).
        "coolant_pressure_mpa": 15.0,
        "coolant_flow_rate_kg_s": 0.0,
        "void_fraction": 0.0,
        "steam_quality": 0.0,
        "pressurizer_level_pct": 80.0,
        "pressurizer_level_apparent_pct": 85.0,
        "pressurizer_pressure_mpa": 15.0,
        "decay_heat_fraction": 0.06,
        "time_since_shutdown_s": 60.0,
        "fuel_damage_fraction": 0.0,
        "cladding_oxidation_pct": 0.0,
        "hydrogen_generated_kg": 0.0,
        "containment_pressure_mpa": 0.1,
        "containment_temp_c": 30.0,
        "containment_hydrogen_pct": 0.0,
        "radiation_level_sv_hr": 0.0001,
        "environmental_release_tbq": 0.0,
        "control_rod_position_pct": 0.0,
        "orm_count": 0,
        "equipment": {
            "rcp_1": {"status": "tripped"},
            "rcp_2": {"status": "tripped"},
            "rcp_3": {"status": "tripped"},
            "rcp_4": {"status": "tripped"},
            "porv": {"status": "stuck_open", "visible_indicator": "closed"},
            "block_valve": {"status": "running", "position_pct": 100.0},
            "hpi_1": {"status": "running"},
            "hpi_2": {"status": "running"},
            "diesel_1": {"status": "running"},
            "diesel_2": {"status": "running"},
            "eccs": {"status": "armed"},
        },
        "instrument_failures": [
            {"instrument": "porv_position", "failure": "reads_closed_when_open"},
        ],
    },
)

TMI_LOSS_OF_COOLANT = Scenario(
    id="tmi_loss_of_coolant",
    scenario="tmi_loss_of_coolant",
    reactor_type="pwr",
    difficulty="hard",
    time_step_minutes=2.0,
    max_steps=150,
    target_outcome="stabilize",
    description=(
        "March 28, 1979, ~05:00 — One hour into the accident. Pressure has dropped "
        "to 8 MPa (from normal 15.5 MPa). Operators have throttled HPI to ~20% "
        "because pressurizer level reads 95% ('going solid' — they fear damaging "
        "equipment). But the high level is an illusion caused by void formation. "
        "In reality, the primary system is losing coolant through the still-stuck "
        "PORV. Core is beginning to uncover. Cladding temps rising to 500°C. "
        "Core exit thermocouples are reading intermittently offscale."
    ),
    initial_conditions={
        "thermal_power_mw": 0.0,
        "neutron_population": 0.0,
        "core_excess_reactivity": 0.043,
        "reactivity_total": -0.10,
        "reactivity_rods": -0.10,
        "reactivity_xenon": -0.025,
        "reactivity_void": 0.0,
        "reactivity_doppler": 0.0,
        "reactivity_moderator_temp": 0.0,
        "xenon_concentration": 1.5,
        "iodine_concentration": 1.2,
        "fuel_temp_c": 700.0,
        "fuel_centerline_temp_c": 900.0,
        "cladding_temp_c": 500.0,
        "coolant_inlet_temp_c": 285.0,
        "coolant_outlet_temp_c": 300.0,
        "coolant_pressure_mpa": 8.0,
        "coolant_flow_rate_kg_s": 0.0,
        "void_fraction": 0.3,
        "steam_quality": 0.05,
        "pressurizer_level_pct": 70.0,
        "pressurizer_level_apparent_pct": 95.0,
        "pressurizer_pressure_mpa": 8.0,
        "decay_heat_fraction": 0.03,
        "time_since_shutdown_s": 3600.0,
        "fuel_damage_fraction": 0.0,
        "cladding_oxidation_pct": 0.0,
        "hydrogen_generated_kg": 0.0,
        "containment_pressure_mpa": 0.15,
        "containment_temp_c": 45.0,
        "containment_hydrogen_pct": 0.0,
        "radiation_level_sv_hr": 0.005,
        "environmental_release_tbq": 0.0,
        "control_rod_position_pct": 0.0,
        "orm_count": 0,
        "equipment": {
            "rcp_1": {"status": "tripped"},
            "rcp_2": {"status": "tripped"},
            "rcp_3": {"status": "tripped"},
            "rcp_4": {"status": "tripped"},
            "porv": {"status": "stuck_open", "visible_indicator": "closed"},
            "block_valve": {"status": "running", "position_pct": 100.0},
            "hpi_1": {"status": "throttled", "flow_pct": 20},
            "hpi_2": {"status": "throttled", "flow_pct": 20},
            "diesel_1": {"status": "running"},
            "diesel_2": {"status": "running"},
            "eccs": {"status": "armed"},
        },
        "instrument_failures": [
            {"instrument": "porv_position", "failure": "reads_closed_when_open"},
            {"instrument": "core_exit_thermocouples", "failure": "offscale_high_intermittent"},
        ],
    },
)

TMI_RECOVERY = Scenario(
    id="tmi_recovery",
    scenario="tmi_recovery",
    reactor_type="pwr",
    difficulty="expert",
    time_step_minutes=5.0,
    max_steps=200,
    target_outcome="stabilize",
    description=(
        "March 28, 1979, ~06:30 — Core is 40% uncovered. Cladding temperatures "
        "exceeding 1100°C. Zircaloy-water reaction generating hydrogen. The block "
        "valve has finally been closed (stopping the LOCA), but significant core "
        "damage has already occurred. Must restore cooling to prevent complete "
        "meltdown. HPI is available. RCPs are tripped but could potentially be "
        "restarted (risk of pump damage from steam). A hydrogen bubble has formed "
        "in the reactor vessel upper head."
    ),
    initial_conditions={
        "thermal_power_mw": 0.0,
        "neutron_population": 0.0,
        "core_excess_reactivity": 0.064,
        "reactivity_total": -0.10,
        "reactivity_rods": -0.10,
        "reactivity_xenon": -0.02,
        "reactivity_void": 0.0,
        "reactivity_doppler": 0.0,
        "reactivity_moderator_temp": 0.0,
        "xenon_concentration": 1.8,
        "iodine_concentration": 1.0,
        "fuel_temp_c": 1200.0,
        "fuel_centerline_temp_c": 1800.0,
        "cladding_temp_c": 1100.0,
        "coolant_inlet_temp_c": 280.0,
        "coolant_outlet_temp_c": 280.0,
        "coolant_pressure_mpa": 6.0,
        "coolant_flow_rate_kg_s": 0.0,
        "void_fraction": 0.6,
        "steam_quality": 0.2,
        "pressurizer_level_pct": 50.0,
        "pressurizer_level_apparent_pct": 100.0,
        "pressurizer_pressure_mpa": 6.0,
        "decay_heat_fraction": 0.015,
        "time_since_shutdown_s": 7200.0,
        "fuel_damage_fraction": 0.15,
        "cladding_oxidation_pct": 8.0,
        # TMI-2 total H2 was ~400-460 kg per OSTI forensic analysis
        # (Ref: OSTI 5799972). At ~06:30 (2.5h into accident), ~300 kg generated.
        "hydrogen_generated_kg": 300.0,
        "containment_pressure_mpa": 0.19,
        "containment_temp_c": 60.0,
        "containment_hydrogen_pct": 3.5,
        "radiation_level_sv_hr": 5.0,
        "environmental_release_tbq": 0.5,
        "control_rod_position_pct": 0.0,
        "orm_count": 0,
        "equipment": {
            "rcp_1": {"status": "tripped"},
            "rcp_2": {"status": "tripped"},
            "rcp_3": {"status": "tripped"},
            "rcp_4": {"status": "tripped"},
            "porv": {"status": "standby", "position_pct": 0.0},
            "block_valve": {"status": "standby", "position_pct": 0.0},
            "hpi_1": {"status": "standby"},
            "hpi_2": {"status": "standby"},
            "diesel_1": {"status": "running"},
            "diesel_2": {"status": "running"},
            "eccs": {"status": "armed"},
        },
        "instrument_failures": [
            {"instrument": "core_exit_thermocouples", "failure": "offscale_high"},
        ],
    },
)


# =============================================================================
# FUKUSHIMA SCENARIOS (BWR)
# =============================================================================

FUKUSHIMA_BLACKOUT = Scenario(
    id="fukushima_blackout",
    scenario="fukushima_blackout",
    reactor_type="bwr",
    difficulty="hard",
    time_step_minutes=5.0,
    max_steps=300,
    target_outcome="stabilize",
    description=(
        "March 11, 2011, 15:41 — The 13-15 meter tsunami has destroyed all emergency "
        "diesel generators at Fukushima Daiichi Unit 1. All AC power is lost (station "
        "blackout). DC batteries are intact but will deplete in approximately 8 hours. "
        "The RCIC (Reactor Core Isolation Cooling) system is running — it is steam-driven "
        "and does not require AC power. The reactor scrammed automatically during the "
        "earthquake 55 minutes ago. Decay heat is ~2% of rated power (~48 MW). You must "
        "manage cooling with diminishing resources. When batteries die, you lose "
        "instrumentation and SRV control."
    ),
    initial_conditions={
        "thermal_power_mw": 0.0,
        "neutron_population": 0.0,
        "core_excess_reactivity": 0.059,
        "reactivity_total": -0.10,
        "reactivity_rods": -0.10,
        "reactivity_xenon": -0.02,
        "reactivity_void": 0.0,
        "reactivity_doppler": 0.0,
        "reactivity_moderator_temp": 0.0,
        "xenon_concentration": 1.5,
        "iodine_concentration": 1.3,
        "fuel_temp_c": 400.0,
        "fuel_centerline_temp_c": 500.0,
        "cladding_temp_c": 300.0,
        "coolant_inlet_temp_c": 200.0,
        "coolant_outlet_temp_c": 250.0,
        "coolant_pressure_mpa": 7.0,
        "coolant_flow_rate_kg_s": 50.0,
        "void_fraction": 0.05,
        "steam_quality": 0.0,
        "pressurizer_level_pct": 0.0,
        "pressurizer_pressure_mpa": 0.0,
        "decay_heat_fraction": 0.02,
        "time_since_shutdown_s": 3300.0,
        "fuel_damage_fraction": 0.0,
        "cladding_oxidation_pct": 0.0,
        "hydrogen_generated_kg": 0.0,
        "containment_pressure_mpa": 0.12,
        "containment_temp_c": 40.0,
        "containment_hydrogen_pct": 0.0,
        "radiation_level_sv_hr": 0.0001,
        "environmental_release_tbq": 0.0,
        "control_rod_position_pct": 0.0,
        "orm_count": 0,
        "equipment": {
            "rcp_a": {"status": "tripped"},
            "rcp_b": {"status": "tripped"},
            "rcic": {"status": "running"},
            "hpci": {"status": "standby"},
            "eccs_lp": {"status": "unavailable"},
            "diesel_1": {"status": "destroyed"},
            "diesel_2": {"status": "destroyed"},
            "batteries": {"status": "running", "charge_pct": 95.0, "hours_remaining": 8.0},
            "srv_1": {"status": "standby", "position_pct": 0.0},
            "srv_2": {"status": "standby", "position_pct": 0.0},
            "srv_3": {"status": "standby", "position_pct": 0.0},
            "msiv": {"status": "standby", "position_pct": 0.0},
            "fire_truck": {"status": "unavailable"},
            "suppression_pool": {"status": "running", "temp_c": 40.0, "level_pct": 100.0},
        },
        "instrument_failures": [],
    },
)

FUKUSHIMA_RCIC_FAILURE = Scenario(
    id="fukushima_rcic_failure",
    scenario="fukushima_rcic_failure",
    reactor_type="bwr",
    difficulty="expert",
    time_step_minutes=10.0,
    max_steps=150,
    target_outcome="stabilize",
    description=(
        "March 12, 2011, ~06:00 — RCIC has failed after running for ~14 hours. "
        "Batteries are nearly depleted (10%, ~48 minutes remaining). Reactor vessel "
        "pressure is rising with no steam removal. No cooling is available. A fire "
        "truck has finally arrived and can inject water, but only at low pressure "
        "(~1 MPa). The reactor vessel is at 7.5 MPa. You must open SRVs to "
        "depressurize the reactor vessel so the fire truck can inject. But SRVs "
        "require DC power to operate their solenoids — and the batteries are almost "
        "dead. Every minute counts."
    ),
    initial_conditions={
        "thermal_power_mw": 0.0,
        "neutron_population": 0.0,
        "core_excess_reactivity": 0.054,
        "reactivity_total": -0.10,
        "reactivity_rods": -0.10,
        "reactivity_xenon": -0.015,
        "reactivity_void": 0.0,
        "reactivity_doppler": 0.0,
        "reactivity_moderator_temp": 0.0,
        "xenon_concentration": 1.2,
        "iodine_concentration": 0.8,
        "fuel_temp_c": 600.0,
        "fuel_centerline_temp_c": 800.0,
        "cladding_temp_c": 450.0,
        "coolant_inlet_temp_c": 200.0,
        "coolant_outlet_temp_c": 286.0,
        "coolant_pressure_mpa": 7.5,
        "coolant_flow_rate_kg_s": 0.0,
        "void_fraction": 0.15,
        "steam_quality": 0.05,
        "pressurizer_level_pct": 0.0,
        "pressurizer_pressure_mpa": 0.0,
        "decay_heat_fraction": 0.008,
        "time_since_shutdown_s": 50400.0,
        "fuel_damage_fraction": 0.0,
        "cladding_oxidation_pct": 0.0,
        "hydrogen_generated_kg": 0.0,
        "containment_pressure_mpa": 0.3,
        "containment_temp_c": 80.0,
        "containment_hydrogen_pct": 0.5,
        "radiation_level_sv_hr": 0.01,
        "environmental_release_tbq": 0.0,
        "control_rod_position_pct": 0.0,
        "orm_count": 0,
        "equipment": {
            "rcp_a": {"status": "tripped"},
            "rcp_b": {"status": "tripped"},
            "rcic": {"status": "failed"},
            "hpci": {"status": "failed"},
            "eccs_lp": {"status": "unavailable"},
            "diesel_1": {"status": "destroyed"},
            "diesel_2": {"status": "destroyed"},
            "batteries": {"status": "running", "charge_pct": 10.0, "hours_remaining": 0.8},
            "srv_1": {"status": "standby", "position_pct": 0.0},
            "srv_2": {"status": "standby", "position_pct": 0.0},
            "srv_3": {"status": "standby", "position_pct": 0.0},
            "fire_truck": {"status": "standby", "flow_kg_s": 5.0},
            "suppression_pool": {"status": "running", "temp_c": 95.0, "level_pct": 80.0},
        },
        "instrument_failures": [
            {"instrument": "reactor_water_level", "failure": "reads_high_intermittent"},
        ],
    },
)

FUKUSHIMA_HYDROGEN = Scenario(
    id="fukushima_hydrogen",
    scenario="fukushima_hydrogen",
    reactor_type="bwr",
    difficulty="expert",
    time_step_minutes=5.0,
    max_steps=200,
    target_outcome="stabilize",
    description=(
        "March 12, 2011, ~15:00 — Core damage is underway. Hydrogen concentration in "
        "the containment has reached 8% (above the 4% flammability limit). Containment "
        "pressure is 0.75 MPa (design is 0.53 MPa — already exceeded!). Fire truck is "
        "injecting 5 kg/s of water but it's not enough to keep up with decay heat. "
        "You must decide on containment venting: filtered venting through the wetwell "
        "(suppression pool) scrubs most fission products but the pool is at 100°C and "
        "losing effectiveness. Direct venting releases more radiation. NOT venting "
        "risks hydrogen detonation (at 13% concentration) or containment failure. "
        "Historically, the hydrogen leaked into the reactor building and exploded."
    ),
    initial_conditions={
        "thermal_power_mw": 0.0,
        "neutron_population": 0.0,
        "core_excess_reactivity": 0.060,
        "reactivity_total": -0.10,
        "reactivity_rods": -0.10,
        "reactivity_xenon": -0.01,
        "reactivity_void": 0.0,
        "reactivity_doppler": 0.0,
        "reactivity_moderator_temp": 0.0,
        "xenon_concentration": 0.8,
        "iodine_concentration": 0.5,
        "fuel_temp_c": 1500.0,
        "fuel_centerline_temp_c": 2200.0,
        "cladding_temp_c": 1300.0,
        "coolant_inlet_temp_c": 200.0,
        "coolant_outlet_temp_c": 200.0,
        "coolant_pressure_mpa": 0.8,
        "coolant_flow_rate_kg_s": 5.0,
        "void_fraction": 0.7,
        "steam_quality": 0.3,
        "pressurizer_level_pct": 0.0,
        "pressurizer_pressure_mpa": 0.0,
        "decay_heat_fraction": 0.005,
        "time_since_shutdown_s": 86400.0,
        "fuel_damage_fraction": 0.4,
        "cladding_oxidation_pct": 25.0,
        "hydrogen_generated_kg": 300.0,
        "containment_pressure_mpa": 0.75,
        "containment_temp_c": 150.0,
        "containment_hydrogen_pct": 8.0,
        "radiation_level_sv_hr": 50.0,
        "environmental_release_tbq": 5.0,
        "control_rod_position_pct": 0.0,
        "orm_count": 0,
        "equipment": {
            "rcp_a": {"status": "tripped"},
            "rcp_b": {"status": "tripped"},
            "rcic": {"status": "failed"},
            "hpci": {"status": "failed"},
            "eccs_lp": {"status": "unavailable"},
            "diesel_1": {"status": "destroyed"},
            "diesel_2": {"status": "destroyed"},
            "batteries": {"status": "depleted"},
            "srv_1": {"status": "running", "position_pct": 100.0},
            "srv_2": {"status": "running", "position_pct": 100.0},
            "fire_truck": {"status": "running", "flow_kg_s": 5.0},
            "containment_vent": {"status": "standby", "position_pct": 0.0},
            "suppression_pool": {"status": "running", "temp_c": 100.0, "level_pct": 50.0},
        },
        "instrument_failures": [
            {"instrument": "reactor_water_level", "failure": "unreliable"},
            {"instrument": "containment_radiation", "failure": "offscale_high"},
        ],
    },
)


# =============================================================================
# WINDSCALE SCENARIOS (Air-Cooled Graphite Pile)
# =============================================================================

WINDSCALE_ANNEAL = Scenario(
    id="windscale_anneal",
    scenario="windscale_anneal",
    reactor_type="windscale",
    difficulty="hard",
    time_step_minutes=5.0,
    max_steps=200,
    target_outcome="stabilize",
    description=(
        "October 10, 1957 — Ninth Wigner energy release anneal on Windscale "
        "Pile 1 (Ref: Penney Report, 1957; Windscale fire Wikipedia). The first nuclear "
        "heating during this anneal did not produce the expected temperature rise in the "
        "graphite. A second heating has been applied. Graphite temperature "
        "is rising, but thermocouple coverage is sparse — zone 3 reads suspiciously low. "
        "The pile has accumulated significant Wigner energy after the longest irradiation "
        "period between anneals. Stored energy is estimated at 2500 J/kg. You must "
        "manage the anneal: too little heating fails to release the energy safely, "
        "too much heating risks fuel cartridge failure and fire. Magnox cladding melts "
        "at 650°C, uranium ignites in air at ~300°C."
    ),
    initial_conditions={
        "thermal_power_mw": 0.0,
        "neutron_population": 0.05,
        "core_excess_reactivity": 0.012,
        "reactivity_total": -0.005,
        "reactivity_rods": -0.005,
        "reactivity_xenon": -0.005,
        "reactivity_void": 0.0,
        "reactivity_doppler": 0.0,
        "reactivity_moderator_temp": 0.005,
        "xenon_concentration": 0.5,
        "iodine_concentration": 0.3,
        "fuel_temp_c": 300.0,
        "fuel_centerline_temp_c": 320.0,
        "cladding_temp_c": 290.0,
        "coolant_inlet_temp_c": 20.0,
        "coolant_outlet_temp_c": 250.0,
        "coolant_pressure_mpa": 0.101,
        "coolant_flow_rate_kg_s": 200.0,
        "void_fraction": 0.0,
        "steam_quality": 0.0,
        "pressurizer_level_pct": 0.0,
        "pressurizer_pressure_mpa": 0.0,
        "decay_heat_fraction": 0.0,
        "time_since_shutdown_s": 0.0,
        "fuel_damage_fraction": 0.0,
        "cladding_oxidation_pct": 0.0,
        "hydrogen_generated_kg": 0.0,
        "containment_pressure_mpa": 0.101,
        "containment_temp_c": 20.0,
        "containment_hydrogen_pct": 0.0,
        "radiation_level_sv_hr": 0.001,
        "environmental_release_tbq": 0.0,
        "control_rod_position_pct": 80.0,
        "orm_count": 0,
        "graphite_temp_c": 300.0,
        "wigner_energy_stored_j_per_kg": 2500.0,
        "equipment": {
            "blower_1": {"status": "running", "speed_pct": 100},
            "blower_2": {"status": "running", "speed_pct": 100},
            "burst_cartridge_detector": {"status": "running"},
        },
        "instrument_failures": [
            {"instrument": "graphite_thermocouple_zone_3", "failure": "reads_low"},
        ],
    },
)

WINDSCALE_FIRE = Scenario(
    id="windscale_fire",
    scenario="windscale_fire",
    reactor_type="windscale",
    difficulty="expert",
    time_step_minutes=2.0,
    max_steps=250,
    target_outcome="stabilize",
    description=(
        "October 10, 1957, 16:30 — Fire has been visually confirmed in Pile 1 through "
        "inspection holes. Fuel cartridges are cherry-red and burning. Approximately "
        "150 fuel channels are involved. The fire is being fed by the air flowing "
        "through the pile for cooling. The burst cartridge detector (BCD) is in alarm. "
        "Stack monitors show radioactivity is being released. You face an agonizing "
        "dilemma: (1) Increase blower speed to cool the graphite — but this supplies "
        "more oxygen to the fire. (2) Stop the blowers — removes oxygen but also "
        "removes all cooling. (3) Inject water — highly effective at cooling but risks "
        "hydrogen explosion from water-on-hot-graphite reaction and steam explosion. "
        "Historically, water was used and it worked, but the decision was described as "
        "'the bravest act in the history of nuclear power.'"
    ),
    initial_conditions={
        "thermal_power_mw": 0.0,
        "neutron_population": 0.0,
        "core_excess_reactivity": 0.044,
        "reactivity_total": -0.05,
        "reactivity_rods": -0.05,
        "reactivity_xenon": 0.0,
        "reactivity_void": 0.0,
        "reactivity_doppler": 0.0,
        "reactivity_moderator_temp": 0.0,
        "xenon_concentration": 0.1,
        "iodine_concentration": 0.1,
        "fuel_temp_c": 1200.0,
        "fuel_centerline_temp_c": 1300.0,
        "cladding_temp_c": 1200.0,
        "coolant_inlet_temp_c": 20.0,
        "coolant_outlet_temp_c": 400.0,
        "coolant_pressure_mpa": 0.101,
        "coolant_flow_rate_kg_s": 200.0,
        "void_fraction": 0.0,
        "steam_quality": 0.0,
        "pressurizer_level_pct": 0.0,
        "pressurizer_pressure_mpa": 0.0,
        "decay_heat_fraction": 0.0,
        "time_since_shutdown_s": 86400.0,
        "fuel_damage_fraction": 0.15,
        "cladding_oxidation_pct": 40.0,
        "hydrogen_generated_kg": 0.0,
        "containment_pressure_mpa": 0.101,
        "containment_temp_c": 100.0,
        "containment_hydrogen_pct": 0.0,
        "radiation_level_sv_hr": 10.0,
        "environmental_release_tbq": 2.0,
        "control_rod_position_pct": 0.0,
        "orm_count": 0,
        "graphite_temp_c": 800.0,
        "wigner_energy_stored_j_per_kg": 500.0,
        "equipment": {
            "blower_1": {"status": "running", "speed_pct": 50},
            "blower_2": {"status": "standby"},
            "burst_cartridge_detector": {"status": "running"},
            "water_injection": {"status": "standby", "flow_kg_s": 50.0},
        },
        "instrument_failures": [
            {"instrument": "graphite_thermocouple_zone_3", "failure": "destroyed"},
        ],
    },
)


# =============================================================================
# SCENARIO REGISTRY
# =============================================================================

ALL_SCENARIOS: dict[str, Scenario] = {
    "chernobyl_xenon_pit": CHERNOBYL_XENON_PIT,
    "chernobyl_test_start": CHERNOBYL_TEST_START,
    "chernobyl_normal_ops": CHERNOBYL_NORMAL_OPS,
    "tmi_porv_stuck": TMI_PORV_STUCK,
    "tmi_loss_of_coolant": TMI_LOSS_OF_COOLANT,
    "tmi_recovery": TMI_RECOVERY,
    "fukushima_blackout": FUKUSHIMA_BLACKOUT,
    "fukushima_rcic_failure": FUKUSHIMA_RCIC_FAILURE,
    "fukushima_hydrogen": FUKUSHIMA_HYDROGEN,
    "windscale_anneal": WINDSCALE_ANNEAL,
    "windscale_fire": WINDSCALE_FIRE,
}

TRAIN_SCENARIOS = [
    "chernobyl_normal_ops",
    "tmi_porv_stuck",
    "fukushima_blackout",
    "windscale_anneal",
]

TEST_SCENARIOS = list(ALL_SCENARIOS.keys())

SEEDS_PER_SCENARIO_TRAIN = 10
SEEDS_PER_SCENARIO_TEST = 10


class ScenarioRegistry:
    """Registry for looking up scenarios and generating task lists."""

    @staticmethod
    def get(scenario_name: str) -> Scenario:
        if scenario_name not in ALL_SCENARIOS:
            raise ValueError(
                f"Unknown scenario: {scenario_name}. "
                f"Available: {list(ALL_SCENARIOS.keys())}"
            )
        return ALL_SCENARIOS[scenario_name]

    @staticmethod
    def list_tasks(split: str) -> list[dict]:
        """Generate task list for a split."""
        tasks = []
        if split == "train":
            scenario_names = TRAIN_SCENARIOS
            seeds = SEEDS_PER_SCENARIO_TRAIN
        elif split == "test":
            scenario_names = TEST_SCENARIOS
            seeds = SEEDS_PER_SCENARIO_TEST
        else:
            raise ValueError(f"Unknown split: {split}. Available: ['train', 'test']")

        for scenario_name in scenario_names:
            scenario = ALL_SCENARIOS[scenario_name]
            for seed in range(seeds):
                task_id = f"{scenario_name}_{seed:03d}"
                tasks.append({
                    "id": task_id,
                    "scenario": scenario_name,
                    "reactor_type": scenario.reactor_type,
                    "difficulty": scenario.difficulty,
                    "initial_conditions": scenario.initial_conditions,
                    "time_step_minutes": scenario.time_step_minutes,
                    "max_steps": scenario.max_steps,
                    "target_outcome": scenario.target_outcome,
                    "seed": seed,
                })
        return tasks

    @staticmethod
    def list_splits() -> list[str]:
        return ["train", "test"]
