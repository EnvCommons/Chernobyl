"""Reactor simulation engine.

Couples all physics models (neutronics, thermal-hydraulics, xenon dynamics,
decay heat, fuel integrity, containment) into a unified simulation that
advances the complete reactor state at each timestep.
"""

from __future__ import annotations

import random as _random
from dataclasses import dataclass, field
from typing import Any

from physics import (
    REACTOR_PARAMS,
    ContainmentModel,
    DecayHeatModel,
    FuelIntegrityModel,
    NeutronicsModel,
    PressurizerModel,
    ReactorParams,
    ThermalHydraulicsModel,
    WignerEnergyModel,
    XenonDynamics,
)
from equipment import EquipmentManager, EquipmentStatus, Valve


@dataclass
class ReactorState:
    """Complete snapshot of reactor state at a point in time."""

    # Time
    time_seconds: float = 0.0

    # Neutronics
    thermal_power_mw: float = 0.0
    neutron_population: float = 0.0
    precursor_concentration: float = 0.0
    reactivity_total: float = 0.0
    reactivity_rods: float = 0.0
    reactivity_xenon: float = 0.0
    reactivity_void: float = 0.0
    reactivity_doppler: float = 0.0
    reactivity_moderator_temp: float = 0.0

    # Xenon dynamics
    xenon_concentration: float = 1.0   # Normalized: 1.0 = equilibrium at rated
    iodine_concentration: float = 1.0

    # Thermal-hydraulics
    fuel_temp_c: float = 400.0
    fuel_centerline_temp_c: float = 600.0
    cladding_temp_c: float = 300.0
    coolant_inlet_temp_c: float = 270.0
    coolant_outlet_temp_c: float = 284.0
    coolant_pressure_mpa: float = 7.0
    coolant_flow_rate_kg_s: float = 10000.0
    void_fraction: float = 0.0
    steam_quality: float = 0.0

    # Pressurizer (PWR)
    pressurizer_level_pct: float = 0.0
    pressurizer_level_apparent_pct: float = 0.0
    pressurizer_pressure_mpa: float = 0.0

    # Decay heat
    decay_heat_fraction: float = 0.0
    decay_heat_mw: float = 0.0
    time_since_shutdown_s: float = 0.0
    was_operating: bool = True

    # Fuel integrity
    fuel_damage_fraction: float = 0.0
    cladding_oxidation_pct: float = 0.0
    hydrogen_generated_kg: float = 0.0

    # Containment
    containment_pressure_mpa: float = 0.1
    containment_temp_c: float = 30.0
    containment_hydrogen_pct: float = 0.0
    radiation_level_sv_hr: float = 0.0001
    environmental_release_tbq: float = 0.0

    # Control rods
    control_rod_position_pct: float = 50.0
    orm_count: int = 30
    rods_inserting: bool = False
    scram_active: bool = False

    # Windscale-specific
    graphite_temp_c: float = 0.0
    wigner_energy_stored_j_per_kg: float = 0.0

    # Injection state
    injection_rate_kg_s: float = 0.0
    injection_boron_ppm: float = 0.0

    # Coolant inventory tracking (fraction of nominal, 1.0 = full)
    coolant_inventory_fraction: float = 1.0

    # Core excess reactivity from fuel loading (dk/k). This is the positive
    # reactivity inherent in the fuel that is compensated by control rods,
    # xenon, and temperature feedback to achieve criticality.
    # Ref: Duderstadt & Hamilton, Nuclear Reactor Analysis (1976), Ch. 8.
    core_excess_reactivity: float = 0.0

    # Manual rod reactivity offset: difference between full-worth and
    # manual-scaled rod reactivity. When the operator adjusts a small
    # rod group, only a fraction of the total worth changes. This offset
    # preserves the scaled reactivity across advance() calls.
    # Reset to 0.0 during scram (all rods insert with full worth).
    manual_rod_reactivity_offset: float = 0.0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ReactorState:
        """Create state from initial conditions dictionary."""
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known_fields}
        state = cls(**filtered)
        # Initialize precursor concentration at steady state
        if state.neutron_population > 0 and state.precursor_concentration == 0:
            # At steady state: C = (beta/Lambda) * n / lambda_d
            # Using generic values; will be overridden if needed
            state.precursor_concentration = state.neutron_population
        return state


class ReactorSimulation:
    """Core simulation engine coupling all physics models."""

    def __init__(
        self,
        reactor_type: str,
        initial_conditions: dict[str, Any],
        time_step_minutes: float,
        difficulty: str = "normal",
        seed: int = 42,
    ) -> None:
        self.reactor_type = reactor_type
        self.params = REACTOR_PARAMS[reactor_type]
        self.dt = time_step_minutes * 60.0  # seconds
        self.difficulty = difficulty
        self.rng = _random.Random(seed)

        # Initialize state
        eq_config = initial_conditions.get("equipment", {})
        instr_failures = initial_conditions.get("instrument_failures", [])

        # Remove non-state fields before creating ReactorState
        state_dict = {
            k: v for k, v in initial_conditions.items()
            if k not in ("equipment", "instrument_failures")
        }
        self.state = ReactorState.from_dict(state_dict)

        # Initialize precursor concentration for steady state
        if self.state.neutron_population > 0:
            beta = self.params.beta_eff
            lam = self.params.neutron_gen_time
            lam_d = self.params.lambda_d
            self.state.precursor_concentration = (
                (beta / lam) * self.state.neutron_population / lam_d
            )

        # Equipment
        self.equipment = EquipmentManager(reactor_type, eq_config, instr_failures)

        # Physics models
        self.neutronics = NeutronicsModel(self.params)
        self.thermal = ThermalHydraulicsModel(self.params)
        self.decay_heat_model = DecayHeatModel()
        self.xenon = XenonDynamics(self.params, reactor_type=reactor_type)
        self.fuel_integrity = FuelIntegrityModel(self.params)
        self.containment_model = ContainmentModel(self.params)
        self.wigner = WignerEnergyModel()
        self.pressurizer = PressurizerModel()

    def _coupled_power_update(
        self,
        dt: float,
        power_mw: float,
        neutron_pop: float,
        precursor_conc: float,
        base_reactivity: float,
    ) -> tuple[float, float, float]:
        """Advance power with implicit Doppler and void feedback coupling.

        Instead of using a fixed reactivity for the entire timestep, this
        method embeds fuel temperature and void fraction feedback into the
        neutronics sub-steps. As power changes, fuel temperature and void
        evolve (with thermal inertia), and Doppler + void feedback modify
        the effective reactivity.

        This prevents the oscillatory instability that occurs with explicit
        operator splitting when feedback timescales are shorter than the
        integration timestep.

        Ref: Quasi-static method — Ott & Neuhold, Nuclear Reactor Dynamics
        (1985); used in PARCS/TRACE coupling (NUREG/CR-6899).
        """
        beta = self.params.beta_eff
        lam = self.params.neutron_gen_time
        lam_d = self.params.lambda_d
        rated = self.params.rated_power_mw
        dopp_coeff = self.params.doppler_coeff
        fuel_temp_ref = self.params.fuel_temp_ref
        void_coeff = self.params.void_coeff
        void_ref = self.params.void_fraction_ref

        # Current fuel temperature (already updated by thermal model)
        current_fuel_temp = self.state.fuel_temp_c

        # Void coupling: for reactors with subcooled boiling (RBMK),
        # void fraction evolves with power within the sub-steps.
        has_subcooled_boiling = (
            void_ref > 0 and not self.params.has_pressurizer
        )
        void_fraction = self.state.void_fraction

        # Strip both Doppler and void from base_reactivity to get the
        # "fixed" components (rods, xenon, moderator temp, core excess).
        current_doppler = self.neutronics.doppler_feedback(current_fuel_temp)
        current_void_rho = void_coeff * (void_fraction - void_ref)
        rho_fixed = base_reactivity - current_doppler - current_void_rho

        # Sub-stepping for neutronics (based on system stiffness).
        # Fast eigenvalue is |rho-beta|/Lambda (prompt neutron timescale).
        # Ref: Ott & Neuhold, "Introductory Nuclear Reactor Dynamics" (1985).
        rho = base_reactivity
        stiff_eigenvalue = abs(rho - beta) / lam
        stiff_eigenvalue = max(stiff_eigenvalue, beta / lam)
        max_stable_dt = 2.5 / max(stiff_eigenvalue, 1.0)
        n_substeps = max(1, int(dt / max_stable_dt) + 1)

        if rho > 0.5 * beta:
            n_substeps = max(n_substeps, int(dt / 0.001))
        elif rho > 0.1 * beta:
            n_substeps = max(n_substeps, int(dt / 0.01))

        sub_dt = dt / n_substeps
        n = neutron_pop
        c = precursor_conc

        # Fuel thermal time constant and initial temperature
        tau_fuel = 8.0  # seconds (must match thermal model)
        fuel_temp = current_fuel_temp

        for _ in range(n_substeps):
            # Current power from neutron population
            current_power = n * rated

            # Update fuel temperature with thermal lag toward power-implied target.
            # Target fuel temp at current power: T_ref * max(P/P_rated, 0.05)
            power_frac = max(current_power / rated, 0.05)
            target_fuel = fuel_temp_ref * power_frac
            alpha_f = min(1.0, sub_dt / tau_fuel)
            fuel_temp = fuel_temp + alpha_f * (target_fuel - fuel_temp)

            # Compute Doppler from current fuel temp
            doppler_rho = dopp_coeff * (fuel_temp - fuel_temp_ref)

            # Update void fraction (subcooled boiling model for RBMK)
            if has_subcooled_boiling:
                power_frac_v = max(current_power / rated, 0.0)
                void_eq = void_ref * power_frac_v
                alpha_void = min(1.0, sub_dt / 5.0)  # tau_void = 5s
                void_fraction = void_fraction + alpha_void * (void_eq - void_fraction)
                void_fraction = max(0.0, min(0.95, void_fraction))

            # Compute void reactivity from current void fraction
            void_rho = void_coeff * (void_fraction - void_ref)

            # Effective reactivity for this sub-step
            rho = rho_fixed + doppler_rho + void_rho

            # Point kinetics RK4 step
            def dn(n_val: float, c_val: float) -> float:
                return ((rho - beta) / lam) * n_val + lam_d * c_val

            def dc(n_val: float, c_val: float) -> float:
                return (beta / lam) * n_val - lam_d * c_val

            k1n = dn(n, c)
            k1c = dc(n, c)
            k2n = dn(n + 0.5 * sub_dt * k1n, c + 0.5 * sub_dt * k1c)
            k2c = dc(n + 0.5 * sub_dt * k1n, c + 0.5 * sub_dt * k1c)
            k3n = dn(n + 0.5 * sub_dt * k2n, c + 0.5 * sub_dt * k2c)
            k3c = dc(n + 0.5 * sub_dt * k2n, c + 0.5 * sub_dt * k2c)
            k4n = dn(n + sub_dt * k3n, c + sub_dt * k3c)
            k4c = dc(n + sub_dt * k3n, c + sub_dt * k3c)

            n = n + (sub_dt / 6.0) * (k1n + 2 * k2n + 2 * k3n + k4n)
            c = c + (sub_dt / 6.0) * (k1c + 2 * k2c + 2 * k3c + k4c)

            n = max(0.0, n)
            c = max(0.0, c)

        new_power = n * rated

        # Update state to reflect the coupled evolution.
        # These override the thermal model's estimates with the coupled result.
        self.state.fuel_temp_c = fuel_temp
        self.state.fuel_centerline_temp_c = min(fuel_temp * 1.5, 2865.0)
        self.state.reactivity_doppler = dopp_coeff * (fuel_temp - fuel_temp_ref)
        self.state.void_fraction = void_fraction
        self.state.reactivity_void = void_coeff * (void_fraction - void_ref)

        return new_power, n, c

    def advance(self, dt_override: float | None = None) -> None:
        """Advance simulation by one timestep. All physics are coupled."""
        dt = dt_override if dt_override is not None else self.dt
        s = self.state

        # 0. Update equipment (battery drain, etc.)
        self.equipment.update(dt, s)

        # 1. Control rod movement (if inserting/scramming)
        # Sub-step rod movement to capture transient reactivity effects.
        # Critical for RBMK graphite-tipped rods where the first ~17% of
        # insertion ADDS positive reactivity from graphite displacers.
        # Ref: INSAG-7 §4.3 — positive scram effect from graphite tips
        # during initial rod insertion was a key factor in the Chernobyl
        # accident (AZ-5 button caused initial power surge).
        if s.rods_inserting or s.scram_active:
            # Scram / rod insertion: ALL rods insert with full worth.
            # Clear manual offset — this is no longer a small-group adjustment.
            s.manual_rod_reactivity_offset = 0.0

            rod_speed = self.params.rod_speed_pct_per_s
            total_movement = rod_speed * dt
            old_position = s.control_rod_position_pct
            inserting_from_withdrawn = old_position > 80.0

            # Use sub-steps to capture graphite tip transient (important for RBMK)
            n_rod_substeps = max(1, int(total_movement / 1.0))  # ~1% position per substep
            sub_movement = total_movement / n_rod_substeps

            # Time-averaged rod reactivity across sub-steps
            rho_sum = 0.0
            pos = old_position
            for _ in range(n_rod_substeps):
                pos = max(0.0, pos - sub_movement)
                rho_sub = self.neutronics.control_rod_reactivity(
                    pos,
                    inserting_from_withdrawn=inserting_from_withdrawn and self.params.has_graphite_tip,
                )
                rho_sum += rho_sub

            s.control_rod_position_pct = pos
            s.reactivity_rods = rho_sum / n_rod_substeps  # Time-averaged reactivity

            # Update ORM (RBMK-specific)
            if self.reactor_type == "rbmk":
                s.orm_count = int(
                    self.params.num_control_rods * (1.0 - s.control_rod_position_pct / 100.0)
                )

            if s.control_rod_position_pct <= 0.0:
                s.rods_inserting = False
                s.scram_active = False
        else:
            # Normal operation: rod reactivity from position plus any manual
            # offset from scaled rod group adjustments.
            base_rho = self.neutronics.control_rod_reactivity(
                s.control_rod_position_pct,
                inserting_from_withdrawn=False,
            )
            s.reactivity_rods = base_rho + s.manual_rod_reactivity_offset

        # 2. Calculate effective coolant flow
        effective_flow = self.equipment.get_effective_coolant_flow()
        # Add fire truck flow
        effective_flow += self.equipment.get_fire_truck_flow()
        # Add injection
        effective_flow += s.injection_rate_kg_s

        # 2a. Track coolant inventory loss from leaks (PORV, SRV, etc.)
        # Ref: TMI-2 lost ~32,000 gallons (~120,000 kg) in first 2h22m through
        # stuck-open PORV at ~20 kg/s (NUREG-0600). Loss of inventory degrades
        # core cooling even with pumps running.
        leak_rate_preview = self.equipment.get_porv_leak_rate(s.coolant_pressure_mpa)
        srv_flow_preview = self.equipment.get_srv_flow(s.coolant_pressure_mpa)
        net_loss_kg_s = leak_rate_preview + srv_flow_preview - s.injection_rate_kg_s
        if net_loss_kg_s > 0 and self.params.coolant_mass_kg > 0:
            mass_lost = net_loss_kg_s * dt
            s.coolant_inventory_fraction = max(
                0.0,
                s.coolant_inventory_fraction - mass_lost / self.params.coolant_mass_kg,
            )
        elif net_loss_kg_s < 0:
            # Injection exceeding losses: inventory recovering
            mass_gained = abs(net_loss_kg_s) * dt
            s.coolant_inventory_fraction = min(
                1.0,
                s.coolant_inventory_fraction + mass_gained / self.params.coolant_mass_kg,
            )

        # Degrade effective cooling when inventory drops below core coverage
        # (~50% inventory means core begins to uncover)
        if s.coolant_inventory_fraction < 0.5:
            inventory_factor = max(0.0, s.coolant_inventory_fraction / 0.5)
            effective_flow *= inventory_factor

        # 3. Xenon dynamics
        power_fraction = s.thermal_power_mw / self.params.rated_power_mw
        s.iodine_concentration, s.xenon_concentration = self.xenon.update(
            dt, power_fraction, s.iodine_concentration, s.xenon_concentration
        )
        s.reactivity_xenon = self.xenon.get_reactivity(s.xenon_concentration)

        # 4-8. Coupled thermal-neutronics advancement.
        #
        # The thermal state and neutron kinetics are tightly coupled through
        # reactivity feedback (Doppler, void, moderator temp). Computing them
        # sequentially over a large dt causes oscillatory instability: power
        # changes → temperature changes → feedback changes → power changes.
        #
        # We use sub-stepping with coupled feedback: divide the macro timestep
        # into N_couple sub-intervals and update both thermal and neutronics
        # within each sub-interval. This ensures negative feedback (Doppler)
        # can arrest power excursions within the timestep.
        #
        # Ref: "Operator splitting" / Strang splitting methods for coupled
        # multi-physics; standard approach in RELAP5, TRACE, and other NRC-
        # approved nuclear safety codes (NUREG/CR-5535).

        leak_rate = self.equipment.get_porv_leak_rate(s.coolant_pressure_mpa)
        srv_flow = self.equipment.get_srv_flow(s.coolant_pressure_mpa)

        decay_heat_mw = self.decay_heat_model.calculate(
            s.time_since_shutdown_s, self.params.rated_power_mw
        )
        s.decay_heat_mw = decay_heat_mw
        s.decay_heat_fraction = self.decay_heat_model.fraction(s.time_since_shutdown_s)

        # Boron injection negative reactivity
        boron_reactivity = 0.0
        if s.injection_boron_ppm > 0 and s.injection_rate_kg_s > 0:
            boron_reactivity = -s.injection_boron_ppm * 1e-5

        # 4. Thermal-hydraulics update (single pass for the full timestep).
        thermal_result = self.thermal.update(
            dt=dt,
            power_mw=s.thermal_power_mw,
            decay_heat_mw=decay_heat_mw,
            coolant_flow_kg_s=effective_flow,
            coolant_inlet_temp=s.coolant_inlet_temp_c,
            coolant_pressure=s.coolant_pressure_mpa,
            fuel_temp=s.fuel_temp_c,
            cladding_temp=s.cladding_temp_c,
            coolant_outlet_temp=s.coolant_outlet_temp_c,
            void_fraction=s.void_fraction,
            leak_rate_kg_s=leak_rate,
            injection_rate_kg_s=s.injection_rate_kg_s,
            injection_temp_c=30.0,
            srv_flow_kg_s=srv_flow,
            coolant_inventory_fraction=s.coolant_inventory_fraction,
        )

        s.fuel_temp_c = thermal_result.fuel_temp
        s.fuel_centerline_temp_c = thermal_result.fuel_centerline_temp
        s.cladding_temp_c = thermal_result.cladding_temp
        s.coolant_outlet_temp_c = thermal_result.coolant_outlet_temp
        s.void_fraction = thermal_result.void_fraction
        s.coolant_pressure_mpa = thermal_result.coolant_pressure

        # 6. Reactivity feedback from updated thermal state.
        s.reactivity_doppler = self.neutronics.doppler_feedback(s.fuel_temp_c)
        s.reactivity_void = self.neutronics.void_feedback(s.void_fraction)
        s.reactivity_moderator_temp = self.neutronics.moderator_temp_feedback(
            s.coolant_outlet_temp_c
        )

        # 7. Total reactivity
        s.reactivity_total = (
            s.core_excess_reactivity
            + s.reactivity_rods
            + s.reactivity_xenon
            + s.reactivity_doppler
            + s.reactivity_void
            + s.reactivity_moderator_temp
            + boron_reactivity
        )

        # 8. Power update using point kinetics.
        #
        # For reactors near criticality (|rho| < beta), the power evolves
        # on the slow timescale of delayed neutrons (inhour equation), not
        # the fast prompt timescale. We use the point kinetics equation with
        # an implicit Doppler correction: as the neutronics solver advances
        # power in sub-steps, we track the implied fuel temperature change
        # and adjust reactivity accordingly. This prevents the oscillatory
        # instability that occurs with explicit operator splitting at large
        # timesteps.
        #
        # Ref: "Quasi-static method" — Ott & Neuhold, Nuclear Reactor
        # Dynamics (1985); PARCS/TRACE coupling methodology (NUREG/CR-6899).
        if s.neutron_population > 1e-10 or s.reactivity_total > 0:
            new_power, new_n, new_c = self._coupled_power_update(
                dt, s.thermal_power_mw, s.neutron_population,
                s.precursor_concentration, s.reactivity_total,
            )
            s.thermal_power_mw = new_power
            s.neutron_population = new_n
            s.precursor_concentration = new_c
        else:
            s.thermal_power_mw = 0.0

        s.coolant_flow_rate_kg_s = effective_flow

        # 5. Pressurizer (PWR) — updated once per macro timestep
        if self.params.has_pressurizer:
            hpi_flow = 0.0
            for eq_id in ("hpi_1", "hpi_2"):
                eq = self.equipment.get(eq_id)
                if eq and eq.status == EquipmentStatus.RUNNING:
                    from equipment import PUMP_RATED_FLOWS
                    hpi_flow += PUMP_RATED_FLOWS.get(eq_id, 30.0)
                elif eq and eq.status == EquipmentStatus.THROTTLED:
                    from equipment import PUMP_RATED_FLOWS
                    hpi_flow += PUMP_RATED_FLOWS.get(eq_id, 30.0) * (eq.flow_pct / 100.0)

            actual_level, apparent_level = self.pressurizer.update(
                dt, s.pressurizer_level_pct, s.coolant_pressure_mpa,
                leak_rate, hpi_flow, s.coolant_outlet_temp_c,
            )
            s.pressurizer_level_pct = actual_level
            s.pressurizer_level_apparent_pct = apparent_level
            s.pressurizer_pressure_mpa = s.coolant_pressure_mpa

        # 9. Decay heat tracking
        operating_threshold = 0.01 * self.params.rated_power_mw
        if s.thermal_power_mw < operating_threshold:
            if s.was_operating:
                s.time_since_shutdown_s = 0.0
                s.was_operating = False
            s.time_since_shutdown_s += dt
        else:
            s.was_operating = True
            s.time_since_shutdown_s = 0.0

        # 10. Fuel integrity
        fuel_result = self.fuel_integrity.update(
            dt, s.cladding_temp_c, s.fuel_centerline_temp_c,
            s.fuel_damage_fraction, s.cladding_oxidation_pct,
            s.hydrogen_generated_kg,
        )
        s.fuel_damage_fraction = fuel_result.damage_fraction
        s.cladding_oxidation_pct = fuel_result.oxidation_pct
        s.hydrogen_generated_kg = fuel_result.hydrogen_kg

        # 11. Containment
        venting = self.equipment.containment_venting
        vent_to_wetwell = False
        vent_eq = self.equipment.get("containment_vent")
        if vent_eq and isinstance(vent_eq, Valve) and vent_eq.is_open():
            venting = True

        suppression_temp = 30.0
        pool = self.equipment.get("suppression_pool")
        if pool:
            suppression_temp = pool.temp_c

        containment_result = self.containment_model.update(
            dt, s.containment_pressure_mpa, s.containment_temp_c,
            s.containment_hydrogen_pct, fuel_result.hydrogen_rate_kg_s,
            s.fuel_damage_fraction, venting, vent_to_wetwell, suppression_temp,
        )
        s.containment_pressure_mpa = containment_result.pressure
        s.containment_temp_c = containment_result.temperature
        s.containment_hydrogen_pct = containment_result.hydrogen_pct
        s.radiation_level_sv_hr = containment_result.radiation_level
        s.environmental_release_tbq += containment_result.cumulative_release

        # 12. Windscale-specific: Wigner energy
        if self.reactor_type == "windscale" and s.wigner_energy_stored_j_per_kg > 0:
            air_flow = 0.0
            for eq_id in ("blower_1", "blower_2"):
                eq = self.equipment.get(eq_id)
                if eq and eq.status == EquipmentStatus.RUNNING:
                    from equipment import PUMP_RATED_FLOWS
                    air_flow += PUMP_RATED_FLOWS.get(eq_id, 200.0) * (eq.speed_pct / 100.0)

            water_injection = 0.0
            water_eq = self.equipment.get("water_injection")
            if water_eq and water_eq.status == EquipmentStatus.RUNNING:
                water_injection = water_eq.flow_kg_s if water_eq.flow_kg_s > 0 else 50.0

            new_temp, new_stored, release_mw = self.wigner.update(
                dt, s.graphite_temp_c, s.wigner_energy_stored_j_per_kg,
                air_flow_kg_s=air_flow,
                water_injection_kg_s=water_injection,
            )
            s.graphite_temp_c = new_temp
            s.wigner_energy_stored_j_per_kg = new_stored

            # Wigner release adds to fuel heating
            if s.graphite_temp_c > 300 and air_flow > 0:
                # Uranium can ignite in air above 300C
                s.cladding_temp_c = max(s.cladding_temp_c, s.graphite_temp_c * 0.9)
                s.fuel_temp_c = max(s.fuel_temp_c, s.graphite_temp_c * 0.8)

        # 13. Update time
        s.time_seconds += dt

    def get_instrument_readings(self, rng: _random.Random | None = None) -> dict[str, Any]:
        """Return instrument readings as the operator would see them.

        Applies instrument failures so readings may be misleading.
        """
        if rng is None:
            rng = self.rng

        s = self.state
        em = self.equipment

        readings: dict[str, Any] = {}

        # -- Neutronics --
        readings["neutronics"] = {
            "thermal_power_mw": round(em.get_instrument_reading(
                "thermal_power", s.thermal_power_mw, rng
            ), 1) if isinstance(em.get_instrument_reading("thermal_power", s.thermal_power_mw, rng), (int, float)) else em.get_instrument_reading("thermal_power", s.thermal_power_mw, rng),
            "power_pct": round(s.thermal_power_mw / self.params.rated_power_mw * 100, 1),
            "neutron_flux": "NORMAL" if 0.01 < s.neutron_population < 2.0 else (
                "HIGH" if s.neutron_population >= 2.0 else "LOW"
            ),
            "control_rod_position_pct": round(s.control_rod_position_pct, 1),
            "scram_active": s.scram_active,
        }

        if self.reactor_type == "rbmk":
            readings["neutronics"]["orm_count"] = s.orm_count

        # -- Thermal --
        readings["thermal"] = {
            "fuel_temp_c": round(em.get_instrument_reading(
                "fuel_temperature", s.fuel_temp_c, rng
            ), 1) if isinstance(em.get_instrument_reading("fuel_temperature", s.fuel_temp_c, rng), (int, float)) else em.get_instrument_reading("fuel_temperature", s.fuel_temp_c, rng),
            "cladding_temp_c": round(em.get_instrument_reading(
                "core_exit_thermocouples", s.cladding_temp_c, rng
            ), 1) if isinstance(em.get_instrument_reading("core_exit_thermocouples", s.cladding_temp_c, rng), (int, float)) else em.get_instrument_reading("core_exit_thermocouples", s.cladding_temp_c, rng),
            "coolant_inlet_temp_c": round(s.coolant_inlet_temp_c, 1),
            "coolant_outlet_temp_c": round(s.coolant_outlet_temp_c, 1),
            "coolant_pressure_mpa": round(s.coolant_pressure_mpa, 2),
            "coolant_flow_rate_kg_s": round(s.coolant_flow_rate_kg_s, 0),
            "void_fraction_pct": round(s.void_fraction * 100, 1),
        }

        if self.params.has_pressurizer:
            # Show APPARENT level (may be misleading in LOCA!)
            readings["thermal"]["pressurizer_level_pct"] = round(
                s.pressurizer_level_apparent_pct, 1
            )
            readings["thermal"]["pressurizer_pressure_mpa"] = round(
                s.pressurizer_pressure_mpa, 2
            )

        if self.reactor_type == "bwr":
            reactor_water_level = em.get_instrument_reading(
                "reactor_water_level",
                100.0 * (1.0 - s.void_fraction),  # Simplified water level
                rng,
            )
            readings["thermal"]["reactor_water_level_pct"] = (
                round(reactor_water_level, 1) if isinstance(reactor_water_level, (int, float))
                else reactor_water_level
            )

        # Decay heat
        readings["thermal"]["decay_heat_mw"] = round(s.decay_heat_mw, 1)

        # -- Containment --
        readings["containment"] = {
            "pressure_mpa": round(s.containment_pressure_mpa, 3),
            "temperature_c": round(s.containment_temp_c, 1),
            "hydrogen_pct": round(s.containment_hydrogen_pct, 1),
            "radiation_sv_hr": round(em.get_instrument_reading(
                "containment_radiation", s.radiation_level_sv_hr, rng
            ), 3) if isinstance(em.get_instrument_reading("containment_radiation", s.radiation_level_sv_hr, rng), (int, float)) else em.get_instrument_reading("containment_radiation", s.radiation_level_sv_hr, rng),
        }

        # Windscale-specific
        if self.reactor_type == "windscale":
            graphite_reading = em.get_instrument_reading(
                "graphite_thermocouple_zone_3", s.graphite_temp_c, rng
            )
            readings["thermal"]["graphite_temp_c"] = (
                round(graphite_reading, 1) if isinstance(graphite_reading, (int, float))
                else graphite_reading
            )
            readings["thermal"]["wigner_energy_stored"] = (
                "HIGH" if s.wigner_energy_stored_j_per_kg > 1000 else
                "MODERATE" if s.wigner_energy_stored_j_per_kg > 100 else "LOW"
            )

        # -- Equipment --
        readings["equipment"] = {}
        for eq_id, eq in self.equipment.equipment.items():
            apparent = eq.get_apparent_status()
            # For PORV, apply instrument failure
            if eq_id == "porv":
                apparent = em.get_instrument_reading("porv_position", apparent, rng)
                if apparent == "CLOSED":
                    readings["equipment"][eq_id] = {"status": "CLOSED"}
                    continue
            info: dict[str, Any] = {"status": apparent}
            if hasattr(eq, "charge_pct") and isinstance(eq, type(eq)):
                # Battery
                from equipment import Battery
                if isinstance(eq, Battery):
                    info["charge_pct"] = round(eq.charge_pct, 1)
                    info["hours_remaining"] = round(max(0, eq.hours_remaining), 1)
            if hasattr(eq, "temp_c") and eq.temp_c > 0:
                from equipment import SuppressionPool
                if isinstance(eq, SuppressionPool):
                    info["temp_c"] = round(eq.temp_c, 1)
                    info["level_pct"] = round(eq.level_pct, 1)
            readings["equipment"][eq_id] = info

        # -- Time --
        readings["time"] = {
            "elapsed_seconds": round(s.time_seconds, 0),
            "elapsed_minutes": round(s.time_seconds / 60.0, 1),
        }

        return readings

    def format_readings(self, readings: dict[str, Any]) -> str:
        """Format instrument readings for display."""
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append(f"  REACTOR CONTROL ROOM - {self.reactor_type.upper()}")
        lines.append(f"  Time: {readings['time']['elapsed_minutes']:.1f} minutes")
        lines.append("=" * 60)

        # Neutronics
        n = readings.get("neutronics", {})
        lines.append("\n--- NEUTRONICS ---")
        for key, val in n.items():
            label = key.replace("_", " ").title()
            lines.append(f"  {label}: {val}")

        # Thermal
        t = readings.get("thermal", {})
        lines.append("\n--- THERMAL-HYDRAULICS ---")
        for key, val in t.items():
            label = key.replace("_", " ").title()
            lines.append(f"  {label}: {val}")

        # Containment
        c = readings.get("containment", {})
        lines.append("\n--- CONTAINMENT ---")
        for key, val in c.items():
            label = key.replace("_", " ").title()
            lines.append(f"  {label}: {val}")

        # Equipment
        e = readings.get("equipment", {})
        lines.append("\n--- EQUIPMENT STATUS ---")
        for eq_id, info in e.items():
            status = info.get("status", "unknown")
            extra = ""
            if "charge_pct" in info:
                extra = f" (charge: {info['charge_pct']}%, ~{info['hours_remaining']}h remaining)"
            if "temp_c" in info:
                extra = f" (temp: {info['temp_c']}°C, level: {info['level_pct']}%)"
            lines.append(f"  {eq_id}: {status}{extra}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)
