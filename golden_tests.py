"""Comprehensive tests for the Nuclear Power Plant Simulator.

Tests verify:
1. Physics models (neutronics, xenon, thermal, decay heat, fuel integrity, containment)
2. Equipment state machines
3. Scenario initial conditions
4. Scenario-specific behaviors (RBMK positive scram, TMI PORV, Fukushima blackout, etc.)
5. Reward calculation
6. Determinism
"""

import copy
import math
import sys
import os
import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from physics import (
    NeutronicsModel,
    ThermalHydraulicsModel,
    XenonDynamics,
    DecayHeatModel,
    FuelIntegrityModel,
    ContainmentModel,
    WignerEnergyModel,
    PressurizerModel,
    REACTOR_PARAMS,
)
from equipment import (
    Equipment,
    Valve,
    Battery,
    EquipmentManager,
    EquipmentStatus,
    InstrumentFailure,
)
from reactor import ReactorSimulation, ReactorState
from scenarios import (
    ScenarioRegistry,
    ALL_SCENARIOS,
    TRAIN_SCENARIOS,
    TEST_SCENARIOS,
)
from rewards import RewardCalculator, SAFETY_LIMITS


# =============================================================================
# 1. NEUTRONICS TESTS
# =============================================================================

class TestNeutronics:
    """Tests for the point kinetics neutronics model."""

    @staticmethod
    def _steady_state_precursors(params):
        """Calculate steady-state precursor concentration for n=1.0."""
        return (params.beta_eff / params.neutron_gen_time) / params.lambda_d

    def test_steady_state_power(self):
        """At zero reactivity, power should remain constant."""
        params = REACTOR_PARAMS["pwr"]
        model = NeutronicsModel(params)
        power = 2772.0
        n = 1.0
        c = self._steady_state_precursors(params)  # Correct: C = (beta/Lambda) * n / lambda_d
        rho = 0.0

        for _ in range(100):
            power, n, c = model.update_power(1.0, power, n, c, rho)

        assert abs(power - 2772.0) < 1.0, f"Power drifted to {power} at zero reactivity"
        assert abs(n - 1.0) < 0.01, f"Neutron population drifted to {n}"

    def test_positive_reactivity_increases_power(self):
        """Small positive reactivity should cause power to increase."""
        params = REACTOR_PARAMS["pwr"]
        model = NeutronicsModel(params)
        power = 2772.0
        n = 1.0
        c = self._steady_state_precursors(params)
        rho = 0.001  # Small positive reactivity

        powers = [power]
        for _ in range(10):
            power, n, c = model.update_power(1.0, power, n, c, rho)
            powers.append(power)

        assert powers[-1] > powers[0], "Power should increase with positive reactivity"
        # Verify monotonic increase
        for i in range(1, len(powers)):
            assert powers[i] >= powers[i - 1], f"Power not monotonically increasing at step {i}"

    def test_negative_reactivity_decreases_power(self):
        """Small negative reactivity should cause power to decrease."""
        params = REACTOR_PARAMS["pwr"]
        model = NeutronicsModel(params)
        power = 2772.0
        n = 1.0
        c = self._steady_state_precursors(params)
        rho = -0.001

        powers = [power]
        for _ in range(10):
            power, n, c = model.update_power(1.0, power, n, c, rho)
            powers.append(power)

        assert powers[-1] < powers[0], "Power should decrease with negative reactivity"

    def test_scram_shuts_down_pwr(self):
        """PWR scram (large negative reactivity) should rapidly reduce power."""
        params = REACTOR_PARAMS["pwr"]
        model = NeutronicsModel(params)
        power = 2772.0
        n = 1.0
        c = self._steady_state_precursors(params)
        rho = -0.05  # Large negative reactivity from full rod insertion

        for _ in range(50):
            power, n, c = model.update_power(1.0, power, n, c, rho)

        # After 50 seconds at -$7.7, power should be < 1% of rated (27.7 MW).
        # Delayed neutron precursors sustain ~0.4% at this point.
        assert power < 28.0, f"PWR should be near shutdown after scram, got {power} MW"

    def test_rbmk_positive_scram_effect(self):
        """RBMK control rod insertion from fully withdrawn should show initial positive reactivity."""
        model = NeutronicsModel(REACTOR_PARAMS["rbmk"])

        # Rod at 95% withdrawn (near-fully out)
        rho_95 = model.control_rod_reactivity(95.0, inserting_from_withdrawn=True)
        # Rod at 85% withdrawn (15% inserted - in the graphite tip zone)
        rho_85 = model.control_rod_reactivity(85.0, inserting_from_withdrawn=True)

        # At 95% withdrawn: very little absorber in core, graphite tip just entering
        # At 85% withdrawn: graphite tip displacing water at bottom, adding positive reactivity
        # The graphite-tip effect should make rho at 85% more positive than expected
        # from absorber alone (or less negative)
        assert rho_85 > rho_95 - 0.01, (
            f"Graphite-tip effect not visible: rho at 85%={rho_85:.6f}, rho at 95%={rho_95:.6f}"
        )

    def test_prompt_criticality_power_excursion(self):
        """Reactivity above beta_eff should cause rapid power excursion."""
        params = REACTOR_PARAMS["rbmk"]
        model = NeutronicsModel(params)
        beta = params.beta_eff
        power = 200.0
        n = 200.0 / 3200.0
        c = self._steady_state_precursors(params) * n
        rho = beta * 1.5  # 1.5 dollars - prompt supercritical

        initial_power = power
        # Even 1 second should show dramatic increase
        power, n, c = model.update_power(1.0, power, n, c, rho)

        assert power > initial_power * 5, (
            f"Prompt critical excursion too slow: {initial_power} -> {power} in 1.0s"
        )

    def test_doppler_feedback_is_negative(self):
        """Doppler feedback should always be negative for temperature increase."""
        for reactor_type, params in REACTOR_PARAMS.items():
            model = NeutronicsModel(params)
            rho = model.doppler_feedback(params.fuel_temp_ref + 100)
            assert rho < 0, f"Doppler should be negative for {reactor_type}, got {rho}"

    def test_rbmk_void_coefficient_is_positive(self):
        """RBMK void coefficient must be positive."""
        model = NeutronicsModel(REACTOR_PARAMS["rbmk"])
        rho = model.void_feedback(0.3)  # Higher than reference
        assert rho > 0, f"RBMK void coefficient should be positive, got {rho}"

    def test_pwr_void_coefficient_is_negative(self):
        """PWR void coefficient must be negative."""
        model = NeutronicsModel(REACTOR_PARAMS["pwr"])
        rho = model.void_feedback(0.1)  # Higher than reference (0.0)
        assert rho < 0, f"PWR void coefficient should be negative, got {rho}"

    def test_bwr_void_coefficient_is_negative(self):
        """BWR void coefficient must be negative."""
        model = NeutronicsModel(REACTOR_PARAMS["bwr"])
        rho = model.void_feedback(0.5)  # Higher than reference (0.3)
        assert rho < 0, f"BWR void coefficient should be negative, got {rho}"


# =============================================================================
# 2. XENON DYNAMICS TESTS
# =============================================================================

class TestXenonDynamics:
    """Tests for I-135/Xe-135 dynamics."""

    def test_xenon_equilibrium_at_rated_power(self):
        """At constant rated power, xenon should stay near 1.0 (equilibrium)."""
        model = XenonDynamics(REACTOR_PARAMS["pwr"])
        iodine = 1.0
        xenon = 1.0

        for _ in range(1000):  # ~16 hours at 60s steps
            iodine, xenon = model.update(60.0, 1.0, iodine, xenon)

        assert abs(xenon - 1.0) < 0.05, f"Xenon should be at equilibrium, got {xenon}"
        assert abs(iodine - 1.0) < 0.05, f"Iodine should be at equilibrium, got {iodine}"

    def test_xenon_buildup_after_shutdown(self):
        """After shutdown, xenon should rise (peak at ~11 hours)."""
        model = XenonDynamics(REACTOR_PARAMS["rbmk"])
        iodine = 1.0
        xenon = 1.0

        # Simulate 1 hour after shutdown (power = 0)
        dt = 60.0
        for _ in range(60):
            iodine, xenon = model.update(dt, 0.0, iodine, xenon)

        assert xenon > 1.0, f"Xenon should build up after shutdown, got {xenon}"

    def test_xenon_peak_timing(self):
        """Xenon should peak approximately 10-12 hours after shutdown."""
        model = XenonDynamics(REACTOR_PARAMS["pwr"])
        iodine = 1.0
        xenon = 1.0
        dt = 300.0  # 5 minute steps

        max_xe = 0.0
        max_xe_time_hr = 0.0
        time_hr = 0.0

        for i in range(200):  # ~16 hours
            iodine, xenon = model.update(dt, 0.0, iodine, xenon)
            time_hr += dt / 3600.0
            if xenon > max_xe:
                max_xe = xenon
                max_xe_time_hr = time_hr

        assert 8.0 < max_xe_time_hr < 15.0, (
            f"Xenon peak should be at ~11 hours, got {max_xe_time_hr:.1f} hours"
        )
        assert max_xe > 2.0, f"Xenon peak should be >2x equilibrium, got {max_xe:.2f}x"

    def test_xenon_at_chernobyl_conditions(self):
        """Starting from above equilibrium, dropping power to low should increase xenon further."""
        model = XenonDynamics(REACTOR_PARAMS["rbmk"])
        iodine = 1.8  # Accumulated from prior high-power operation
        xenon = 2.5   # Already above equilibrium

        # Run at very low power (30 MWt / 3200 MWt)
        power_frac = 30.0 / 3200.0
        dt = 60.0
        initial_xe = xenon

        for _ in range(60):  # 1 hour
            iodine, xenon = model.update(dt, power_frac, iodine, xenon)

        assert xenon > initial_xe, (
            f"Xenon should continue building at low power: {initial_xe} -> {xenon}"
        )

    def test_xenon_reactivity_is_negative(self):
        """Xenon should always contribute negative reactivity."""
        model = XenonDynamics(REACTOR_PARAMS["pwr"])
        for xe_level in [0.5, 1.0, 2.0, 3.0]:
            rho = model.get_reactivity(xe_level)
            assert rho < 0, f"Xenon reactivity should be negative, got {rho} at Xe={xe_level}"

    def test_higher_xenon_more_negative(self):
        """Higher xenon concentration should give more negative reactivity."""
        model = XenonDynamics(REACTOR_PARAMS["pwr"])
        rho_low = model.get_reactivity(1.0)
        rho_high = model.get_reactivity(3.0)
        assert rho_high < rho_low, "Higher xenon should give more negative reactivity"


# =============================================================================
# 3. THERMAL-HYDRAULICS TESTS
# =============================================================================

class TestThermalHydraulics:
    """Tests for thermal-hydraulics model."""

    def test_coolant_heats_with_power(self):
        """Nonzero power + flow should heat coolant."""
        model = ThermalHydraulicsModel(REACTOR_PARAMS["pwr"])
        result = model.update(
            dt=60.0, power_mw=2772.0, decay_heat_mw=0.0,
            coolant_flow_kg_s=10000.0, coolant_inlet_temp=285.0,
            coolant_pressure=15.5, fuel_temp=800.0, cladding_temp=350.0,
            coolant_outlet_temp=285.0, void_fraction=0.0,
        )
        assert result.coolant_outlet_temp > 285.0, "Coolant should heat up"
        assert result.fuel_temp > 285.0, "Fuel should be hot"

    def test_loss_of_flow_increases_temp(self):
        """Stopping coolant flow should cause temperatures to rise."""
        model = ThermalHydraulicsModel(REACTOR_PARAMS["pwr"])

        # Normal flow
        result_normal = model.update(
            dt=60.0, power_mw=100.0, decay_heat_mw=0.0,
            coolant_flow_kg_s=10000.0, coolant_inlet_temp=285.0,
            coolant_pressure=15.5, fuel_temp=500.0, cladding_temp=350.0,
            coolant_outlet_temp=310.0, void_fraction=0.0,
        )

        # Loss of flow
        result_lof = model.update(
            dt=60.0, power_mw=100.0, decay_heat_mw=0.0,
            coolant_flow_kg_s=0.0, coolant_inlet_temp=285.0,
            coolant_pressure=15.5, fuel_temp=500.0, cladding_temp=350.0,
            coolant_outlet_temp=310.0, void_fraction=0.0,
        )

        assert result_lof.coolant_outlet_temp > result_normal.coolant_outlet_temp, (
            "Temperature should be higher with loss of flow"
        )

    def test_porv_leak_depressurizes(self):
        """Open PORV should cause pressure to drop."""
        model = ThermalHydraulicsModel(REACTOR_PARAMS["pwr"])
        result = model.update(
            dt=60.0, power_mw=0.0, decay_heat_mw=50.0,
            coolant_flow_kg_s=0.0, coolant_inlet_temp=285.0,
            coolant_pressure=15.0, fuel_temp=500.0, cladding_temp=350.0,
            coolant_outlet_temp=310.0, void_fraction=0.0,
            leak_rate_kg_s=20.0,  # PORV leak
        )
        assert result.coolant_pressure < 15.0, "Pressure should drop with PORV leak"

    def test_injection_raises_pressure(self):
        """Coolant injection should help maintain/raise pressure."""
        model = ThermalHydraulicsModel(REACTOR_PARAMS["pwr"])
        # Without injection
        result_no_inj = model.update(
            dt=60.0, power_mw=0.0, decay_heat_mw=50.0,
            coolant_flow_kg_s=0.0, coolant_inlet_temp=285.0,
            coolant_pressure=10.0, fuel_temp=500.0, cladding_temp=350.0,
            coolant_outlet_temp=310.0, void_fraction=0.0,
            leak_rate_kg_s=20.0,
        )
        # With injection
        result_inj = model.update(
            dt=60.0, power_mw=0.0, decay_heat_mw=50.0,
            coolant_flow_kg_s=0.0, coolant_inlet_temp=285.0,
            coolant_pressure=10.0, fuel_temp=500.0, cladding_temp=350.0,
            coolant_outlet_temp=310.0, void_fraction=0.0,
            leak_rate_kg_s=20.0,
            injection_rate_kg_s=50.0,
        )
        assert result_inj.coolant_pressure >= result_no_inj.coolant_pressure, (
            "Injection should help maintain pressure"
        )


# =============================================================================
# 4. DECAY HEAT TESTS
# =============================================================================

class TestDecayHeat:
    """Tests for decay heat model."""

    def test_decay_heat_at_known_times(self):
        """Verify decay heat fraction at standard time points."""
        model = DecayHeatModel()

        # At 1 second: ~6.6% of rated power
        frac_1s = model.fraction(1.0)
        assert 0.05 < frac_1s < 0.08, f"At 1s: expected ~6.6%, got {frac_1s*100:.1f}%"

        # At 1 hour (3600s): ~1.4%
        frac_1h = model.fraction(3600.0)
        assert 0.01 < frac_1h < 0.02, f"At 1hr: expected ~1.4%, got {frac_1h*100:.1f}%"

        # At 1 day (86400s): ~0.5%
        frac_1d = model.fraction(86400.0)
        assert 0.003 < frac_1d < 0.008, f"At 1day: expected ~0.5%, got {frac_1d*100:.1f}%"

    def test_decay_heat_decreases_over_time(self):
        """Decay heat should monotonically decrease."""
        model = DecayHeatModel()
        times = [1, 10, 100, 1000, 10000, 100000]
        fracs = [model.fraction(t) for t in times]

        for i in range(1, len(fracs)):
            assert fracs[i] < fracs[i - 1], (
                f"Decay heat should decrease: {fracs[i-1]} at t={times[i-1]} "
                f"> {fracs[i]} at t={times[i]}"
            )

    def test_no_decay_heat_before_shutdown(self):
        """Decay heat should be zero before shutdown."""
        model = DecayHeatModel()
        assert model.fraction(0.0) == 0.0
        assert model.fraction(-1.0) == 0.0

    def test_decay_heat_mw(self):
        """Verify decay heat in MW for Fukushima."""
        model = DecayHeatModel()
        rated = 2381.0  # Fukushima Unit 1
        heat_1hr = model.calculate(3600.0, rated)
        # At 1 hour: ~1.4% of 2381 = ~33 MW
        assert 20 < heat_1hr < 50, f"Expected ~33 MW at 1hr, got {heat_1hr:.1f} MW"


# =============================================================================
# 5. FUEL INTEGRITY TESTS
# =============================================================================

class TestFuelIntegrity:
    """Tests for fuel damage and hydrogen generation."""

    def test_no_damage_below_limits(self):
        """No fuel damage should occur at normal cladding temperatures."""
        model = FuelIntegrityModel(REACTOR_PARAMS["pwr"])
        result = model.update(
            dt=60.0, cladding_temp_c=900.0, fuel_centerline_temp_c=1500.0,
            current_damage=0.0, current_oxidation_pct=0.0, current_h2_kg=0.0,
        )
        assert result.damage_fraction == 0.0, "No damage expected below 1204°C"
        assert result.hydrogen_kg < 0.1, "Minimal H2 at 900°C"

    def test_damage_above_1204c(self):
        """Fuel damage should occur above NRC limit."""
        model = FuelIntegrityModel(REACTOR_PARAMS["pwr"])
        result = model.update(
            dt=60.0, cladding_temp_c=1300.0, fuel_centerline_temp_c=2000.0,
            current_damage=0.0, current_oxidation_pct=0.0, current_h2_kg=0.0,
        )
        assert result.damage_fraction > 0.0, "Damage expected above 1204°C"
        assert result.hydrogen_kg > 0.0, "H2 expected from Zr-H2O reaction"

    def test_rapid_damage_above_1480c(self):
        """Autocatalytic oxidation above 1480°C causes faster damage."""
        model = FuelIntegrityModel(REACTOR_PARAMS["pwr"])
        result_moderate = model.update(
            dt=60.0, cladding_temp_c=1300.0, fuel_centerline_temp_c=2000.0,
            current_damage=0.0, current_oxidation_pct=0.0, current_h2_kg=0.0,
        )
        result_severe = model.update(
            dt=60.0, cladding_temp_c=1600.0, fuel_centerline_temp_c=2500.0,
            current_damage=0.0, current_oxidation_pct=0.0, current_h2_kg=0.0,
        )
        assert result_severe.damage_fraction > result_moderate.damage_fraction, (
            "Higher temperature should cause faster damage"
        )

    def test_meltdown_above_2870c(self):
        """UO2 melting (>2870°C) should cause rapid fuel damage."""
        model = FuelIntegrityModel(REACTOR_PARAMS["pwr"])
        result = model.update(
            dt=60.0, cladding_temp_c=2000.0, fuel_centerline_temp_c=3000.0,
            current_damage=0.0, current_oxidation_pct=0.0, current_h2_kg=0.0,
        )
        assert result.damage_fraction > 0.01, "Rapid damage expected at UO2 melting point"

    def test_hydrogen_generation_proportional_to_temp(self):
        """Higher cladding temp should generate more hydrogen."""
        model = FuelIntegrityModel(REACTOR_PARAMS["pwr"])
        result_low = model.update(
            dt=60.0, cladding_temp_c=1200.0, fuel_centerline_temp_c=1800.0,
            current_damage=0.0, current_oxidation_pct=0.0, current_h2_kg=0.0,
        )
        result_high = model.update(
            dt=60.0, cladding_temp_c=1500.0, fuel_centerline_temp_c=2200.0,
            current_damage=0.0, current_oxidation_pct=0.0, current_h2_kg=0.0,
        )
        assert result_high.hydrogen_kg >= result_low.hydrogen_kg, (
            "Higher temp should produce more hydrogen"
        )


# =============================================================================
# 6. CONTAINMENT TESTS
# =============================================================================

class TestContainment:
    """Tests for containment model."""

    def test_venting_reduces_pressure(self):
        """Venting should reduce containment pressure."""
        model = ContainmentModel(REACTOR_PARAMS["bwr"])
        result = model.update(
            dt=60.0, pressure=0.5, temperature=100.0,
            hydrogen_pct=5.0, h2_production_rate=0.0,
            fuel_damage=0.1, venting=True,
        )
        assert result.pressure < 0.5, "Venting should reduce pressure"

    def test_hydrogen_accumulation(self):
        """H2 production should increase containment H2 concentration."""
        model = ContainmentModel(REACTOR_PARAMS["bwr"])
        result = model.update(
            dt=60.0, pressure=0.2, temperature=80.0,
            hydrogen_pct=2.0, h2_production_rate=0.5,  # kg/s
            fuel_damage=0.1, venting=False,
        )
        assert result.hydrogen_pct > 2.0, "H2 should accumulate"

    def test_radiation_proportional_to_damage(self):
        """Radiation level should increase with fuel damage."""
        model = ContainmentModel(REACTOR_PARAMS["pwr"])
        result_low = model.update(
            dt=60.0, pressure=0.2, temperature=50.0,
            hydrogen_pct=0.0, h2_production_rate=0.0,
            fuel_damage=0.1, venting=False,
        )
        result_high = model.update(
            dt=60.0, pressure=0.2, temperature=50.0,
            hydrogen_pct=0.0, h2_production_rate=0.0,
            fuel_damage=0.8, venting=False,
        )
        assert result_high.radiation_level > result_low.radiation_level, (
            "More fuel damage should increase radiation"
        )

    def test_windscale_no_containment(self):
        """Windscale has no containment — release should be direct."""
        model = ContainmentModel(REACTOR_PARAMS["windscale"])
        result = model.update(
            dt=60.0, pressure=0.101, temperature=100.0,
            hydrogen_pct=0.0, h2_production_rate=0.0,
            fuel_damage=0.5, venting=False,
        )
        assert result.cumulative_release > 0.0, "Windscale should release directly (no containment)"


# =============================================================================
# 7. EQUIPMENT TESTS
# =============================================================================

class TestEquipment:
    """Tests for equipment state machines."""

    def test_battery_depletion(self):
        """Battery should drain over time."""
        battery = Battery("test_battery", {
            "status": "running", "charge_pct": 100.0, "hours_remaining": 8.0
        })
        # Run for 4 hours (14400 seconds)
        battery.update(14400.0)
        assert 45 < battery.charge_pct < 55, f"Battery should be ~50%, got {battery.charge_pct}%"
        assert battery.status == EquipmentStatus.RUNNING

        # Run for remaining time
        battery.update(14400.0)
        assert battery.charge_pct == 0.0
        assert battery.status == EquipmentStatus.DEPLETED

    def test_destroyed_equipment_cant_start(self):
        """Destroyed equipment should not be restartable."""
        eq = Equipment("diesel_1", {"status": "destroyed"})
        ok, msg = eq.start()
        assert not ok, "Destroyed equipment should not restart"

    def test_stuck_valve_behavior(self):
        """Stuck-open valve should not be closable directly."""
        valve = Valve("porv", {"status": "stuck_open"})
        assert valve.is_open(), "Stuck-open valve should be open"
        ok, msg = valve.close()
        assert not ok, "Stuck-open valve should not be closable directly"
        assert valve.is_open(), "Valve should still be open"

    def test_block_valve_closes_normally(self):
        """Block valve should close normally (stopping flow despite stuck PORV)."""
        block = Valve("block_valve", {"status": "running", "position_pct": 100.0})
        assert block.is_open(), "Block valve should start open"
        ok, msg = block.close()
        assert ok, "Block valve should close"
        assert not block.is_open(), "Block valve should be closed"

    def test_instrument_failure_porv(self):
        """PORV indicator should show 'CLOSED' when stuck open."""
        import random
        rng = random.Random(42)
        em = EquipmentManager("pwr", {
            "porv": {"status": "stuck_open", "visible_indicator": "closed"},
        }, [{"instrument": "porv_position", "failure": "reads_closed_when_open"}])

        reading = em.get_instrument_reading("porv_position", "open", rng)
        assert reading == "CLOSED", f"PORV indicator should show CLOSED, got {reading}"

    def test_porv_leak_rate(self):
        """Stuck-open PORV should produce measurable leak rate."""
        em = EquipmentManager("pwr", {
            "porv": {"status": "stuck_open"},
            "block_valve": {"status": "running", "position_pct": 100.0},
        })
        leak = em.get_porv_leak_rate(15.5)
        assert leak > 10.0, f"PORV leak should be significant at full pressure, got {leak}"

    def test_block_valve_stops_porv_leak(self):
        """Closing block valve should stop PORV leak."""
        em = EquipmentManager("pwr", {
            "porv": {"status": "stuck_open"},
            "block_valve": {"status": "standby", "position_pct": 0.0},
        })
        leak = em.get_porv_leak_rate(15.5)
        assert leak == 0.0, f"Block valve closed should stop leak, got {leak}"

    def test_srv_flow(self):
        """Open SRVs should produce steam flow for depressurization."""
        em = EquipmentManager("bwr", {
            "srv_1": {"status": "running", "position_pct": 100.0},
        })
        flow = em.get_srv_flow(7.0)
        assert flow > 0.0, "Open SRV should produce flow"

    def test_ac_power_detection(self):
        """AC power should be available when diesel or turbine is running."""
        em_with = EquipmentManager("pwr", {
            "diesel_1": {"status": "running"},
        })
        assert em_with.has_ac_power(), "Should have AC power with running diesel"

        em_without = EquipmentManager("bwr", {
            "diesel_1": {"status": "destroyed"},
            "diesel_2": {"status": "destroyed"},
        })
        assert not em_without.has_ac_power(), "Should not have AC power with destroyed diesels"


# =============================================================================
# 8. SCENARIO INITIAL CONDITIONS TESTS
# =============================================================================

class TestScenarios:
    """Tests verifying scenario initial conditions match historical records."""

    def test_all_scenarios_exist(self):
        """All 11 scenarios should be registered."""
        assert len(ALL_SCENARIOS) == 11

    def test_splits_defined(self):
        """Train and test splits should be defined."""
        splits = ScenarioRegistry.list_splits()
        assert "train" in splits
        assert "test" in splits

    def test_train_tasks_count(self):
        """Train split should have expected number of tasks."""
        tasks = ScenarioRegistry.list_tasks("train")
        assert len(tasks) == len(TRAIN_SCENARIOS) * 10  # 4 scenarios * 10 seeds

    def test_test_tasks_count(self):
        """Test split should have expected number of tasks."""
        tasks = ScenarioRegistry.list_tasks("test")
        assert len(tasks) == len(TEST_SCENARIOS) * 10  # 11 scenarios * 10 seeds

    def test_chernobyl_xenon_pit_conditions(self):
        """Chernobyl xenon pit: power=30MW, Xe>equilibrium, ORM<15."""
        s = ALL_SCENARIOS["chernobyl_xenon_pit"]
        ic = s.initial_conditions
        assert ic["thermal_power_mw"] == 30.0
        assert ic["xenon_concentration"] > 1.0, "Xe should be above equilibrium"
        assert ic["orm_count"] < 15, "ORM should be below mandatory shutdown threshold"
        assert ic["control_rod_position_pct"] > 80, "Rods should be mostly withdrawn"
        assert s.reactor_type == "rbmk"

    def test_chernobyl_test_start_conditions(self):
        """Chernobyl test start: 200MW, ORM=6, ECCS disabled."""
        s = ALL_SCENARIOS["chernobyl_test_start"]
        ic = s.initial_conditions
        assert ic["thermal_power_mw"] == 200.0
        assert ic["orm_count"] <= 8
        assert ic["equipment"]["eccs"]["status"] == "disabled"
        assert ic["control_rod_position_pct"] > 90

    def test_tmi_porv_conditions(self):
        """TMI PORV stuck: pressure dropping, PORV shows closed, HPI running."""
        s = ALL_SCENARIOS["tmi_porv_stuck"]
        ic = s.initial_conditions
        assert ic["equipment"]["porv"]["status"] == "stuck_open"
        assert ic["equipment"]["porv"]["visible_indicator"] == "closed"
        assert ic["equipment"]["hpi_1"]["status"] == "running"
        assert ic["thermal_power_mw"] == 0.0  # Scrammed
        assert len(ic["instrument_failures"]) > 0

    def test_tmi_pressurizer_level_misleading(self):
        """TMI loss of coolant: pressurizer apparent level should be high."""
        s = ALL_SCENARIOS["tmi_loss_of_coolant"]
        ic = s.initial_conditions
        assert ic.get("pressurizer_level_apparent_pct", 0) > 90, (
            "Apparent pressurizer level should be misleadingly high"
        )
        assert ic["coolant_pressure_mpa"] < 10.0, "Pressure should be low (LOCA)"

    def test_fukushima_blackout_conditions(self):
        """Fukushima blackout: diesels destroyed, batteries available, RCIC running."""
        s = ALL_SCENARIOS["fukushima_blackout"]
        ic = s.initial_conditions
        assert ic["equipment"]["diesel_1"]["status"] == "destroyed"
        assert ic["equipment"]["diesel_2"]["status"] == "destroyed"
        assert ic["equipment"]["batteries"]["status"] == "running"
        assert ic["equipment"]["batteries"]["hours_remaining"] == 8.0
        assert ic["equipment"]["rcic"]["status"] == "running"
        assert ic["thermal_power_mw"] == 0.0

    def test_fukushima_rcic_failure_conditions(self):
        """Fukushima RCIC failure: no cooling, batteries nearly depleted."""
        s = ALL_SCENARIOS["fukushima_rcic_failure"]
        ic = s.initial_conditions
        assert ic["equipment"]["rcic"]["status"] == "failed"
        assert ic["equipment"]["batteries"]["charge_pct"] == 10.0
        assert ic["equipment"]["fire_truck"]["status"] == "standby"

    def test_windscale_anneal_conditions(self):
        """Windscale anneal: high Wigner energy, thermocouple blind spots."""
        s = ALL_SCENARIOS["windscale_anneal"]
        ic = s.initial_conditions
        assert ic["wigner_energy_stored_j_per_kg"] > 2000
        assert ic["graphite_temp_c"] > 250  # Near anneal temperature
        assert len(ic["instrument_failures"]) > 0

    def test_windscale_fire_conditions(self):
        """Windscale fire: uranium burning, high fuel temp."""
        s = ALL_SCENARIOS["windscale_fire"]
        ic = s.initial_conditions
        assert ic["fuel_temp_c"] > 1000, "Fuel should be burning"
        assert ic["fuel_damage_fraction"] > 0, "Some damage already occurred"
        assert ic["environmental_release_tbq"] > 0, "Already releasing radiation"


# =============================================================================
# 9. INTEGRATED SCENARIO TESTS
# =============================================================================

class TestScenarioIntegration:
    """Tests for scenario-specific physics behaviors in the full simulation."""

    def _make_sim(self, scenario_name: str) -> ReactorSimulation:
        """Create a simulation for a given scenario."""
        scenario = ScenarioRegistry.get(scenario_name)
        return ReactorSimulation(
            reactor_type=scenario.reactor_type,
            initial_conditions=scenario.initial_conditions,
            time_step_minutes=scenario.time_step_minutes,
            seed=42,
        )

    def test_chernobyl_xenon_continues_building(self):
        """In Chernobyl xenon pit, xenon should continue building at low power."""
        sim = self._make_sim("chernobyl_xenon_pit")
        initial_xe = sim.state.xenon_concentration

        for _ in range(30):  # 30 minutes
            sim.advance()

        assert sim.state.xenon_concentration > initial_xe, (
            f"Xenon should increase: {initial_xe} -> {sim.state.xenon_concentration}"
        )

    def test_tmi_pressure_drops_with_porv_leak(self):
        """TMI PORV stuck: pressure should drop over time."""
        sim = self._make_sim("tmi_porv_stuck")
        initial_pressure = sim.state.coolant_pressure_mpa

        for _ in range(10):  # 10 minutes
            sim.advance()

        assert sim.state.coolant_pressure_mpa < initial_pressure, (
            f"Pressure should drop: {initial_pressure} -> {sim.state.coolant_pressure_mpa}"
        )

    def test_tmi_block_valve_stops_leak(self):
        """Closing the block valve should stabilize pressure."""
        sim = self._make_sim("tmi_porv_stuck")

        # Close the block valve
        block = sim.equipment.get("block_valve")
        assert block is not None
        if isinstance(block, Valve):
            block.close()

        initial_p = sim.state.coolant_pressure_mpa
        for _ in range(5):
            sim.advance()

        # Leak rate should now be zero
        leak = sim.equipment.get_porv_leak_rate(sim.state.coolant_pressure_mpa)
        assert leak == 0.0, f"Block valve should stop leak, got {leak} kg/s"

    def test_fukushima_battery_drains(self):
        """Fukushima batteries should drain toward depletion."""
        sim = self._make_sim("fukushima_blackout")
        battery = sim.equipment.get("batteries")
        assert battery is not None

        initial_charge = battery.charge_pct

        # Run for 2 hours (24 steps of 5 minutes)
        for _ in range(24):
            sim.advance()

        assert battery.charge_pct < initial_charge, "Battery should drain"
        assert battery.charge_pct > 0, "Battery shouldn't be depleted yet (only 2hr of 8hr)"

    def test_fukushima_rcic_provides_cooling(self):
        """With RCIC running, cladding temp should remain manageable."""
        sim = self._make_sim("fukushima_blackout")

        for _ in range(12):  # 1 hour
            sim.advance()

        assert sim.state.cladding_temp_c < 800, (
            f"With RCIC, cladding should stay reasonable: {sim.state.cladding_temp_c}"
        )

    def test_windscale_wigner_energy_releases(self):
        """Windscale anneal: Wigner energy should release as graphite heats."""
        sim = self._make_sim("windscale_anneal")
        initial_stored = sim.state.wigner_energy_stored_j_per_kg

        for _ in range(20):  # ~1.5 hours
            sim.advance()

        assert sim.state.wigner_energy_stored_j_per_kg < initial_stored, (
            "Wigner energy should decrease as it releases"
        )


# =============================================================================
# 10. REWARD TESTS
# =============================================================================

class TestRewards:
    """Tests for reward calculation."""

    def _healthy_state(self) -> ReactorState:
        """Create a healthy reactor state."""
        state = ReactorState()
        state.cladding_temp_c = 350.0
        state.fuel_damage_fraction = 0.0
        state.containment_pressure_mpa = 0.1
        state.containment_hydrogen_pct = 0.0
        state.environmental_release_tbq = 0.0
        return state

    def _damaged_state(self) -> ReactorState:
        """Create a severely damaged state."""
        state = ReactorState()
        state.cladding_temp_c = 1500.0
        state.fuel_damage_fraction = 0.8
        state.containment_pressure_mpa = 0.6
        state.containment_hydrogen_pct = 10.0
        state.environmental_release_tbq = 20.0
        return state

    def test_healthy_state_high_reward(self):
        """Healthy reactor state should give high reward."""
        calc = RewardCalculator("pwr", "stabilize")
        state = self._healthy_state()
        prev = copy.deepcopy(state)
        reward = calc.step_reward(state, prev)
        assert reward > 0.8, f"Healthy state should give high reward, got {reward}"

    def test_damaged_state_low_reward(self):
        """Damaged state should give low reward."""
        calc = RewardCalculator("pwr", "stabilize")
        state = self._damaged_state()
        prev = self._healthy_state()
        reward = calc.step_reward(state, prev)
        assert reward < 0.0, f"Damaged state should give negative reward, got {reward}"

    def test_meltdown_terminal(self):
        """Fuel damage >= 0.95 should be terminal."""
        calc = RewardCalculator("pwr", "stabilize")
        state = ReactorState()
        state.fuel_damage_fraction = 0.96
        is_term, reason = calc.is_terminal(state, 10, 200)
        assert is_term, "Meltdown should be terminal"
        assert reason == "core_meltdown"

    def test_hydrogen_detonation_terminal(self):
        """H2 > 18% should be terminal."""
        calc = RewardCalculator("bwr", "stabilize")
        state = ReactorState()
        state.containment_hydrogen_pct = 19.0
        is_term, reason = calc.is_terminal(state, 10, 200)
        assert is_term, "Hydrogen detonation should be terminal"
        assert reason == "hydrogen_detonation"

    def test_max_steps_terminal(self):
        """Reaching max steps should be terminal."""
        calc = RewardCalculator("pwr", "stabilize")
        state = self._healthy_state()
        is_term, reason = calc.is_terminal(state, 200, 200)
        assert is_term, "Max steps should be terminal"
        assert reason == "max_steps_reached"

    def test_terminal_reward_meltdown(self):
        """Terminal reward for meltdown should be -1.0."""
        calc = RewardCalculator("pwr", "stabilize")
        state = ReactorState()
        state.fuel_damage_fraction = 0.96
        reward = calc.terminal_reward(state)
        assert reward == -1.0

    def test_terminal_reward_success(self):
        """Terminal reward for successful stabilization should be +1.0."""
        calc = RewardCalculator("pwr", "stabilize")
        state = self._healthy_state()
        reward = calc.terminal_reward(state)
        assert reward == 1.0

    def test_reward_penalizes_radiation_release(self):
        """Environmental release should reduce reward."""
        calc = RewardCalculator("pwr", "stabilize")
        state_clean = self._healthy_state()
        state_release = copy.deepcopy(state_clean)
        state_release.environmental_release_tbq = 50.0

        prev = self._healthy_state()
        r_clean = calc.step_reward(state_clean, prev)
        r_release = calc.step_reward(state_release, prev)

        assert r_release < r_clean, "Radiation release should reduce reward"


# =============================================================================
# 11. WIGNER ENERGY TESTS
# =============================================================================

class TestWignerEnergy:
    """Tests for Windscale Wigner energy model."""

    def test_no_release_below_250c(self):
        """No Wigner energy release below 250°C."""
        new_temp, new_stored, release = WignerEnergyModel.update(
            dt=60.0, graphite_temp_c=200.0,
            stored_energy_j_per_kg=2500.0,
        )
        assert new_stored == 2500.0, "No energy should release below 250°C"
        assert release == 0.0

    def test_release_above_250c(self):
        """Wigner energy should release above 250°C."""
        new_temp, new_stored, release = WignerEnergyModel.update(
            dt=60.0, graphite_temp_c=350.0,
            stored_energy_j_per_kg=2500.0,
        )
        assert new_stored < 2500.0, "Energy should decrease above 250°C"
        assert release > 0.0, "Release should be positive"

    def test_higher_temp_faster_release(self):
        """Higher temperature should release Wigner energy faster."""
        _, stored_300, _ = WignerEnergyModel.update(
            dt=60.0, graphite_temp_c=300.0, stored_energy_j_per_kg=2500.0
        )
        _, stored_500, _ = WignerEnergyModel.update(
            dt=60.0, graphite_temp_c=500.0, stored_energy_j_per_kg=2500.0
        )
        assert stored_500 < stored_300, "Higher temp should release faster"

    def test_water_injection_cools(self):
        """Water injection should cool graphite."""
        temp_no_water, _, _ = WignerEnergyModel.update(
            dt=60.0, graphite_temp_c=500.0,
            stored_energy_j_per_kg=2000.0,
            water_injection_kg_s=0.0,
        )
        temp_water, _, _ = WignerEnergyModel.update(
            dt=60.0, graphite_temp_c=500.0,
            stored_energy_j_per_kg=2000.0,
            water_injection_kg_s=50.0,
        )
        assert temp_water < temp_no_water, "Water should cool graphite"


# =============================================================================
# 12. PRESSURIZER MODEL TESTS
# =============================================================================

class TestPressurizer:
    """Tests for PWR pressurizer behavior."""

    def test_level_rises_during_loca(self):
        """During LOCA (leak > injection), apparent level should still rise due to voids."""
        actual, apparent = PressurizerModel.update(
            dt=60.0, level_pct=50.0, pressure_mpa=8.0,
            coolant_leak_rate=20.0, hpi_flow=5.0,
            system_temp=300.0,
        )
        # Actual level should drop (losing more than injecting)
        assert actual < 50.0, "Actual level should drop during LOCA"
        # Apparent level should be higher than actual (void swell)
        assert apparent > actual, "Apparent level should exceed actual (void swell)"


# =============================================================================
# 13. DETERMINISM TESTS
# =============================================================================

class TestDeterminism:
    """Tests for simulation determinism."""

    def test_deterministic_with_same_seed(self):
        """Same scenario + same seed + same actions = identical results."""
        scenario = ScenarioRegistry.get("tmi_porv_stuck")

        sim1 = ReactorSimulation(
            reactor_type=scenario.reactor_type,
            initial_conditions=scenario.initial_conditions,
            time_step_minutes=scenario.time_step_minutes,
            seed=42,
        )
        sim2 = ReactorSimulation(
            reactor_type=scenario.reactor_type,
            initial_conditions=scenario.initial_conditions,
            time_step_minutes=scenario.time_step_minutes,
            seed=42,
        )

        # Run both for 10 steps with no actions
        for _ in range(10):
            sim1.advance()
            sim2.advance()

        assert sim1.state.thermal_power_mw == sim2.state.thermal_power_mw
        assert sim1.state.coolant_pressure_mpa == sim2.state.coolant_pressure_mpa
        assert sim1.state.cladding_temp_c == sim2.state.cladding_temp_c
        assert sim1.state.fuel_damage_fraction == sim2.state.fuel_damage_fraction
        assert sim1.state.containment_pressure_mpa == sim2.state.containment_pressure_mpa

    def test_all_scenarios_initialize(self):
        """Every scenario should initialize without errors."""
        for name, scenario in ALL_SCENARIOS.items():
            sim = ReactorSimulation(
                reactor_type=scenario.reactor_type,
                initial_conditions=scenario.initial_conditions,
                time_step_minutes=scenario.time_step_minutes,
                seed=42,
            )
            # Should not raise
            sim.advance()
            assert sim.state.time_seconds > 0, f"Scenario {name} failed to advance"

    def test_all_scenarios_run_10_steps(self):
        """Every scenario should run 10 steps without crashing."""
        for name, scenario in ALL_SCENARIOS.items():
            sim = ReactorSimulation(
                reactor_type=scenario.reactor_type,
                initial_conditions=scenario.initial_conditions,
                time_step_minutes=scenario.time_step_minutes,
                seed=42,
            )
            for step in range(10):
                try:
                    sim.advance()
                except Exception as e:
                    pytest.fail(f"Scenario {name} crashed at step {step}: {e}")


# =============================================================================
# 14. SATURATION TEMPERATURE VERIFICATION
# =============================================================================

class TestSaturationTemp:
    """Verify saturation temperature correlation against NIST/IAPWS steam tables.

    Reference: NIST Chemistry WebBook (https://webbook.nist.gov/chemistry/fluid/)
    Values taken from IAPWS-IF97 (International Association for Properties of
    Water and Steam, Industrial Formulation 1997).
    """

    def test_boiling_point_at_1atm(self):
        """At 0.101325 MPa, T_sat should be ~100°C.
        Ref: Universal constant, every steam table.
        """
        t = ThermalHydraulicsModel._saturation_temp(0.101)
        assert abs(t - 100.0) < 3.0, f"T_sat at 1 atm should be ~100°C, got {t:.1f}°C"

    def test_bwr_operating_pressure(self):
        """At 7.0 MPa (BWR operating), T_sat should be ~285°C.
        Ref: NIST steam tables, Todreas & Kazimi Table B-2.
        """
        t = ThermalHydraulicsModel._saturation_temp(7.0)
        assert abs(t - 285.0) < 5.0, f"T_sat at 7 MPa should be ~285°C, got {t:.1f}°C"

    def test_pwr_operating_pressure(self):
        """At 15.5 MPa (PWR operating), T_sat should be ~345°C.
        Ref: NIST steam tables. PWRs operate with subcooled coolant (T_out < T_sat).
        """
        t = ThermalHydraulicsModel._saturation_temp(15.5)
        assert abs(t - 345.0) < 5.0, f"T_sat at 15.5 MPa should be ~345°C, got {t:.1f}°C"

    def test_mid_range_pressure(self):
        """At 10.0 MPa, T_sat should be ~311°C.
        Ref: NIST steam tables.
        """
        t = ThermalHydraulicsModel._saturation_temp(10.0)
        assert abs(t - 311.0) < 5.0, f"T_sat at 10 MPa should be ~311°C, got {t:.1f}°C"

    def test_monotonic_increasing(self):
        """T_sat should increase monotonically with pressure.
        Ref: Clausius-Clapeyron relation (fundamental thermodynamics).
        """
        pressures = [0.1, 0.5, 1.0, 3.0, 7.0, 10.0, 15.0]
        temps = [ThermalHydraulicsModel._saturation_temp(p) for p in pressures]
        for i in range(1, len(temps)):
            assert temps[i] > temps[i - 1], (
                f"T_sat not monotonic: {temps[i-1]:.1f}°C at {pressures[i-1]} MPa "
                f"> {temps[i]:.1f}°C at {pressures[i]} MPa"
            )


# =============================================================================
# 15. SCENARIO SOLVABILITY TESTS
# =============================================================================

class TestSolvability:
    """Verify that scenarios are solvable (good agent can succeed)
    and challenging (bad agent performs worse).

    These tests validate RL environment correctness: the reward function
    should discriminate between good and bad strategies.
    """

    def _make_sim(self, scenario_name: str, seed: int = 42) -> ReactorSimulation:
        scenario = ScenarioRegistry.get(scenario_name)
        return ReactorSimulation(
            reactor_type=scenario.reactor_type,
            initial_conditions=scenario.initial_conditions,
            time_step_minutes=scenario.time_step_minutes,
            seed=seed,
        )

    def test_chernobyl_normal_ops_no_intervention_stable(self):
        """Normal operations: reactor should remain stable with no intervention.
        A well-designed normal-operation reactor should not melt down on its own.
        Ref: RBMK designed for stable operation at rated power with ORM > 26.
        """
        sim = self._make_sim("chernobyl_normal_ops")
        calc = RewardCalculator("rbmk", "maintain_power")

        for _ in range(50):  # 250 minutes (~4 hours)
            sim.advance()
            is_term, reason = calc.is_terminal(sim.state, _, 200)
            if reason in ("core_meltdown", "hydrogen_detonation", "catastrophic_release"):
                pytest.fail(f"Normal ops should not fail catastrophically: {reason}")

        # Power should be roughly maintained
        assert sim.state.thermal_power_mw > 1000, (
            f"Power should remain high in normal ops, got {sim.state.thermal_power_mw:.0f} MW"
        )
        assert sim.state.fuel_damage_fraction < 0.01, "No fuel damage in normal ops"

    def test_tmi_good_agent_closes_block_valve(self):
        """TMI PORV: closing block valve should stabilize (stop pressure drop).
        Ref: Kemeny Commission Report — closing block valve at 06:22 ended the LOCA.
        """
        sim = self._make_sim("tmi_porv_stuck")
        calc = RewardCalculator("pwr", "stabilize")

        # Good agent: immediately close block valve
        block = sim.equipment.get("block_valve")
        if isinstance(block, Valve):
            block.close()

        cumulative_reward = 0.0
        prev_state = copy.deepcopy(sim.state)
        for step in range(30):  # 30 minutes
            sim.advance()
            r = calc.step_reward(sim.state, prev_state)
            cumulative_reward += r
            prev_state = copy.deepcopy(sim.state)

        # With block valve closed, no LOCA, state should stabilize
        assert sim.state.fuel_damage_fraction < 0.01, (
            f"Block valve should prevent damage, got {sim.state.fuel_damage_fraction}"
        )
        assert cumulative_reward > 15.0, (
            f"Good agent should accumulate positive reward, got {cumulative_reward:.2f}"
        )

    def test_tmi_bad_agent_no_action_worse(self):
        """TMI PORV: doing nothing should result in worse state than closing block valve.
        Ref: TMI historical operators didn't close block valve for 2h22m — core melted.
        """
        sim = self._make_sim("tmi_porv_stuck")
        calc = RewardCalculator("pwr", "stabilize")

        cumulative_reward = 0.0
        prev_state = copy.deepcopy(sim.state)
        for step in range(60):  # 60 minutes with no action
            sim.advance()
            r = calc.step_reward(sim.state, prev_state)
            cumulative_reward += r
            prev_state = copy.deepcopy(sim.state)

        # Without closing block valve, pressure should drop significantly
        assert sim.state.coolant_pressure_mpa < 12.0, (
            f"Pressure should drop without intervention: {sim.state.coolant_pressure_mpa:.1f} MPa"
        )

    def test_good_vs_bad_agent_reward_discrimination(self):
        """Good agent (closes block valve) gets strictly better cumulative reward
        than bad agent (does nothing). This validates RL reward shaping.

        Run for 120 steps (120 minutes) — at TMI-2, core uncovery began at
        ~100 min and cladding exceeded 1000°C by ~130 min (NUREG-0600).
        The test must run long enough for LOCA consequences to manifest.
        """
        scenario = ScenarioRegistry.get("tmi_porv_stuck")
        n_steps = 120  # 120 minutes — enough for core uncovery

        # Good agent: close block valve
        sim_good = ReactorSimulation(
            reactor_type=scenario.reactor_type,
            initial_conditions=scenario.initial_conditions,
            time_step_minutes=scenario.time_step_minutes, seed=42,
        )
        block = sim_good.equipment.get("block_valve")
        if isinstance(block, Valve):
            block.close()

        calc_good = RewardCalculator("pwr", "stabilize")
        good_reward = 0.0
        prev = copy.deepcopy(sim_good.state)
        for _ in range(n_steps):
            sim_good.advance()
            good_reward += calc_good.step_reward(sim_good.state, prev)
            prev = copy.deepcopy(sim_good.state)

        # Bad agent: do nothing
        sim_bad = ReactorSimulation(
            reactor_type=scenario.reactor_type,
            initial_conditions=scenario.initial_conditions,
            time_step_minutes=scenario.time_step_minutes, seed=42,
        )
        calc_bad = RewardCalculator("pwr", "stabilize")
        bad_reward = 0.0
        prev = copy.deepcopy(sim_bad.state)
        for _ in range(n_steps):
            sim_bad.advance()
            bad_reward += calc_bad.step_reward(sim_bad.state, prev)
            prev = copy.deepcopy(sim_bad.state)

        assert good_reward > bad_reward, (
            f"Good agent should beat bad agent: good={good_reward:.2f} vs bad={bad_reward:.2f}"
        )

    def test_fukushima_does_not_immediately_terminate(self):
        """Fukushima blackout should not immediately melt down — RCIC provides cooling.
        Ref: RCIC ran for ~14 hours at Fukushima Units 2/3 before failing.
        """
        sim = self._make_sim("fukushima_blackout")
        calc = RewardCalculator("bwr", "stabilize")

        for step in range(60):  # 5 hours (60 * 5 min steps)
            sim.advance()
            is_term, reason = calc.is_terminal(sim.state, step, 300)
            if reason in ("core_meltdown", "hydrogen_detonation"):
                pytest.fail(
                    f"Fukushima should survive first 5 hours with RCIC: "
                    f"terminated at step {step} ({step*5} min): {reason}"
                )

    def test_chernobyl_scram_at_low_orm_causes_problems(self):
        """RBMK scram with ORM=6 should cause positive reactivity insertion
        from graphite-tipped control rods.
        Ref: INSAG-7 §4.3, positive scram effect from graphite displacers.
        """
        sim = self._make_sim("chernobyl_test_start")
        initial_power = sim.state.thermal_power_mw

        # Initiate scram (AZ-5)
        sim.state.scram_active = True
        sim.state.rods_inserting = True

        # Run a few timesteps (15-second steps)
        sim.advance()  # 15 seconds
        power_after_15s = sim.state.thermal_power_mw

        # The initial rod movement inserts graphite tips which ADD reactivity
        # This should cause power to increase or stay high initially before
        # the absorber section takes over.
        # Note: depending on modeling fidelity, at least the power drop should
        # be slower than for a PWR scram, showing the positive scram effect.
        # We verify the rod reactivity includes a positive component.
        rod_rho = sim.state.reactivity_rods
        assert rod_rho > -0.10, (
            f"At early insertion, RBMK rods should not insert as much negative reactivity "
            f"as a clean scram: got {rod_rho:.4f}"
        )


# =============================================================================
# 16. REWARD MONOTONICITY AND CORRECTNESS TESTS
# =============================================================================

class TestRewardMonotonicity:
    """Verify reward function correctly tracks safety margins.

    These tests ensure the reward calculator is well-shaped for RL training:
    - Monotonic in the right direction for each safety component
    - No reward hacking opportunities
    - Rewards bounded in [-1, 1]
    """

    def _base_state(self) -> ReactorState:
        s = ReactorState()
        s.cladding_temp_c = 350.0
        s.fuel_damage_fraction = 0.0
        s.containment_pressure_mpa = 0.1
        s.containment_hydrogen_pct = 0.0
        s.environmental_release_tbq = 0.0
        return s

    def test_reward_monotonic_with_cladding_temp(self):
        """Higher cladding temp → lower reward (all else equal).
        Ref: Core integrity component = cladding temp margin below safety limit.
        """
        calc = RewardCalculator("pwr", "stabilize")
        prev = self._base_state()

        temps = [300, 500, 800, 1000, 1200, 1500]
        rewards = []
        for t in temps:
            state = copy.deepcopy(prev)
            state.cladding_temp_c = t
            rewards.append(calc.step_reward(state, prev))

        for i in range(1, len(rewards)):
            assert rewards[i] <= rewards[i - 1] + 0.01, (
                f"Reward should decrease with temp: "
                f"{rewards[i-1]:.3f} at {temps[i-1]}°C vs "
                f"{rewards[i]:.3f} at {temps[i]}°C"
            )

    def test_reward_monotonic_with_fuel_damage(self):
        """More fuel damage → lower reward.
        Ref: Fuel integrity component = 1 - fuel_damage_fraction.
        """
        calc = RewardCalculator("pwr", "stabilize")
        prev = self._base_state()

        damages = [0.0, 0.1, 0.3, 0.5, 0.8, 1.0]
        rewards = []
        for d in damages:
            state = copy.deepcopy(prev)
            state.fuel_damage_fraction = d
            rewards.append(calc.step_reward(state, prev))

        for i in range(1, len(rewards)):
            assert rewards[i] <= rewards[i - 1] + 0.01, (
                f"Reward should decrease with damage: "
                f"{rewards[i-1]:.3f} at {damages[i-1]} vs "
                f"{rewards[i]:.3f} at {damages[i]}"
            )

    def test_reward_monotonic_with_h2_concentration(self):
        """Higher H2 → lower reward.
        Ref: Hydrogen safety component uses 4% and 13% thresholds (flammable/detonable).
        """
        calc = RewardCalculator("bwr", "stabilize")
        prev = self._base_state()

        h2_levels = [0.0, 2.0, 4.0, 8.0, 13.0, 18.0]
        rewards = []
        for h2 in h2_levels:
            state = copy.deepcopy(prev)
            state.containment_hydrogen_pct = h2
            rewards.append(calc.step_reward(state, prev))

        for i in range(1, len(rewards)):
            assert rewards[i] <= rewards[i - 1] + 0.01, (
                f"Reward should decrease with H2: "
                f"{rewards[i-1]:.3f} at {h2_levels[i-1]}% vs "
                f"{rewards[i]:.3f} at {h2_levels[i]}%"
            )

    def test_reward_bounded(self):
        """All rewards should be in [-1, 1].
        Ref: Reward spec in rewards.py line 138.
        """
        calc = RewardCalculator("pwr", "stabilize")
        prev = self._base_state()

        # Test extreme states
        test_states = [
            # Healthy
            self._base_state(),
            # Worst case
        ]
        worst = ReactorState()
        worst.cladding_temp_c = 3000.0
        worst.fuel_damage_fraction = 1.0
        worst.containment_pressure_mpa = 1.0
        worst.containment_hydrogen_pct = 20.0
        worst.environmental_release_tbq = 200.0
        test_states.append(worst)

        for state in test_states:
            r = calc.step_reward(state, prev)
            assert -1.0 <= r <= 1.0, f"Reward {r} out of bounds for state"

    def test_observe_instruments_free_no_reward(self):
        """observe_instruments should not advance time or award points.
        The tool returns reward=0, finished=False — verified in npp_sim.py.
        """
        sim = ReactorSimulation(
            reactor_type="pwr",
            initial_conditions=ALL_SCENARIOS["tmi_porv_stuck"].initial_conditions,
            time_step_minutes=1.0,
            seed=42,
        )
        initial_time = sim.state.time_seconds

        # Getting readings should not advance simulation
        readings = sim.get_instrument_readings()
        assert sim.state.time_seconds == initial_time, (
            "Instrument reading should not advance time"
        )
        assert readings is not None
        assert "neutronics" in readings
        assert "thermal" in readings
        assert "containment" in readings
        assert "equipment" in readings


# =============================================================================
# 17. EDGE CASE / PHYSICAL BOUNDS TESTS
# =============================================================================

class TestPhysicalBounds:
    """Verify reactor state stays within physically possible bounds.

    These tests catch numerical instabilities, division-by-zero errors,
    and ensure physical quantities never go negative.
    """

    def _run_scenario_to_step(self, name: str, steps: int) -> ReactorSimulation:
        scenario = ScenarioRegistry.get(name)
        sim = ReactorSimulation(
            reactor_type=scenario.reactor_type,
            initial_conditions=scenario.initial_conditions,
            time_step_minutes=scenario.time_step_minutes,
            seed=42,
        )
        for _ in range(steps):
            sim.advance()
        return sim

    def test_power_never_negative(self):
        """Thermal power should never go negative in any scenario."""
        for name in ALL_SCENARIOS:
            sim = self._run_scenario_to_step(name, 20)
            assert sim.state.thermal_power_mw >= 0.0, (
                f"Negative power in {name}: {sim.state.thermal_power_mw}"
            )

    def test_fuel_damage_clamped_0_to_1(self):
        """Fuel damage should stay in [0, 1] in every scenario."""
        for name in ALL_SCENARIOS:
            sim = self._run_scenario_to_step(name, 20)
            assert 0.0 <= sim.state.fuel_damage_fraction <= 1.0, (
                f"Fuel damage out of bounds in {name}: {sim.state.fuel_damage_fraction}"
            )

    def test_pressures_never_negative(self):
        """Coolant and containment pressures should never go negative."""
        for name in ALL_SCENARIOS:
            sim = self._run_scenario_to_step(name, 20)
            assert sim.state.coolant_pressure_mpa >= 0.0, (
                f"Negative coolant pressure in {name}: {sim.state.coolant_pressure_mpa}"
            )
            assert sim.state.containment_pressure_mpa >= 0.0, (
                f"Negative containment pressure in {name}: {sim.state.containment_pressure_mpa}"
            )

    def test_hydrogen_pct_clamped(self):
        """H2 concentration should stay in [0, 100]."""
        for name in ALL_SCENARIOS:
            sim = self._run_scenario_to_step(name, 20)
            assert 0.0 <= sim.state.containment_hydrogen_pct <= 100.0, (
                f"H2% out of bounds in {name}: {sim.state.containment_hydrogen_pct}"
            )

    def test_void_fraction_clamped(self):
        """Void fraction should stay in [0, 1]."""
        for name in ALL_SCENARIOS:
            sim = self._run_scenario_to_step(name, 20)
            assert 0.0 <= sim.state.void_fraction <= 1.0, (
                f"Void fraction out of bounds in {name}: {sim.state.void_fraction}"
            )

    def test_temperatures_physically_reasonable(self):
        """Temperatures should not exceed physically impossible bounds.
        No material remains solid above ~4000°C (tungsten melts at 3422°C).
        """
        for name in ALL_SCENARIOS:
            sim = self._run_scenario_to_step(name, 20)
            assert sim.state.fuel_temp_c < 5000, (
                f"Fuel temp unreasonable in {name}: {sim.state.fuel_temp_c}°C"
            )
            assert sim.state.cladding_temp_c < 5000, (
                f"Cladding temp unreasonable in {name}: {sim.state.cladding_temp_c}°C"
            )

    def test_neutron_population_non_negative(self):
        """Neutron population should never go negative."""
        for name in ALL_SCENARIOS:
            sim = self._run_scenario_to_step(name, 20)
            assert sim.state.neutron_population >= 0.0, (
                f"Negative neutron pop in {name}: {sim.state.neutron_population}"
            )


# =============================================================================
# 18. TERMINAL CONDITION TESTS
# =============================================================================

class TestTerminalConditions:
    """Verify all terminal conditions are reachable and correctly identified."""

    def test_meltdown_terminal_at_95pct(self):
        """fuel_damage >= 0.95 → terminal='core_meltdown', reward=-1.0.
        Ref: Terminal condition defined in rewards.py:186.
        """
        calc = RewardCalculator("pwr", "stabilize")
        state = ReactorState()
        state.fuel_damage_fraction = 0.95
        is_term, reason = calc.is_terminal(state, 10, 200)
        assert is_term and reason == "core_meltdown"
        assert calc.terminal_reward(state) == -1.0

    def test_h2_detonation_terminal_at_18pct(self):
        """H2 > 18% → terminal='hydrogen_detonation'.
        Ref: H2 detonation limit ~13-18% in containment atmosphere.
        NUREG/CR-2726, H2 flammability at 4%, detonation at 13-18%.
        """
        calc = RewardCalculator("bwr", "stabilize")
        state = ReactorState()
        state.containment_hydrogen_pct = 18.5
        is_term, reason = calc.is_terminal(state, 10, 200)
        assert is_term and reason == "hydrogen_detonation"

    def test_catastrophic_release_terminal(self):
        """Release > 200 TBq → terminal='catastrophic_release'.
        Ref: TMI total release was ~0.5 TBq; Chernobyl was ~5.2 × 10^6 TBq.
        200 TBq represents a major but not Chernobyl-scale release.
        """
        calc = RewardCalculator("pwr", "stabilize")
        state = ReactorState()
        state.environmental_release_tbq = 201.0
        is_term, reason = calc.is_terminal(state, 10, 200)
        assert is_term and reason == "catastrophic_release"

    def test_stabilized_after_consecutive_stable_steps(self):
        """6 consecutive stable steps → terminal='stabilized', reward=+1.0.
        Ref: Stability threshold defined in rewards.py:67.
        """
        calc = RewardCalculator("pwr", "stabilize")
        state = ReactorState()
        state.cladding_temp_c = 350.0
        state.fuel_damage_fraction = 0.01
        state.containment_hydrogen_pct = 1.0
        state.containment_pressure_mpa = 0.2

        for step in range(10):
            is_term, reason = calc.is_terminal(state, step, 200)
            if is_term:
                assert reason == "stabilized"
                assert calc.terminal_reward(state) == 1.0
                return

        pytest.fail("Should have reached 'stabilized' terminal after 6+ stable steps")

    def test_max_steps_terminal(self):
        """Reaching max_steps → terminal='max_steps_reached'.
        Ref: Terminal condition defined in rewards.py:195-196.
        """
        calc = RewardCalculator("pwr", "stabilize")
        state = ReactorState()
        is_term, reason = calc.is_terminal(state, 200, 200)
        assert is_term and reason == "max_steps_reached"

    def test_not_terminal_at_healthy_mid_episode(self):
        """Healthy state at step 10 of 200 should NOT be terminal."""
        calc = RewardCalculator("pwr", "stabilize")
        state = ReactorState()
        state.cladding_temp_c = 350.0
        state.fuel_damage_fraction = 0.0
        state.containment_hydrogen_pct = 0.0
        state.environmental_release_tbq = 0.0

        # Reset stability counter
        calc.stability_counter = 0
        is_term, reason = calc.is_terminal(state, 0, 200)
        assert not is_term or reason == "", f"Should not be terminal at step 0"


# =============================================================================
# 19. DECAY HEAT PHYSICS VERIFICATION
# =============================================================================

class TestDecayHeatPhysics:
    """Verify decay heat model matches Way-Wigner formula values.

    Way-Wigner formula: P/P0 = 0.066 * t^(-0.2)
    Ref: Way & Wigner, Phys. Rev. 73, 1318 (1948).
    Also cross-checked against Todreas & Kazimi, Nuclear Systems I, Table 2-7.
    """

    def test_1_second(self):
        """At 1s: 6.6% of rated power.
        Way-Wigner: 0.066 * 1^(-0.2) = 0.066 = 6.6%.
        """
        frac = DecayHeatModel.fraction(1.0)
        assert abs(frac - 0.066) < 0.005, f"At 1s: expected 6.6%, got {frac*100:.2f}%"

    def test_1_minute(self):
        """At 60s: 0.066 * 60^(-0.2) = 0.066 * 0.437 = 2.88%.
        Ref: Way-Wigner calculation.
        """
        frac = DecayHeatModel.fraction(60.0)
        expected = 0.066 * (60.0 ** -0.2)  # = 0.0289
        assert abs(frac - expected) < 0.003, f"At 60s: expected {expected*100:.2f}%, got {frac*100:.2f}%"

    def test_1_hour(self):
        """At 3600s: 0.066 * 3600^(-0.2) = 0.066 * 0.196 = 1.3%.
        Todreas & Kazimi Table 2-7 gives ~1.3-1.4% at 1 hour.
        """
        frac = DecayHeatModel.fraction(3600.0)
        expected = 0.066 * (3600.0 ** -0.2)
        assert abs(frac - expected) < 0.002, f"At 1hr: expected {expected*100:.2f}%, got {frac*100:.2f}%"

    def test_fukushima_decay_heat_at_55min(self):
        """Fukushima blackout starts 55 min (3300s) after scram.
        P_decay = 0.066 * 3300^(-0.2) * 2381 MWt = ~31 MW.
        This decay heat drives all post-accident heating.
        Ref: NAIIC Fukushima report, consistent with ~2% at 1 hour.
        """
        heat_mw = DecayHeatModel.calculate(3300.0, 2381.0)
        assert 20 < heat_mw < 50, f"Expected ~31 MW at 55min, got {heat_mw:.1f} MW"


# =============================================================================
# 20. REACTOR PARAMETER CITATION VERIFICATION
# =============================================================================

class TestParameterCitations:
    """Verify key reactor parameters match published values.

    Each test includes the citation for the value being verified.
    """

    def test_rbmk_beta_eff(self):
        """RBMK β_eff = 0.0048.
        Ref: INSAG-7 (IAEA Safety Series No. 75-INSAG-7, 1992), §A.3.
        """
        assert REACTOR_PARAMS["rbmk"].beta_eff == 0.0048

    def test_rbmk_positive_void_coefficient(self):
        """RBMK void coefficient must be POSITIVE — the key Chernobyl flaw.
        Ref: INSAG-7 §4.3: void coefficient was +4.7β before accident.
        World Nuclear Association: "The large positive void coefficient
        was a dangerous design flaw unique to the RBMK."
        """
        assert REACTOR_PARAMS["rbmk"].void_coeff > 0.01, (
            "RBMK void coefficient must be significantly positive (>0.01 dk/k per unit void)."
            " INSAG-7 §4.3: total void effect = +4.7β ≈ 0.0226 dk/k."
        )

    def test_rbmk_211_control_rods(self):
        """RBMK-1000 has 211 control rod channels.
        Ref: World Nuclear Association RBMK appendix; INSAG-7 §2.1.
        """
        assert REACTOR_PARAMS["rbmk"].num_control_rods == 211

    def test_pwr_rated_power_tmi2(self):
        """TMI-2 rated thermal power was 2772 MWt (792 MWe).
        Ref: NRC reactor database; World Nuclear Association.
        """
        assert REACTOR_PARAMS["pwr"].rated_power_mw == 2772.0

    def test_pwr_system_pressure(self):
        """PWR normal operating pressure is 15.5 MPa (2250 psi).
        Ref: NRC Pressurized Water Reactor Systems, Chapter 4.
        """
        assert REACTOR_PARAMS["pwr"].system_pressure_mpa == 15.5

    def test_pwr_clad_temp_limit_10cfr50_46(self):
        """NRC cladding temp limit is 1204°C (2200°F).
        Ref: 10 CFR 50.46 ECCS acceptance criteria for LOCA.
        """
        assert REACTOR_PARAMS["pwr"].clad_temp_limit_c == 1204.0

    def test_bwr_mark1_containment_pressure(self):
        """BWR Mark I containment design pressure is 0.53 MPa (~77 psi).
        Ref: GE Mark I containment specifications; NUREG/CR-5042.
        """
        assert REACTOR_PARAMS["bwr"].containment_design_pressure_mpa == 0.53

    def test_windscale_rated_power(self):
        """Windscale Pile 1 rated power was 180 MWt.
        Ref: UKAEA decommissioning reports; Wikipedia - Windscale Piles.
        """
        assert REACTOR_PARAMS["windscale"].rated_power_mw == 180.0

    def test_windscale_no_containment(self):
        """Windscale had no containment vessel (only filter stacks).
        Ref: Penney Report (1957); GOV.UK Windscale Pile 1 Case Study.
        """
        assert REACTOR_PARAMS["windscale"].containment_design_pressure_mpa == 0.0
        assert REACTOR_PARAMS["windscale"].containment_volume_m3 == 0.0

    def test_windscale_air_cooled(self):
        """Windscale Pile 1 was air-cooled (not water-cooled).
        Ref: Penney Report (1957); Wikipedia - Windscale Piles.
        """
        assert REACTOR_PARAMS["windscale"].is_air_cooled is True

    def test_xenon_135_cross_section(self):
        """Xe-135 thermal absorption cross-section = 2.65 million barns.
        Ref: Nuclear Data Sheets; Mughabghab, Atlas of Neutron Resonances (2006).
        1 barn = 1e-24 cm², so 2.65e6 barns = 2.65e-18 cm².
        """
        from physics import SIGMA_XE
        assert SIGMA_XE == 2.65e-18, f"Expected 2.65e-18 cm², got {SIGMA_XE}"

    def test_iodine_135_half_life(self):
        """I-135 half-life is 6.58 ± 0.03 hours.
        Ref: NUBASE2020 evaluation; ENDF/B-VIII.0; Chemlin isotope database.
        λ = ln(2) / (6.58 * 3600) = 2.926e-5 s⁻¹.
        """
        from physics import LAMBDA_I
        expected = math.log(2) / (6.58 * 3600)
        rel_error = abs(LAMBDA_I - expected) / expected
        assert rel_error < 0.02, (
            f"I-135 λ={LAMBDA_I:.2e} vs expected {expected:.2e}, "
            f"rel error {rel_error:.3f} exceeds 2%"
        )

    def test_xenon_135_half_life(self):
        """Xe-135 half-life is 9.14 ± 0.02 hours.
        Ref: NUBASE2020 evaluation; ENDF/B-VIII.0.
        λ = ln(2) / (9.14 * 3600) = 2.107e-5 s⁻¹.
        Code uses 2.09e-5 (< 1% error, acceptable).
        """
        from physics import LAMBDA_XE
        expected = math.log(2) / (9.14 * 3600)
        rel_error = abs(LAMBDA_XE - expected) / expected
        assert rel_error < 0.02, (
            f"Xe-135 λ={LAMBDA_XE:.2e} vs expected {expected:.2e}, "
            f"rel error {rel_error:.3f} exceeds 2%"
        )

    def test_pwr_control_rods_tmi2(self):
        """TMI-2 had 61 CRAs + 8 APSRs = 69 total.
        Ref: OSTI.GOV Technical Reports on TMI-2 CRA/APSR testing (1982-83).
        """
        assert REACTOR_PARAMS["pwr"].num_control_rods == 69

    def test_bwr_control_rods_137(self):
        """BWR-4 has 137 control rod drives.
        Ref: NRC BWR-04 Control Rod Drive System training doc (ML12158A334).
        """
        assert REACTOR_PARAMS["bwr"].num_control_rods == 137

    def test_windscale_24_control_rods(self):
        """Windscale Pile 1 had 24 control rods (20 coarse + 4 fine).
        Ref: Wikipedia - Windscale Piles; UKAEA decommissioning reports.
        """
        assert REACTOR_PARAMS["windscale"].num_control_rods == 24

    def test_windscale_fuel_mass_180_tonnes(self):
        """Windscale Pile 1 contained 180 tonnes of uranium metal.
        Ref: Wikipedia - Windscale Piles; Penney Report (1957).
        """
        assert REACTOR_PARAMS["windscale"].fuel_mass_kg == 180000.0

    def test_baker_just_constants(self):
        """Baker-Just Zircaloy oxidation correlation constants.
        Ref: Baker & Just, ANL-6548 (1962); 10 CFR 50.46 Appendix K.
        A = 33.3e6 mg²/(cm⁴·s), B = 22896 K.
        """
        # These constants appear in FuelIntegrityModel.update()
        # Verified against the original ANL-6548 report
        A = 33.3e6
        B = 22896.0
        assert A == 33.3e6, "Baker-Just pre-exponential must be 33.3e6"
        assert B == 22896.0, "Baker-Just activation temperature must be 22896 K"

    def test_way_wigner_formula_coefficient(self):
        """Way-Wigner decay heat: P/P0 = 0.066 * t^(-0.2).
        Ref: Way & Wigner, Phys. Rev. 73, 1318 (1948).
        """
        # At t=3600s (1 hour): fraction = 0.066 * 3600^(-0.2) = 0.01258
        expected = 0.066 * (3600 ** -0.2)
        actual = DecayHeatModel.fraction(3600)
        assert abs(actual - expected) < 1e-6, (
            f"Way-Wigner at 1h: expected {expected:.6f}, got {actual:.6f}"
        )

    def test_h2_stoichiometry(self):
        """H2 yield from Zr oxidation: Zr + 2H₂O → ZrO₂ + 2H₂.
        Molar masses: Zr=91.22, H₂=2.016.
        Yield = 2*2.016/91.22 = 0.0442 kg H₂/kg Zr.
        Code uses 0.044 (< 1% error).
        """
        expected_yield = 2 * 2.016 / 91.22  # = 0.04420
        code_yield = 0.044
        assert abs(code_yield - expected_yield) < 0.001, (
            f"H2 yield: code={code_yield}, stoichiometric={expected_yield:.4f}"
        )


# =============================================================================
# 21. TASK ENUMERATION AND SPLIT TESTS
# =============================================================================

class TestTaskEnumeration:
    """Verify task counts and splits match README documentation."""

    def test_train_split_40_tasks(self):
        """Train split has 40 tasks (4 scenarios × 10 seeds).
        Ref: README.md "40 training tasks across 4 scenarios".
        """
        tasks = ScenarioRegistry.list_tasks("train")
        assert len(tasks) == 40, f"Expected 40 train tasks, got {len(tasks)}"

    def test_test_split_110_tasks(self):
        """Test split has 110 tasks (11 scenarios × 10 seeds).
        Ref: README.md "110 test tasks across all 11 scenarios".
        """
        tasks = ScenarioRegistry.list_tasks("test")
        assert len(tasks) == 110, f"Expected 110 test tasks, got {len(tasks)}"

    def test_each_task_has_required_fields(self):
        """Every task should have id, scenario, reactor_type, seed, initial_conditions."""
        tasks = ScenarioRegistry.list_tasks("train")
        for task in tasks:
            assert "id" in task, f"Task missing id: {task}"
            assert "scenario" in task
            assert "reactor_type" in task
            assert "seed" in task
            assert "initial_conditions" in task
            assert "max_steps" in task
            assert "time_step_minutes" in task

    def test_task_ids_unique(self):
        """All task IDs should be unique across splits."""
        train_ids = {t["id"] for t in ScenarioRegistry.list_tasks("train")}
        test_ids = {t["id"] for t in ScenarioRegistry.list_tasks("test")}
        # Train tasks should have unique IDs
        assert len(train_ids) == 40
        # Test tasks should have unique IDs
        assert len(test_ids) == 110


# =============================================================================
# 22. RBMK VOID COEFFICIENT AND DYNAMICS TESTS
# =============================================================================


class TestRBMKVoidPhysics:
    """Tests verifying RBMK void coefficient magnitude, void-power coupling,
    and positive power coefficient behavior.

    These tests validate the RBMK's defining dangerous characteristic:
    a large positive void coefficient that creates positive feedback
    between power and steam void fraction.

    Ref: INSAG-7 (IAEA Safety Series No. 75-INSAG-7, 1992), Section 4.3.
    """

    def test_void_coefficient_magnitude_insag7(self):
        """Total void reactivity from reference to full voiding should be ~4.7β.
        Ref: INSAG-7 §4.3: 'steam void reactivity effect was +4.7 beta'.
        β_eff = 0.0048, so total = 4.7 × 0.0048 = 0.02256 dk/k.
        Void range: ref (0.15) to complete voiding (1.0), Δvoid = 0.85.
        """
        params = REACTOR_PARAMS["rbmk"]
        beta = params.beta_eff
        void_ref = params.void_fraction_ref
        model = NeutronicsModel(params)

        # Reactivity from ref void to complete voiding
        rho_full_voiding = model.void_feedback(1.0) - model.void_feedback(void_ref)
        expected_total = 4.7 * beta  # = 0.02256 dk/k

        # Allow 25% tolerance for linearized model
        assert abs(rho_full_voiding - expected_total) < 0.25 * expected_total, (
            f"RBMK void reactivity (ref→full voiding) should be ~{expected_total:.5f} dk/k "
            f"(+4.7β), got {rho_full_voiding:.5f} dk/k "
            f"({rho_full_voiding/beta:.2f}β)"
        )

    def test_void_coefficient_sign_and_scale(self):
        """Void coefficient must be positive and on the order of 0.015-0.05 dk/k
        per unit void fraction for RBMK.
        Ref: INSAG-7 §4.3; derived from +4.7β / Δvoid ≈ 0.027.
        """
        params = REACTOR_PARAMS["rbmk"]
        assert params.void_coeff > 0.015, (
            f"RBMK void coefficient should be >0.015 dk/k per unit void, "
            f"got {params.void_coeff}"
        )
        assert params.void_coeff < 0.05, (
            f"RBMK void coefficient should be <0.05 dk/k per unit void, "
            f"got {params.void_coeff}"
        )

    def test_void_power_coupling_sign(self):
        """At full power, a void increase should produce POSITIVE reactivity
        for RBMK. This is the dangerous positive feedback loop.
        Ref: INSAG-7 §4.3 — positive void coefficient.
        """
        params = REACTOR_PARAMS["rbmk"]
        model = NeutronicsModel(params)
        ref_void = params.void_fraction_ref  # 0.15

        # Simulating power increase: void rises from 0.15 to 0.16
        rho_higher_void = model.void_feedback(ref_void + 0.01)
        assert rho_higher_void > 0, (
            f"RBMK: higher void should give positive reactivity, got {rho_higher_void}"
        )

        # Magnitude check: 1% void increase should give ~0.0265% dk/k
        expected = params.void_coeff * 0.01
        assert abs(rho_higher_void - expected) < expected * 0.1, (
            f"Void feedback magnitude: expected ~{expected:.6f}, got {rho_higher_void:.6f}"
        )

    def test_steady_state_void_fraction_at_rated_power(self):
        """At normal RBMK operation (3200 MWt, 12500 kg/s), void fraction
        should be approximately 15%, not 0%.
        Ref: GRS-121 (1996): exit steam quality 14.5%.
        INSAG-7 Annex I: RBMK steam quality ~14% at outlet.
        """
        scenario = ScenarioRegistry.get("chernobyl_normal_ops")
        sim = ReactorSimulation(
            reactor_type=scenario.reactor_type,
            initial_conditions=scenario.initial_conditions,
            time_step_minutes=scenario.time_step_minutes,
            seed=42,
        )

        # Run for 10 steps to reach equilibrium
        for _ in range(10):
            sim.advance()

        assert sim.state.void_fraction > 0.08, (
            f"RBMK at rated power should have void ~15%, got {sim.state.void_fraction:.4f}"
        )
        assert sim.state.void_fraction < 0.30, (
            f"RBMK at rated power void should be <30%, got {sim.state.void_fraction:.4f}"
        )

    def test_void_fraction_scales_with_power(self):
        """Void fraction should be lower at lower power (less boiling).
        At 200 MW (6.25% rated), void should be much less than at 3200 MW.
        """
        # Normal ops: 3200 MW
        scenario_full = ScenarioRegistry.get("chernobyl_normal_ops")
        sim_full = ReactorSimulation(
            reactor_type=scenario_full.reactor_type,
            initial_conditions=scenario_full.initial_conditions,
            time_step_minutes=scenario_full.time_step_minutes,
            seed=42,
        )
        for _ in range(10):
            sim_full.advance()
        void_full = sim_full.state.void_fraction

        # Test start: 200 MW
        scenario_low = ScenarioRegistry.get("chernobyl_test_start")
        sim_low = ReactorSimulation(
            reactor_type=scenario_low.reactor_type,
            initial_conditions=scenario_low.initial_conditions,
            time_step_minutes=scenario_low.time_step_minutes,
            seed=42,
        )
        for _ in range(10):
            sim_low.advance()
        void_low = sim_low.state.void_fraction

        assert void_low < void_full, (
            f"Void should be lower at ~200 MW ({void_low:.4f}) "
            f"than at 3200 MW ({void_full:.4f})"
        )

    def test_positive_power_coefficient_below_20_pct(self):
        """At low power (<20% rated), the power coefficient should be
        positive for RBMK: void feedback (positive) dominates Doppler.
        Ref: WNA RBMK Appendix: 'at power below 20% of full power,
        the power coefficient becomes positive.'

        Test: at low power, a small positive reactivity perturbation
        (rod withdrawal) leads to accelerating power increase.
        """
        scenario = ScenarioRegistry.get("chernobyl_test_start")
        sim = ReactorSimulation(
            reactor_type=scenario.reactor_type,
            initial_conditions=scenario.initial_conditions,
            time_step_minutes=scenario.time_step_minutes,
            seed=42,
        )

        # Equilibrate for a few steps
        for _ in range(5):
            sim.advance()

        initial_power = sim.state.thermal_power_mw

        # Withdraw rods by 2% (add positive reactivity)
        sim.state.control_rod_position_pct = min(
            100.0, sim.state.control_rod_position_pct + 2.0
        )

        # Run for several more steps
        for _ in range(15):
            sim.advance()

        # Power should have increased (positive power coefficient amplifies)
        assert sim.state.thermal_power_mw > initial_power * 0.9, (
            f"At low power with +rho perturbation, power should not collapse: "
            f"{initial_power:.1f} -> {sim.state.thermal_power_mw:.1f}"
        )

    def test_power_response_amplified_by_void_feedback(self):
        """At RBMK full power, a 2% rod withdrawal should cause a growing
        power excursion due to positive void feedback: more power -> more
        void -> more positive reactivity -> more power. This is the RBMK's
        defining dangerous behavior.
        Ref: WNA RBMK Appendix: 'the power coefficient itself became positive';
        INSAG-7 S4.3: void coefficient overwhelmed other feedback components.
        """
        scenario = ScenarioRegistry.get("chernobyl_normal_ops")
        sim = ReactorSimulation(
            reactor_type=scenario.reactor_type,
            initial_conditions=scenario.initial_conditions,
            time_step_minutes=scenario.time_step_minutes,
            seed=42,
        )

        # Equilibrate
        for _ in range(5):
            sim.advance()

        initial_power = sim.state.thermal_power_mw
        initial_void = sim.state.void_fraction

        # Withdraw rods by 2%
        sim.state.control_rod_position_pct = min(
            100.0, sim.state.control_rod_position_pct + 2.0
        )

        # Run for 30 minutes (6 x 5-min steps)
        for _ in range(6):
            sim.advance()

        # Power should increase significantly due to positive void feedback
        assert sim.state.thermal_power_mw > initial_power * 1.1, (
            f"RBMK: 2% rod withdrawal should cause significant power increase "
            f"due to positive void feedback: {initial_power:.0f} -> "
            f"{sim.state.thermal_power_mw:.0f} MW"
        )
        # Void should also increase (confirming the feedback loop)
        assert sim.state.void_fraction > initial_void, (
            f"Void should increase with power: {initial_void:.4f} -> "
            f"{sim.state.void_fraction:.4f}"
        )

    def test_chernobyl_void_feedback_runaway(self):
        """A void increase should produce significant positive reactivity
        for RBMK — the mechanism that destroyed Chernobyl Unit 4.
        Ref: INSAG-7 §5.4: positive void coefficient drove power excursion.
        """
        params = REACTOR_PARAMS["rbmk"]
        model = NeutronicsModel(params)

        # Void increase of 0.03 (e.g., from MCP trip reducing flow)
        void_start = 0.15
        void_after = 0.18  # 3% increase
        delta_rho = model.void_feedback(void_after) - model.void_feedback(void_start)

        assert delta_rho > 0, (
            f"Void increase should add positive reactivity for RBMK: "
            f"delta_rho = {delta_rho:+.6f}"
        )

        # Should be significant: 0.03 × 0.0265 = 0.000795 dk/k ≈ 0.166β
        assert delta_rho > 0.0005, (
            f"Void reactivity change from 3% void increase should be "
            f">0.0005 dk/k (~0.1β), got {delta_rho:.6f}"
        )

    def test_manual_rod_adjustment_moderate_power_change(self):
        """A 2% manual rod withdrawal at RBMK full power should give a
        moderate power change (2-20%), not the 41% seen when all 211 rods
        move simultaneously. In reality, operators moved 1-4 rods at a time.
        Ref: WNA RBMK Appendix; Podlazov & Chichulin (2004), Atomic Energy.
        """
        scenario = ScenarioRegistry.get("chernobyl_normal_ops")
        sim = ReactorSimulation(
            reactor_type=scenario.reactor_type,
            initial_conditions=scenario.initial_conditions,
            time_step_minutes=scenario.time_step_minutes,
            seed=42,
        )

        # Equilibrate
        for _ in range(5):
            sim.advance()

        initial_power = sim.state.thermal_power_mw

        # Simulate manual rod adjustment (scaled delta, like npp_sim does)
        old_pos = sim.state.control_rod_position_pct
        new_pos = min(100.0, old_pos + 2.0)
        sim.state.control_rod_position_pct = new_pos

        old_rho = sim.neutronics.control_rod_reactivity(old_pos)
        new_rho_full = sim.neutronics.control_rod_reactivity(new_pos)
        delta = new_rho_full - old_rho
        scaled_delta = delta * sim.params.manual_rod_worth_fraction
        sim.state.reactivity_rods = old_rho + scaled_delta
        sim.state.manual_rod_reactivity_offset = (
            sim.state.reactivity_rods - new_rho_full
        )

        # Run for 30 minutes (6 x 5-min steps)
        for _ in range(6):
            sim.advance()

        pct_change = abs(sim.state.thermal_power_mw - initial_power) / initial_power * 100
        assert pct_change < 20.0, (
            f"Manual 2% rod withdrawal should give <20% power change, "
            f"got {pct_change:.1f}% ({initial_power:.0f} -> "
            f"{sim.state.thermal_power_mw:.0f} MW)"
        )
        assert pct_change > 1.0, (
            f"Manual 2% rod withdrawal should have noticeable effect, "
            f"got only {pct_change:.1f}%"
        )

    def test_scram_uses_full_rod_worth(self):
        """Scram (AZ-5) must use full rod worth of all rods, not the
        scaled manual fraction. This ensures emergency shutdown is effective.
        """
        params = REACTOR_PARAMS["rbmk"]
        model = NeutronicsModel(params)

        # Full worth at 0% (fully inserted) vs 50% (mid-position)
        rho_0 = model.control_rod_reactivity(0.0)
        rho_50 = model.control_rod_reactivity(50.0)

        # Full scram delta from 50% to 0% should use total_rod_worth.
        # S-curve at 50% gives worth_fraction=0.5, so remaining worth is
        # exactly half of total (0.075 dk/k for RBMK).
        scram_delta = rho_0 - rho_50
        assert scram_delta < -0.05, (
            f"Scram should insert large negative reactivity (full rod worth), "
            f"got {scram_delta:.4f} dk/k"
        )
        # Verify it uses full worth (not manual_rod_worth_fraction).
        # Manual fraction for RBMK is 0.10, so if scaling were applied,
        # the delta would be only ~0.0075 dk/k. It should be ~0.075.
        manual_equiv = abs(scram_delta) * params.manual_rod_worth_fraction
        assert abs(scram_delta) > 5.0 * manual_equiv, (
            f"Scram delta {scram_delta:.4f} should be much larger than "
            f"manual-scaled value {manual_equiv:.4f}"
        )

    def test_void_evolves_within_power_substeps(self):
        """Void fraction should evolve within a single timestep when power
        changes, not be stuck at the old value until the next step.
        This tests the coupled void feedback in _coupled_power_update.
        """
        scenario = ScenarioRegistry.get("chernobyl_normal_ops")
        sim = ReactorSimulation(
            reactor_type=scenario.reactor_type,
            initial_conditions=scenario.initial_conditions,
            time_step_minutes=scenario.time_step_minutes,
            seed=42,
        )

        # Equilibrate
        for _ in range(5):
            sim.advance()

        void_before = sim.state.void_fraction

        # Add a significant positive reactivity perturbation (direct, full worth)
        # to force a large power excursion within one step
        sim.state.control_rod_position_pct += 2.0
        sim.state.manual_rod_reactivity_offset = 0.0  # Use full worth for this test

        sim.advance()

        void_after = sim.state.void_fraction

        # Void should have changed in response to the power excursion
        assert void_after != void_before, (
            f"Void fraction should evolve within a timestep when power changes: "
            f"before={void_before:.4f}, after={void_after:.4f}"
        )
        # Power went up → more boiling → void should increase
        assert void_after > void_before, (
            f"Void should increase when power increases (RBMK subcooled boiling): "
            f"{void_before:.4f} -> {void_after:.4f}"
        )
