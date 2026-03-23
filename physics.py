"""Physics models for nuclear reactor simulation.

Implements coupled neutronics, thermal-hydraulics, xenon dynamics,
decay heat, fuel integrity, containment, and Wigner energy models.
All equations are simplified but realistic, based on standard nuclear
engineering references and real reactor parameters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


# ===========================================================================
# Reactor parameters by type
# ===========================================================================

@dataclass(frozen=True)
class ReactorParams:
    """Physical parameters for a specific reactor type."""
    rated_power_mw: float          # Nominal thermal power (MWt)
    beta_eff: float                # Effective delayed neutron fraction
    neutron_gen_time: float        # Prompt neutron generation time (seconds)
    lambda_d: float                # Delayed neutron precursor decay constant (1/s)

    # Reactivity coefficients
    doppler_coeff: float           # dk/k per K of fuel temperature change
    void_coeff: float              # dk/k per unit void fraction change
    moderator_temp_coeff: float    # dk/k per K of moderator temperature change

    # Reference temperatures for feedback calculation
    fuel_temp_ref: float           # Reference fuel temperature (C)
    coolant_temp_ref: float        # Reference coolant temperature (C)
    void_fraction_ref: float       # Reference void fraction

    # Thermal parameters
    coolant_inlet_temp_c: float    # Design inlet temperature (C)
    coolant_outlet_temp_c: float   # Design outlet temperature (C)
    system_pressure_mpa: float     # Normal operating pressure (MPa)
    coolant_mass_kg: float         # Total coolant inventory (kg)
    coolant_cp: float              # Specific heat of coolant (J/kg/K)

    # Fuel parameters
    fuel_mass_kg: float            # Total fuel mass (kg)
    cladding_mass_kg: float        # Total cladding mass (kg, for H2 generation)
    fuel_cp: float                 # Fuel specific heat (J/kg/K)

    # Control rod parameters
    total_rod_worth: float         # Total control rod worth (dk/k)
    rod_speed_pct_per_s: float     # Rod insertion speed (% per second)
    num_control_rods: int          # Total number of control rods
    has_graphite_tip: bool         # RBMK graphite displacer flaw
    graphite_tip_worth: float      # Positive reactivity from graphite tip (dk/k)

    # Safety limits
    clad_temp_limit_c: float       # Regulatory cladding temperature limit
    containment_design_pressure_mpa: float  # Containment design pressure
    containment_volume_m3: float   # Free volume of containment

    # Reactor-specific
    is_bwr: bool                   # Boiling water reactor (direct cycle)
    has_pressurizer: bool          # PWR pressurizer
    is_air_cooled: bool            # Windscale air-cooled


REACTOR_PARAMS: dict[str, ReactorParams] = {
    "rbmk": ReactorParams(
        rated_power_mw=3200.0,
        beta_eff=0.0048,
        neutron_gen_time=5.0e-4,  # Graphite-moderated (~1ms) with water channels (~0.1ms)
        lambda_d=0.0767,
        doppler_coeff=-1.2e-5,
        void_coeff=4.7e-4,           # POSITIVE! Key to Chernobyl (INSAG-7 §4.3: +4.7β)
        moderator_temp_coeff=-0.5e-5,
        fuel_temp_ref=1200.0,
        coolant_temp_ref=270.0,
        void_fraction_ref=0.15,
        coolant_inlet_temp_c=270.0,
        coolant_outlet_temp_c=284.0,
        system_pressure_mpa=7.0,
        coolant_mass_kg=300000.0,
        # Effective Cp for RBMK two-phase flow: accounts for both sensible
        # heating (270→284°C) and latent heat of steam generation.
        # Q=3200MW, m_dot=12500kg/s, deltaT=14°C → Cp_eff ≈ 18300 J/kg/K.
        # Ref: RBMK channel thermal-hydraulics, steam quality ~14% at outlet.
        coolant_cp=18300.0,
        fuel_mass_kg=192000.0,
        cladding_mass_kg=30000.0,
        fuel_cp=300.0,
        total_rod_worth=0.15,
        rod_speed_pct_per_s=5.0,     # 0.4 m/s over ~7m => 5.7%/s, ~18s full travel
        num_control_rods=211,
        has_graphite_tip=True,
        graphite_tip_worth=0.005,     # +0.5 beta positive reactivity per rod group
        clad_temp_limit_c=700.0,
        containment_design_pressure_mpa=0.45,
        containment_volume_m3=5000.0,  # Minimal
        is_bwr=False,
        has_pressurizer=False,
        is_air_cooled=False,
    ),
    "pwr": ReactorParams(
        rated_power_mw=2772.0,
        beta_eff=0.0065,
        neutron_gen_time=2.0e-5,
        lambda_d=0.0767,
        doppler_coeff=-2.5e-5,
        void_coeff=-1.5e-3,           # Negative (safe)
        moderator_temp_coeff=-3.0e-5,
        fuel_temp_ref=800.0,
        coolant_temp_ref=300.0,
        void_fraction_ref=0.0,
        coolant_inlet_temp_c=285.0,    # B&W OTSG plant; generic PWR ~275°C
        coolant_outlet_temp_c=325.0,   # B&W OTSG plant; generic PWR ~315°C
        system_pressure_mpa=15.5,      # NRC: ~155 bar / 2250 psi
        coolant_mass_kg=250000.0,
        coolant_cp=5500.0,
        fuel_mass_kg=100000.0,
        cladding_mass_kg=25000.0,
        fuel_cp=300.0,
        total_rod_worth=0.10,
        rod_speed_pct_per_s=50.0,     # ~2 seconds for full insertion
        num_control_rods=69,
        has_graphite_tip=False,
        graphite_tip_worth=0.0,
        clad_temp_limit_c=1204.0,     # 10 CFR 50.46
        containment_design_pressure_mpa=0.41,
        containment_volume_m3=70000.0,  # Large dry containment
        is_bwr=False,
        has_pressurizer=True,
        is_air_cooled=False,
    ),
    "bwr": ReactorParams(
        rated_power_mw=2381.0,
        beta_eff=0.0054,
        neutron_gen_time=4.0e-5,
        lambda_d=0.0767,
        doppler_coeff=-2.0e-5,
        void_coeff=-1.2e-3,           # Negative (safe)
        moderator_temp_coeff=-1.0e-5,
        fuel_temp_ref=700.0,
        coolant_temp_ref=270.0,
        void_fraction_ref=0.3,        # BWR normally has significant voiding
        coolant_inlet_temp_c=215.0,
        coolant_outlet_temp_c=286.0,
        system_pressure_mpa=7.0,
        coolant_mass_kg=200000.0,
        coolant_cp=5200.0,
        fuel_mass_kg=150000.0,
        cladding_mass_kg=40000.0,
        fuel_cp=300.0,
        total_rod_worth=0.12,
        rod_speed_pct_per_s=50.0,
        num_control_rods=137,
        has_graphite_tip=False,
        graphite_tip_worth=0.0,
        clad_temp_limit_c=1204.0,
        containment_design_pressure_mpa=0.53,
        containment_volume_m3=11000.0,  # Mark I: drywell + torus
        is_bwr=True,
        has_pressurizer=False,
        is_air_cooled=False,
    ),
    "windscale": ReactorParams(
        rated_power_mw=180.0,
        beta_eff=0.0064,          # Natural uranium, U-235 β≈0.0065 (ENDF/B-VIII)
        neutron_gen_time=8.0e-4,  # Pure graphite-moderated: ~1ms
        lambda_d=0.0767,
        doppler_coeff=-0.8e-5,
        void_coeff=0.0,               # Air-cooled, no void coefficient
        moderator_temp_coeff=-1.0e-5,
        fuel_temp_ref=200.0,
        coolant_temp_ref=20.0,
        void_fraction_ref=0.0,
        coolant_inlet_temp_c=20.0,
        coolant_outlet_temp_c=350.0,
        system_pressure_mpa=0.101,     # Atmospheric
        coolant_mass_kg=5000.0,        # Air mass in core
        coolant_cp=1005.0,             # Air Cp
        fuel_mass_kg=180000.0,         # 180 tonnes uranium
        cladding_mass_kg=5000.0,       # Magnox cladding
        fuel_cp=120.0,                 # Uranium metal
        total_rod_worth=0.08,
        rod_speed_pct_per_s=10.0,
        num_control_rods=24,
        has_graphite_tip=False,
        graphite_tip_worth=0.0,
        clad_temp_limit_c=400.0,       # Magnox melts around 650C
        containment_design_pressure_mpa=0.0,  # No containment!
        containment_volume_m3=0.0,
        is_bwr=False,
        has_pressurizer=False,
        is_air_cooled=True,
    ),
}


# ===========================================================================
# Neutronics Model
# ===========================================================================

class NeutronicsModel:
    """Point kinetics reactor model with reactivity feedback."""

    def __init__(self, params: ReactorParams) -> None:
        self.p = params

    def update_power(
        self,
        dt: float,
        power_mw: float,
        neutron_pop: float,
        precursor_conc: float,
        reactivity: float,
    ) -> tuple[float, float, float]:
        """Advance power by dt seconds using point kinetics with RK4.

        Returns (new_power, new_neutron_pop, new_precursor_conc).
        """
        beta = self.p.beta_eff
        lam = self.p.neutron_gen_time
        lam_d = self.p.lambda_d
        rho = reactivity

        # Sub-stepping: the point kinetics system has a fast eigenvalue
        # |(rho-beta)/Lambda| that governs the prompt neutron response.
        # RK4 stability requires dt * |eigenvalue| < ~2.8.
        # Ref: Ott & Neuhold, "Introductory Nuclear Reactor Dynamics" (1985),
        # Ch. 2: prompt neutron eigenvalue is (rho-beta)/Lambda.
        stiff_eigenvalue = abs(rho - beta) / lam  # actual fast eigenvalue
        stiff_eigenvalue = max(stiff_eigenvalue, beta / lam)  # at least beta/Lambda
        max_stable_dt = 2.5 / max(stiff_eigenvalue, 1.0)  # Conservative limit
        n_substeps = max(1, int(dt / max_stable_dt) + 1)
        # Additional substeps for super-prompt-critical reactivity
        if rho > 0.5 * beta:
            n_substeps = max(n_substeps, int(dt / 0.001))
        elif rho > 0.1 * beta:
            n_substeps = max(n_substeps, int(dt / 0.01))

        sub_dt = dt / n_substeps
        n = neutron_pop
        c = precursor_conc

        for _ in range(n_substeps):
            # dn/dt = [(rho - beta) / Lambda] * n + lambda_d * C
            # dC/dt = (beta / Lambda) * n - lambda_d * C
            def dn(n_val: float, c_val: float) -> float:
                return ((rho - beta) / lam) * n_val + lam_d * c_val

            def dc(n_val: float, c_val: float) -> float:
                return (beta / lam) * n_val - lam_d * c_val

            # RK4
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

            # Clamp to prevent negative values
            n = max(0.0, n)
            c = max(0.0, c)

        new_power = n * self.p.rated_power_mw
        return new_power, n, c

    def doppler_feedback(self, fuel_temp: float) -> float:
        """Doppler reactivity feedback (always negative for temp increase)."""
        return self.p.doppler_coeff * (fuel_temp - self.p.fuel_temp_ref)

    def void_feedback(self, void_fraction: float) -> float:
        """Void coefficient reactivity feedback.
        POSITIVE for RBMK, NEGATIVE for PWR/BWR.
        """
        return self.p.void_coeff * (void_fraction - self.p.void_fraction_ref)

    def moderator_temp_feedback(self, coolant_temp: float) -> float:
        """Moderator temperature coefficient feedback."""
        return self.p.moderator_temp_coeff * (coolant_temp - self.p.coolant_temp_ref)

    def control_rod_reactivity(
        self,
        position_pct: float,
        inserting_from_withdrawn: bool = False,
    ) -> float:
        """Calculate control rod reactivity at given position.

        position_pct: 0 = fully inserted, 100 = fully withdrawn
        inserting_from_withdrawn: True if rods are being inserted from near-fully-withdrawn
        """
        x = position_pct / 100.0  # 0 = fully in, 1 = fully out

        # Differential rod worth follows S-curve (integral is also S-shaped)
        # Integral worth: W(x) = x - sin(2*pi*x) / (2*pi)
        # This gives more worth at mid-travel, less at extremes
        worth_fraction = x - math.sin(2 * math.pi * x) / (2 * math.pi)

        # Base reactivity from absorber position
        absorber_rho = -self.p.total_rod_worth * (1.0 - worth_fraction)

        # RBMK graphite-tip flaw
        if self.p.has_graphite_tip and inserting_from_withdrawn:
            insertion_depth = 1.0 - x  # 0 = fully out, 1 = fully in
            if insertion_depth < 0.17:
                # Graphite tip entering bottom of core displaces water
                tip_effect = self.p.graphite_tip_worth * (insertion_depth / 0.17)
                return absorber_rho + tip_effect
            else:
                # Tip effect plateaus then absorber dominates
                return absorber_rho + self.p.graphite_tip_worth * max(0, 1.0 - (insertion_depth - 0.17) / 0.15)

        return absorber_rho


# ===========================================================================
# Xenon Dynamics
# ===========================================================================

# Physical constants for Xe-135 / I-135
LAMBDA_I = 2.93e-5       # I-135 decay constant (1/s), t½ = 6.58 hr (NUBASE2020; ENDF/B-VIII)
LAMBDA_XE = 2.09e-5      # Xe-135 decay constant (1/s), t½ ≈ 9.14 hr (NUBASE2020); λ gives 9.21h
GAMMA_I = 0.061           # I-135 cumulative fission yield
GAMMA_XE = 0.003          # Xe-135 direct fission yield
SIGMA_XE = 2.65e-18       # Xe-135 microscopic absorption cross-section (cm²)


class XenonDynamics:
    """Iodine-135 / Xenon-135 dynamics model.

    Concentrations are normalized: 1.0 = equilibrium at rated power.
    """

    # Xenon equilibrium reactivity worth varies by reactor type.
    # RBMK: ~-0.025 dk/k (Ref: INSAG-7 §5.2, Chernobyl RBMK analysis).
    # PWR/BWR: ~-0.03 dk/k (Ref: Duderstadt & Hamilton, Nuclear Reactor
    # Analysis, 1976, §10.3; typical for enriched uranium LWR).
    XENON_WORTH_BY_TYPE: dict[str, float] = {
        "rbmk": -0.025,
        "pwr": -0.030,
        "bwr": -0.030,
        "windscale": -0.020,
    }

    def __init__(self, params: ReactorParams, reactor_type: str = "pwr") -> None:
        self.p = params
        # Normalized flux at rated power = 1.0
        self._rated_flux = 1.0
        # Select xenon equilibrium worth for this reactor type
        self._xenon_worth_at_eq = self.XENON_WORTH_BY_TYPE.get(reactor_type, -0.028)

    # Burnup-to-decay ratio: R = sigma_Xe * phi_0 / lambda_Xe
    # R ≈ 7 for typical LWR at rated power, giving xenon peak at ~10h
    # and ~2.8x equilibrium after shutdown.
    BURNUP_RATIO = 7.0

    def update(
        self,
        dt: float,
        power_fraction: float,
        iodine: float,
        xenon: float,
    ) -> tuple[float, float]:
        """Advance I-135 and Xe-135 concentrations by dt seconds.

        power_fraction: current power / rated power (0-1+)
        iodine, xenon: normalized concentrations (1.0 = equilibrium at rated)

        Returns (new_iodine, new_xenon).

        Uses the proper normalized formulation where:
        - di/dt = lambda_I * (f - i)
        - dx/dt = lambda_Xe*(1+R) * [gi_frac*i + gx_frac*f - (1+R*f)/(1+R)*x]
        where R = sigma_Xe * phi_0 / lambda_Xe (burnup-to-decay ratio).
        At f=1, i=1, x=1 this gives dx/dt=0 (equilibrium).
        """
        f = power_fraction  # Normalized flux
        R = self.BURNUP_RATIO

        gi_frac = GAMMA_I / (GAMMA_I + GAMMA_XE)
        gx_frac = GAMMA_XE / (GAMMA_I + GAMMA_XE)
        coeff = LAMBDA_XE * (1.0 + R)

        n_substeps = max(1, int(dt / 30.0))
        sub_dt = dt / n_substeps

        i_val = iodine
        x_val = xenon

        for _ in range(n_substeps):
            def di_dt(i_v: float) -> float:
                return LAMBDA_I * (f - i_v)

            def dx_dt(i_v: float, x_v: float) -> float:
                return coeff * (gi_frac * i_v + gx_frac * f - (1.0 + R * f) / (1.0 + R) * x_v)

            # RK4
            k1i = di_dt(i_val)
            k1x = dx_dt(i_val, x_val)

            k2i = di_dt(i_val + 0.5 * sub_dt * k1i)
            k2x = dx_dt(i_val + 0.5 * sub_dt * k1i, x_val + 0.5 * sub_dt * k1x)

            k3i = di_dt(i_val + 0.5 * sub_dt * k2i)
            k3x = dx_dt(i_val + 0.5 * sub_dt * k2i, x_val + 0.5 * sub_dt * k2x)

            k4i = di_dt(i_val + sub_dt * k3i)
            k4x = dx_dt(i_val + sub_dt * k3i, x_val + sub_dt * k3x)

            i_val += (sub_dt / 6.0) * (k1i + 2 * k2i + 2 * k3i + k4i)
            x_val += (sub_dt / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)

            i_val = max(0.0, i_val)
            x_val = max(0.0, x_val)

        return i_val, x_val

    def get_reactivity(self, xenon: float) -> float:
        """Convert normalized xenon concentration to reactivity.

        At equilibrium (xenon=1.0), xenon worth is about -0.025 dk/k for RBMK,
        -0.03 for PWR/BWR. Values vary by reactor type and are selected in
        __init__ from XENON_WORTH_BY_TYPE.
        """
        return self._xenon_worth_at_eq * xenon


# ===========================================================================
# Thermal-Hydraulics
# ===========================================================================

@dataclass
class ThermalResult:
    """Results from thermal-hydraulics calculation."""
    fuel_temp: float           # Average fuel temperature (C)
    fuel_centerline_temp: float  # Peak fuel centerline temperature (C)
    cladding_temp: float       # Cladding surface temperature (C)
    coolant_outlet_temp: float # Coolant outlet temperature (C)
    void_fraction: float       # Average void fraction
    coolant_pressure: float    # System pressure (MPa)


class ThermalHydraulicsModel:
    """Simplified 1D thermal-hydraulics model."""

    def __init__(self, params: ReactorParams) -> None:
        self.p = params

    def update(
        self,
        dt: float,
        power_mw: float,
        decay_heat_mw: float,
        coolant_flow_kg_s: float,
        coolant_inlet_temp: float,
        coolant_pressure: float,
        fuel_temp: float,
        cladding_temp: float,
        coolant_outlet_temp: float,
        void_fraction: float,
        leak_rate_kg_s: float = 0.0,
        injection_rate_kg_s: float = 0.0,
        injection_temp_c: float = 30.0,
        srv_flow_kg_s: float = 0.0,
        coolant_inventory_fraction: float = 1.0,
    ) -> ThermalResult:
        """Advance thermal state by dt seconds.

        Models core uncovery when coolant inventory is depleted. When
        the mixture level drops below the top of the active fuel, the
        uncovered portion is cooled only by steam (heat transfer degrades
        by ~100x), causing rapid temperature rise.

        Ref: TMI-2 core uncovery began at ~100 min when ~30% of RCS
        inventory was lost; cladding exceeded 1000°C by ~140 min.
        NUREG-0600; NUREG/CR-6849 (TMI-2 thermal-hydraulic analysis).
        """
        total_power = power_mw + decay_heat_mw  # MW
        total_power_w = total_power * 1e6  # Convert to watts

        # -- Core uncovery fraction --
        # The mixture level in the vessel is roughly proportional to coolant
        # inventory. Core uncovery begins when inventory drops below ~70%
        # (top of active fuel) and the core is fully uncovered below ~20%
        # (bottom of active fuel). At TMI-2, the core began uncovering after
        # losing ~30% of the ~300,000 kg RCS inventory (~100 min at 20 kg/s);
        # by ~140 min (~50% lost), roughly half the core was uncovered.
        # Ref: NUREG-0600 Section 2.3; Rogovin Report (NUREG/CR-1250).
        top_of_fuel_inv = 0.70   # Inventory fraction at which core starts uncovering
        bottom_of_fuel_inv = 0.20  # Inventory fraction at which core is fully uncovered
        if coolant_inventory_fraction < top_of_fuel_inv:
            uncovered_frac = min(
                1.0,
                (top_of_fuel_inv - coolant_inventory_fraction)
                / (top_of_fuel_inv - bottom_of_fuel_inv),
            )
        else:
            uncovered_frac = 0.0

        # -- Coolant temperature --
        effective_flow = max(coolant_flow_kg_s + injection_rate_kg_s, 0.1)  # Avoid division by zero
        if effective_flow > 1.0:
            # Energy balance: Q = m_dot * Cp * (T_out - T_in)
            delta_t = total_power_w / (effective_flow * self.p.coolant_cp)
            # Limit temperature rise to physical bounds
            delta_t = min(delta_t, 500.0)
            new_outlet_temp = coolant_inlet_temp + delta_t
        else:
            # Near-zero flow: temperature rises based on heat capacity of coolant inventory
            # Adiabatic heating
            heat_rate = total_power_w / (self.p.coolant_mass_kg * self.p.coolant_cp)
            new_outlet_temp = coolant_outlet_temp + heat_rate * dt

        # Blend toward new temperature (thermal inertia)
        tau = 30.0  # Thermal time constant (seconds)
        alpha_t = min(1.0, dt / tau)
        new_outlet_temp = coolant_outlet_temp + alpha_t * (new_outlet_temp - coolant_outlet_temp)

        # -- Fuel temperatures --
        power_fraction = total_power / max(self.p.rated_power_mw, 1.0)

        # Covered portion: fuel-to-coolant temperature difference is driven
        # by heat flux. At steady state: T_fuel = T_coolant + (T_ref - T_cool_ref) * P/P0.
        # This correctly gives fuel at coolant temperature when power is zero,
        # and fuel_temp_ref at rated power.
        # Ref: Todreas & Kazimi, "Nuclear Systems" Vol I (2012), Ch. 8.
        t_cool_avg = (new_outlet_temp + coolant_inlet_temp) / 2.0
        fuel_rise_at_rated = self.p.fuel_temp_ref - self.p.coolant_temp_ref
        target_fuel_covered = t_cool_avg + fuel_rise_at_rated * power_fraction
        if coolant_flow_kg_s < 10.0 and total_power > 0.1:
            # Loss of flow: fuel temperature rises rapidly (adiabatic)
            heat_rate_fuel = total_power_w / (self.p.fuel_mass_kg * self.p.fuel_cp)
            target_fuel_covered = fuel_temp + heat_rate_fuel * dt

        # Uncovered portion: steam-only cooling is ~100x worse than water.
        # The uncovered fuel absorbs its share of decay heat with negligible
        # heat removal, causing rapid temperature rise. Since the reward and
        # safety models care about peak cladding temperature (which occurs in
        # the uncovered region), we track the uncovered-region temperature
        # once uncovery exceeds a significant threshold.
        # Ref: NUREG/CR-5535 (RELAP5), heat transfer regime transition
        # from nucleate boiling to film boiling/steam cooling.
        if uncovered_frac > 0.01 and total_power > 0.1:
            # The uncovered fuel heats adiabatically — steam removes ~1%.
            # Heat is deposited uniformly in fuel, but only the uncovered
            # portion's heat is not removed by water. Net heat rate in the
            # uncovered region per unit mass is the same as the core-average
            # volumetric heat rate, since power density is roughly uniform.
            heat_rate_uncovered = (
                total_power_w * (1.0 - 0.01)  # 99% retained (steam removes 1%)
                / max(self.p.fuel_mass_kg * self.p.fuel_cp, 1.0)
            )
            target_fuel_uncovered = fuel_temp + heat_rate_uncovered * dt
        else:
            target_fuel_uncovered = target_fuel_covered

        # Once significant uncovery occurs, fuel temperature tracks the
        # uncovered (peak) region since that drives safety limits (10 CFR
        # 50.46 peak cladding temperature of 2200°F / 1204°C).
        if uncovered_frac > 0.05:
            target_fuel_temp = target_fuel_uncovered
        else:
            target_fuel_temp = target_fuel_covered

        # Fuel thermal time constant (~5-10 seconds for fuel pellet)
        tau_fuel = 8.0
        alpha_f = min(1.0, dt / tau_fuel)
        new_fuel_temp = fuel_temp + alpha_f * (target_fuel_temp - fuel_temp)

        # Cap fuel temperature based on fuel damage: once fuel has melted
        # (UO2 melting point ~2865°C, Ref: MATPRO/NUREG/CR-6150), molten
        # corium reaches quasi-equilibrium and temperature plateaus.
        # Scale max temp with damage fraction to model fuel relocation.
        uo2_melt_c = 2865.0
        if new_fuel_temp > uo2_melt_c:
            new_fuel_temp = uo2_melt_c

        # Centerline is ~1.5x average, capped at UO2 melting point
        new_centerline = min(new_fuel_temp * 1.5, uo2_melt_c)

        # -- Cladding temperature --
        # Covered portion: normal convective cooling
        if coolant_flow_kg_s > 10.0:
            # Good cooling: cladding tracks coolant with small offset from heat flux
            heat_flux_offset = total_power_w / max(1.0, coolant_flow_kg_s * 50.0)
            target_clad_covered = (
                (new_outlet_temp + coolant_inlet_temp) / 2.0
                + min(heat_flux_offset / 1000.0, 200.0)
            )
        else:
            # Poor cooling: cladding approaches fuel temperature
            target_clad_covered = new_fuel_temp * 0.7 + new_outlet_temp * 0.3

        # Uncovered portion: cladding approaches fuel temp (no water cooling).
        # At TMI-2, uncovered cladding reached >1100°C within ~40 min of
        # uncovery. Ref: NUREG/CR-6849, TMI-2 Vessel Investigation Project.
        target_clad_uncovered = new_fuel_temp * 0.85 + new_outlet_temp * 0.15

        # Weighted cladding target
        target_clad = (
            (1.0 - uncovered_frac) * target_clad_covered
            + uncovered_frac * target_clad_uncovered
        )

        tau_clad = 15.0
        alpha_c = min(1.0, dt / tau_clad)
        new_clad_temp = cladding_temp + alpha_c * (target_clad - cladding_temp)

        # Cap cladding temp at Zircaloy melting point (~1850°C, Ref: MATPRO/
        # NUREG/CR-6150 Zircaloy properties). Beyond this, cladding has fully
        # melted and relocated.
        zr_melt_c = 1850.0
        if new_clad_temp > zr_melt_c:
            new_clad_temp = zr_melt_c

        # -- Void fraction --
        if self.p.is_air_cooled:
            new_void = 0.0
        else:
            # Saturation temperature at current pressure
            t_sat = self._saturation_temp(coolant_pressure)
            if new_outlet_temp > t_sat and not self.p.has_pressurizer:
                # Subcooled/saturated boiling
                excess = new_outlet_temp - t_sat
                new_void = min(0.95, void_fraction + 0.001 * excess * dt / 60.0)
            elif new_outlet_temp > t_sat - 5 and self.p.is_bwr:
                # BWR operates near saturation
                excess = max(0, new_outlet_temp - (t_sat - 5))
                new_void = min(0.8, 0.1 + 0.005 * excess)
            else:
                # Subcooled: voids collapse
                new_void = max(0.0, void_fraction - 0.01 * dt / 60.0)

            # Flow-dependent: higher flow suppresses void
            if coolant_flow_kg_s > 100:
                flow_suppression = min(0.1, coolant_flow_kg_s / 100000.0)
                new_void = max(0.0, new_void - flow_suppression)

        # -- Pressure --
        new_pressure = coolant_pressure
        if not self.p.is_air_cooled:
            # Pressure changes from:
            # - Leak (decreases)
            # - Injection (increases slightly)
            # - Temperature change (increases with temp)
            # - SRV release (decreases)
            dp_leak = -leak_rate_kg_s * 0.0001 * dt  # Simplified
            dp_srv = -srv_flow_kg_s * 0.0001 * dt
            dp_temp = 0.0001 * (new_outlet_temp - coolant_outlet_temp)  # Thermal expansion
            dp_inject = injection_rate_kg_s * 0.00005 * dt
            new_pressure = max(0.1, coolant_pressure + dp_leak + dp_srv + dp_temp + dp_inject)

        return ThermalResult(
            fuel_temp=new_fuel_temp,
            fuel_centerline_temp=new_centerline,
            cladding_temp=new_clad_temp,
            coolant_outlet_temp=new_outlet_temp,
            void_fraction=new_void,
            coolant_pressure=new_pressure,
        )

    @staticmethod
    def _saturation_temp(pressure_mpa: float) -> float:
        """Approximate saturation temperature of water at given pressure.

        Uses piecewise linear interpolation of NIST/IAPWS-IF97 steam table
        reference values. Key points verified against NIST Chemistry WebBook
        (https://webbook.nist.gov/chemistry/fluid/):
          0.101 MPa → 100°C,  1.0 MPa → 180°C,  7.0 MPa → 285°C,
         10.0 MPa  → 311°C, 15.5 MPa → 345°C, 22.064 MPa → 374°C (critical)
        """
        if pressure_mpa <= 0.01:
            return 46.0
        # Piecewise linear interpolation between verified steam table points
        _TABLE = [
            (0.01, 46.0), (0.101, 100.0), (0.5, 152.0), (1.0, 180.0),
            (2.0, 212.0), (3.0, 234.0), (5.0, 264.0), (7.0, 285.0),
            (10.0, 311.0), (15.0, 342.0), (15.5, 345.0), (22.064, 374.0),
        ]
        if pressure_mpa >= _TABLE[-1][0]:
            return _TABLE[-1][1]
        for i in range(len(_TABLE) - 1):
            p0, t0 = _TABLE[i]
            p1, t1 = _TABLE[i + 1]
            if p0 <= pressure_mpa <= p1:
                frac = (pressure_mpa - p0) / (p1 - p0)
                return t0 + frac * (t1 - t0)
        return _TABLE[0][1]


# ===========================================================================
# Decay Heat
# ===========================================================================

class DecayHeatModel:
    """Decay heat calculation using the Way-Wigner approximation (1948).

    P_decay/P0 = 0.066 * t^(-0.2), from Way & Wigner, Phys. Rev. 73, 1318 (1948).
    This is a simplified fit; the full ANS/ANSI-5.1-2014 standard uses summation
    of 23 exponential groups but agrees within ~10% for t > 10s.
    """

    @staticmethod
    def calculate(time_since_shutdown_s: float, rated_power_mw: float) -> float:
        """Return decay heat in MW.

        Uses P_decay/P0 = 0.066 * t^(-0.2) for t > 0.
        """
        if time_since_shutdown_s <= 0:
            return 0.0
        t = max(time_since_shutdown_s, 0.1)  # Avoid singularity
        fraction = 0.066 * (t ** -0.2)
        fraction = min(fraction, 0.07)  # Cap at 7%
        return fraction * rated_power_mw

    @staticmethod
    def fraction(time_since_shutdown_s: float) -> float:
        """Return decay heat fraction (0-1) of rated power."""
        if time_since_shutdown_s <= 0:
            return 0.0
        t = max(time_since_shutdown_s, 0.1)
        return min(0.07, 0.066 * (t ** -0.2))


# ===========================================================================
# Fuel Integrity
# ===========================================================================

@dataclass
class FuelIntegrityResult:
    """Results from fuel integrity calculation."""
    damage_fraction: float       # 0.0 = intact, 1.0 = complete meltdown
    oxidation_pct: float         # Zircaloy oxidation percentage
    hydrogen_kg: float           # Cumulative H2 generated
    hydrogen_rate_kg_s: float    # Current H2 generation rate


class FuelIntegrityModel:
    """Fuel damage and hydrogen generation model."""

    def __init__(self, params: ReactorParams) -> None:
        self.p = params

    def update(
        self,
        dt: float,
        cladding_temp_c: float,
        fuel_centerline_temp_c: float,
        current_damage: float,
        current_oxidation_pct: float,
        current_h2_kg: float,
    ) -> FuelIntegrityResult:
        """Advance fuel integrity state by dt seconds."""
        t_clad_k = cladding_temp_c + 273.15

        # -- Zircaloy oxidation rate (Baker-Just correlation) --
        # Ref: Baker & Just, ANL-6548 (1962); codified in 10 CFR 50.46 Appendix K.
        # Original form: w² = A·exp(-B/T)·t, A=33.3e6 mg²/(cm⁴·s), B=22896 K.
        # Adapted here for simplified fractional oxidation with 50% cap per step.
        h2_rate = 0.0
        oxidation_rate = 0.0
        if cladding_temp_c > 1000 and t_clad_k > 0:
            # Baker-Just parabolic rate: A·exp(-B/T)
            reaction_rate = 33.3e6 * math.exp(-22896.0 / t_clad_k)
            # Convert to fractional oxidation:
            # Cladding wall thickness ~0.6mm = 6e-4 m, so delta_0^2 = 3.6e-7 m^2
            delta_0_sq = 3.6e-7  # m^2
            # Fraction of cladding oxidized this timestep
            oxidation_rate = (reaction_rate * dt / delta_0_sq) * 100.0  # percentage
            # Cap to avoid runaway at very high temps
            oxidation_rate = min(oxidation_rate, 50.0)

            # Hydrogen generation: ~0.044 kg H2 per kg Zr oxidized
            zr_oxidized_kg = (oxidation_rate / 100.0) * self.p.cladding_mass_kg
            h2_rate = zr_oxidized_kg * 0.044 / max(dt, 0.1)

        new_oxidation = min(100.0, current_oxidation_pct + oxidation_rate)
        new_h2 = current_h2_kg + h2_rate * dt

        # -- Damage progression --
        damage_rate = 0.0
        if cladding_temp_c > 1204:
            # Above NRC limit: progressive damage
            excess = cladding_temp_c - 1204
            damage_rate += 0.0005 * (excess / 500.0)
        if cladding_temp_c > 1480:
            # Autocatalytic oxidation: accelerating damage
            excess = cladding_temp_c - 1480
            damage_rate += 0.005 * (excess / 500.0)
        if fuel_centerline_temp_c > 2870:
            # UO2 melting: rapid damage
            damage_rate += 0.05

        new_damage = min(1.0, current_damage + damage_rate * dt / 60.0)

        return FuelIntegrityResult(
            damage_fraction=new_damage,
            oxidation_pct=new_oxidation,
            hydrogen_kg=new_h2,
            hydrogen_rate_kg_s=h2_rate,
        )


# ===========================================================================
# Containment
# ===========================================================================

@dataclass
class ContainmentResult:
    """Results from containment calculation."""
    pressure: float           # Containment pressure (MPa)
    temperature: float        # Containment temperature (C)
    hydrogen_pct: float       # H2 concentration (vol%)
    radiation_level: float    # Radiation level (Sv/hr)
    cumulative_release: float # Cumulative release to environment (TBq)


class ContainmentModel:
    """Containment pressure, hydrogen, and radiation model."""

    def __init__(self, params: ReactorParams) -> None:
        self.p = params

    def update(
        self,
        dt: float,
        pressure: float,
        temperature: float,
        hydrogen_pct: float,
        h2_production_rate: float,
        fuel_damage: float,
        venting: bool,
        vent_to_wetwell: bool = False,
        suppression_pool_temp: float = 30.0,
    ) -> ContainmentResult:
        """Advance containment state by dt seconds."""

        if self.p.containment_volume_m3 <= 0:
            # No containment (Windscale): direct release
            release_rate = 0.1 * fuel_damage  # TBq/s simplified
            return ContainmentResult(
                pressure=0.101,
                temperature=temperature,
                hydrogen_pct=0.0,  # Open to atmosphere
                radiation_level=fuel_damage * 100.0,
                cumulative_release=fuel_damage * release_rate * dt,
            )

        # -- Pressure --
        # Rises from steam and H2 accumulation
        # Steam from boiloff: roughly proportional to power that can't be removed
        dp_h2 = h2_production_rate * 0.001 * dt / self.p.containment_volume_m3

        # Temperature-driven pressure (ideal gas)
        if temperature > 100:
            dp_steam = 0.00001 * max(0, temperature - 100) * dt
        else:
            dp_steam = 0.0

        # Venting
        dp_vent = 0.0
        vent_release_fraction = 0.0
        if venting and pressure > 0.101:
            vent_rate = 0.1 * (pressure - 0.101)  # MPa/s
            dp_vent = -vent_rate * dt
            if vent_to_wetwell:
                vent_release_fraction = 0.01  # Wetwell scrubs 99% of fission products
            else:
                vent_release_fraction = 0.1  # Direct release

        new_pressure = max(0.101, pressure + dp_h2 + dp_steam + dp_vent)

        # -- Temperature --
        # Rises from decay heat deposited in containment atmosphere
        new_temp = temperature + fuel_damage * 0.1 * dt / 60.0
        if venting:
            new_temp = max(30.0, new_temp - 0.05 * dt / 60.0)

        # -- Hydrogen concentration --
        if self.p.containment_volume_m3 > 0:
            # Moles of H2 added
            h2_moles = (h2_production_rate * dt) / 0.002  # kg to moles
            # Total moles in containment (approximate)
            total_moles = (pressure * 1e6 * self.p.containment_volume_m3) / (8.314 * (temperature + 273.15))
            if total_moles > 0:
                new_h2_pct = hydrogen_pct + (h2_moles / total_moles) * 100.0
            else:
                new_h2_pct = hydrogen_pct
        else:
            new_h2_pct = 0.0

        if venting:
            # Venting also removes hydrogen
            new_h2_pct = max(0.0, new_h2_pct * (1.0 - 0.01 * dt))

        new_h2_pct = max(0.0, min(100.0, new_h2_pct))

        # -- Radiation --
        # Proportional to fuel damage and whether fission products are in containment
        radiation = fuel_damage * 50.0  # Sv/hr at 100% damage

        # -- Environmental release --
        release_rate = 0.0
        if venting:
            release_rate = fuel_damage * 0.5 * vent_release_fraction  # TBq/s
        # Leakage through containment (design leak rate ~0.1%/day)
        if pressure > self.p.containment_design_pressure_mpa:
            overpressure_ratio = pressure / max(self.p.containment_design_pressure_mpa, 0.1)
            release_rate += fuel_damage * 0.01 * overpressure_ratio

        cumulative_release = release_rate * dt

        return ContainmentResult(
            pressure=new_pressure,
            temperature=new_temp,
            hydrogen_pct=new_h2_pct,
            radiation_level=radiation,
            cumulative_release=cumulative_release,
        )


# ===========================================================================
# Wigner Energy (Windscale-specific)
# ===========================================================================

class WignerEnergyModel:
    """Wigner energy storage and release in graphite.

    Neutron bombardment displaces carbon atoms, storing potential energy.
    When heated above ~250°C, this energy releases exothermically,
    potentially causing runaway heating.
    """

    @staticmethod
    def update(
        dt: float,
        graphite_temp_c: float,
        stored_energy_j_per_kg: float,
        graphite_mass_kg: float = 1966000.0,  # Windscale Pile 1
        air_flow_kg_s: float = 200.0,
        air_inlet_temp_c: float = 20.0,
        water_injection_kg_s: float = 0.0,
    ) -> tuple[float, float, float]:
        """Advance Wigner energy state.

        Returns (new_graphite_temp, new_stored_energy, heat_release_mw).
        """
        # Release rate depends on temperature
        if graphite_temp_c < 250.0 or stored_energy_j_per_kg <= 0:
            release_rate_j_per_kg_s = 0.0
        else:
            # Exponential increase with temperature
            excess = graphite_temp_c - 250.0
            release_rate_j_per_kg_s = stored_energy_j_per_kg * 0.0001 * (1 + excess / 50.0)

        # Total heat release
        total_release_w = release_rate_j_per_kg_s * graphite_mass_kg
        total_release_mw = total_release_w / 1e6

        # Update stored energy
        new_stored = max(0.0, stored_energy_j_per_kg - release_rate_j_per_kg_s * dt)

        # Graphite temperature change
        graphite_cp = 710.0  # J/kg/K for graphite
        total_thermal_mass = graphite_mass_kg * graphite_cp

        # Heat sources: Wigner release + any nuclear heating
        heat_in = total_release_w

        # Heat removal: air cooling
        air_cp = 1005.0
        heat_out_air = air_flow_kg_s * air_cp * max(0, graphite_temp_c - air_inlet_temp_c)

        # Heat removal: water injection (very effective but risks)
        heat_out_water = 0.0
        if water_injection_kg_s > 0:
            # Water absorbs heat via boiling: ~2260 kJ/kg latent heat
            heat_out_water = water_injection_kg_s * 2260000.0

        # Net heat
        net_heat = heat_in - heat_out_air - heat_out_water

        # Temperature change
        dt_temp = (net_heat / total_thermal_mass) * dt
        new_temp = graphite_temp_c + dt_temp

        # Fire check: uranium ignites in air at ~300°C, graphite above ~700°C
        # This is modeled as accelerating damage in the fuel integrity model

        return new_temp, new_stored, total_release_mw


# ===========================================================================
# Pressurizer Model (PWR-specific)
# ===========================================================================

class PressurizerModel:
    """PWR pressurizer level and pressure model.

    Models the misleading level indication during loss-of-coolant:
    as pressure drops, water flashes to steam, and the expanding
    mixture causes the pressurizer level to RISE even though
    total coolant inventory is DECREASING.
    """

    @staticmethod
    def update(
        dt: float,
        level_pct: float,
        pressure_mpa: float,
        coolant_leak_rate: float,
        hpi_flow: float,
        system_temp: float,
    ) -> tuple[float, float]:
        """Update pressurizer level and pressure.

        Returns (new_level_pct, apparent_level_pct).
        The apparent level may differ from actual due to void formation.
        """
        # Actual level changes from leak and injection
        net_flow = hpi_flow - coolant_leak_rate  # kg/s
        # Simplified: level change proportional to net flow
        level_change = net_flow * 0.0001 * dt  # % per timestep

        # Void formation effect: as pressure drops, water flashes
        # This INCREASES apparent level even during LOCA
        if pressure_mpa < 12.0:
            void_swell = (15.5 - pressure_mpa) * 0.5  # % level increase from voids
        else:
            void_swell = 0.0

        actual_level = max(0.0, min(100.0, level_pct + level_change))
        apparent_level = min(100.0, actual_level + void_swell)

        return actual_level, apparent_level
