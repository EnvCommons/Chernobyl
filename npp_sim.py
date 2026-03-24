"""Nuclear Power Plant Simulator — OpenReward RL Environment.

A hyperrealistic nuclear power plant management environment where agents
must operate reactors and respond to crisis scenarios inspired by real
historical disasters: Chernobyl (1986), Three Mile Island (1979),
Fukushima Daiichi (2011), and Windscale (1957).
"""

from __future__ import annotations

import copy
import json
from typing import Any, List

from pydantic import BaseModel

from openreward.environments import Environment, JSONObject, TextBlock, ToolOutput, tool

from equipment import EquipmentStatus, Valve
from reactor import ReactorSimulation, ReactorState
from rewards import RewardCalculator
from scenarios import ScenarioRegistry


# =============================================================================
# Pydantic models for tool inputs
# =============================================================================

class TaskSpec(BaseModel):
    id: str
    scenario: str
    reactor_type: str
    difficulty: str
    initial_conditions: dict
    time_step_minutes: float
    max_steps: int
    target_outcome: str
    seed: int = 42


class ObserveInstrumentsParams(BaseModel, extra="forbid"):
    """Request current instrument readings. Does NOT advance simulation time."""
    systems: list[str] = []


class AdjustControlRodsParams(BaseModel, extra="forbid"):
    """Insert or withdraw control rods."""
    action: str
    value: float
    rod_group: str = "all"


class OperatePumpParams(BaseModel, extra="forbid"):
    """Start, stop, or adjust a coolant pump."""
    pump_id: str
    action: str
    speed_pct: float = 100.0


class OperateValveParams(BaseModel, extra="forbid"):
    """Open, close, or adjust a valve."""
    valve_id: str
    action: str
    position_pct: float = 100.0


class ActivateSystemParams(BaseModel, extra="forbid"):
    """Activate or deactivate a safety or auxiliary system."""
    system_id: str
    action: str


class InjectCoolantParams(BaseModel, extra="forbid"):
    """Emergency coolant injection."""
    source: str
    flow_rate_kg_s: float
    boron_ppm: float = 0.0


class OrderScramParams(BaseModel, extra="forbid"):
    """Emergency reactor shutdown (SCRAM / AZ-5)."""
    confirm: bool


class VentContainmentParams(BaseModel, extra="forbid"):
    """Vent containment pressure."""
    vent_path: str = "filtered"
    target_pressure_mpa: float = 0.0


class SubmitLogParams(BaseModel, extra="forbid"):
    """Document reasoning and observations. No simulation effect."""
    entry: str
    severity: str = "info"


class WaitParams(BaseModel, extra="forbid"):
    """Wait and monitor — advances simulation time without taking action."""
    duration_steps: int = 1


# =============================================================================
# Main Environment Class
# =============================================================================

class NuclearPlantEnvironment(Environment):
    """Nuclear power plant management RL environment.

    Simulates reactor operation across crisis scenarios inspired by
    Chernobyl, Three Mile Island, Fukushima, and Windscale.
    """

    def __init__(self, task_spec: JSONObject, secrets: dict[str, str] = {}) -> None:
        super().__init__(task_spec)
        self.config = TaskSpec.model_validate(task_spec)
        self.sim: ReactorSimulation | None = None
        self.reward_calc: RewardCalculator | None = None
        self.step_count = 0
        self.cumulative_reward = 0.0
        self.prev_state: ReactorState | None = None
        self.action_log: list[dict] = []
        self.time_advanced_this_turn = False

    async def setup(self) -> None:
        scenario = ScenarioRegistry.get(self.config.scenario)
        self.sim = ReactorSimulation(
            reactor_type=self.config.reactor_type,
            initial_conditions=self.config.initial_conditions,
            time_step_minutes=self.config.time_step_minutes,
            difficulty=self.config.difficulty,
            seed=self.config.seed,
        )
        self.reward_calc = RewardCalculator(
            reactor_type=self.config.reactor_type,
            target_outcome=self.config.target_outcome,
        )
        self.step_count = 0
        self.cumulative_reward = 0.0
        self.prev_state = copy.deepcopy(self.sim.state)
        self.action_log = []
        self.time_advanced_this_turn = False

    async def teardown(self) -> None:
        self.sim = None
        self.reward_calc = None

    async def get_prompt(self) -> List[TextBlock]:
        scenario = ScenarioRegistry.get(self.config.scenario)
        reactor_type_names = {
            "rbmk": "RBMK-1000 (Chernobyl-type)",
            "pwr": "PWR (Pressurized Water Reactor, TMI-type)",
            "bwr": "BWR (Boiling Water Reactor, Fukushima-type)",
            "windscale": "Windscale Pile (air-cooled graphite)",
        }
        prompt = f"""You are the senior reactor operator at a nuclear power plant. You must manage the reactor safely.

## Reactor Type
{reactor_type_names.get(self.config.reactor_type, self.config.reactor_type)}

## Scenario
{scenario.description}

## Your Objective
{self._objective_text()}

## Available Actions
You have the following tools to operate the plant:

1. **observe_instruments** — Read current instrument panel (FREE, no time advance). Call this first to understand the situation. Note: some instruments may be malfunctioning or misleading.
2. **adjust_control_rods** — Insert or withdraw control rods. Actions: "insert", "withdraw", "set_position". Value: change in % or absolute position (0=fully inserted, 100=fully withdrawn).
3. **operate_pump** — Start, stop, or adjust pumps. Actions: "start", "stop", "set_speed".
4. **operate_valve** — Open, close, or adjust valves. Actions: "open", "close", "set_position".
5. **activate_system** — Activate or deactivate safety systems. Actions: "activate", "deactivate".
6. **inject_coolant** — Emergency coolant injection from various sources.
7. **order_scram** — Emergency reactor shutdown (AZ-5 / SCRAM). Use with extreme caution.
8. **vent_containment** — Vent containment pressure. Paths: "filtered", "unfiltered", "wetwell".
9. **submit_log** — Document your reasoning (no effect on simulation).
10. **wait** — Advance time without taking action. Use when the reactor is stable and no intervention is needed.

## Critical Information
- Each action tool call (except observe_instruments and submit_log) advances the simulation by {self.config.time_step_minutes} minutes.
- You have a maximum of {self.config.max_steps} action steps.
- Instrument readings may be UNRELIABLE or MISLEADING — cross-reference multiple sources.
- You must continue taking actions throughout the scenario. If no operational action is needed, use the **wait** tool to advance time and continue monitoring.

## Physics Notes for {self.config.reactor_type.upper()}
{self._physics_notes()}

Begin by observing the instruments to assess the current situation."""

        return [TextBlock(text=prompt)]

    def _objective_text(self) -> str:
        if self.config.target_outcome == "stabilize":
            return (
                "Stabilize the reactor: bring cladding temperatures below safety limits, "
                "prevent fuel damage, contain radiation, and manage hydrogen. "
                "Maintain stability for at least 30 minutes to succeed."
            )
        elif self.config.target_outcome == "maintain_power":
            return (
                "Maintain stable power output at rated capacity while responding "
                "to perturbations. Avoid unnecessary shutdowns (economic penalty) "
                "but prioritize safety above all."
            )
        return "Manage the reactor safely."

    def _physics_notes(self) -> str:
        notes = {
            "rbmk": (
                "- POSITIVE void coefficient: steam voids INCREASE reactivity (dangerous!)\n"
                "- Below ~24% power, overall power coefficient is POSITIVE (unstable)\n"
                "- Control rods have graphite displacers: inserting from fully withdrawn "
                "position initially ADDS positive reactivity before the absorber enters\n"
                "- ORM (Operational Reactivity Margin) must stay above 15 equivalent rods; "
                "below 26 requires authorization\n"
                "- ECCS may be disabled — check equipment status\n"
                "- Xenon-135 buildup after power reduction can prevent restart for 24-48 hours"
            ),
            "pwr": (
                "- Negative void coefficient (self-stabilizing)\n"
                "- PORV (Power-Operated Relief Valve) and block valve are your pressure relief\n"
                "- CRITICAL: PORV indicator shows solenoid command, NOT actual valve position\n"
                "- Pressurizer level can be MISLEADING during depressurization (void swell)\n"
                "- HPI (High Pressure Injection) is your primary emergency cooling\n"
                "- Borated water injection adds negative reactivity"
            ),
            "bwr": (
                "- Negative void coefficient (self-stabilizing)\n"
                "- RCIC (steam-driven, no AC power needed) is your primary emergency cooling\n"
                "- SRVs must be opened to depressurize for low-pressure injection\n"
                "- SRV operation requires DC power from batteries\n"
                "- Suppression pool (torus) condenses steam but heats up over time\n"
                "- Mark I containment has limited volume — watch pressure carefully\n"
                "- Hydrogen can accumulate in reactor building if containment leaks"
            ),
            "windscale": (
                "- Air-cooled, graphite-moderated pile — NO containment building\n"
                "- Wigner energy stored in graphite requires careful annealing\n"
                "- Magnox fuel cladding melts at ~650°C\n"
                "- Uranium metal ignites in air at ~300°C\n"
                "- Blowers provide both cooling AND oxygen — a dilemma during fire\n"
                "- Water injection is effective but risks hydrogen from C + H2O reaction\n"
                "- Thermocouple coverage is sparse — blind spots exist"
            ),
        }
        return notes.get(self.config.reactor_type, "No specific notes available.")

    def _advance_time_and_get_output(self, action_name: str, action_detail: str) -> ToolOutput:
        """Advance simulation, calculate reward, check terminal conditions."""
        assert self.sim is not None
        assert self.reward_calc is not None
        assert self.prev_state is not None

        self.sim.advance()
        self.step_count += 1

        state = self.sim.state
        reward = self.reward_calc.step_reward(state, self.prev_state)
        self.cumulative_reward += reward

        is_terminal, reason = self.reward_calc.is_terminal(
            state, self.step_count, self.config.max_steps
        )

        if is_terminal:
            terminal_r = self.reward_calc.terminal_reward(state)
            reward += terminal_r
            self.cumulative_reward += terminal_r

        # Get readings for display
        readings = self.sim.get_instrument_readings()
        display_text = self.sim.format_readings(readings)

        # Action summary
        summary = f"\n--- ACTION: {action_name} ---\n{action_detail}\n"
        summary += f"Step: {self.step_count}/{self.config.max_steps}\n"
        summary += f"Step Reward: {reward:+.3f} | Cumulative: {self.cumulative_reward:+.3f}\n"

        if is_terminal:
            summary += f"\n*** EPISODE ENDED: {reason.upper().replace('_', ' ')} ***\n"
            if reason == "stabilized":
                summary += "Congratulations — reactor stabilized successfully!\n"
            elif reason == "core_meltdown":
                summary += "CATASTROPHIC FAILURE — Complete core meltdown.\n"
            elif reason == "hydrogen_detonation":
                summary += "CATASTROPHIC FAILURE — Hydrogen detonation in containment.\n"
            elif reason == "catastrophic_release":
                summary += "CATASTROPHIC FAILURE — Massive radiation release to environment.\n"
            elif reason == "max_steps_reached":
                summary += "Maximum time steps reached. Scenario ended.\n"

        # Log action
        self.action_log.append({
            "step": self.step_count,
            "action": action_name,
            "detail": action_detail,
            "reward": reward,
            "cumulative_reward": self.cumulative_reward,
            "terminal": is_terminal,
            "reason": reason if is_terminal else None,
        })

        # Update previous state
        self.prev_state = copy.deepcopy(state)

        breakdown = self.reward_calc.get_reward_breakdown(state, self.prev_state)

        return ToolOutput(
            metadata={
                "step": self.step_count,
                "reward": reward,
                "cumulative_reward": self.cumulative_reward,
                "terminal": is_terminal,
                "reason": reason if is_terminal else None,
                "reward_breakdown": breakdown,
            },
            blocks=[TextBlock(text=summary + "\n" + display_text)],
            reward=reward,
            finished=is_terminal,
        )

    # =========================================================================
    # Tools
    # =========================================================================

    @tool
    async def observe_instruments(self, params: ObserveInstrumentsParams) -> ToolOutput:
        """Read current instrument panels. Does NOT advance simulation time.
        Returns readings from functioning instruments (some may be malfunctioning).
        Systems filter: 'neutronics', 'thermal', 'containment', 'equipment', 'all'.
        """
        assert self.sim is not None

        readings = self.sim.get_instrument_readings()

        # Filter by requested systems
        if params.systems and "all" not in params.systems:
            filtered = {"time": readings["time"]}
            for sys in params.systems:
                if sys in readings:
                    filtered[sys] = readings[sys]
            readings = filtered

        display = self.sim.format_readings(readings)

        return ToolOutput(
            metadata=readings,
            blocks=[TextBlock(text=display)],
            reward=0.0,
            finished=False,
        )

    @tool
    async def adjust_control_rods(self, params: AdjustControlRodsParams) -> ToolOutput:
        """Insert or withdraw control rods. Actions: 'insert', 'withdraw', 'set_position'.
        Value: percentage change (for insert/withdraw) or absolute position (for set_position).
        Position: 0 = fully inserted, 100 = fully withdrawn.
        Rod groups: 'all', 'safety', 'manual', 'automatic', 'shortened_absorber'.
        """
        assert self.sim is not None
        s = self.sim.state

        old_pos = s.control_rod_position_pct

        if params.action == "insert":
            new_pos = max(0.0, s.control_rod_position_pct - abs(params.value))
        elif params.action == "withdraw":
            new_pos = min(100.0, s.control_rod_position_pct + abs(params.value))
        elif params.action == "set_position":
            new_pos = max(0.0, min(100.0, params.value))
        else:
            return ToolOutput(
                metadata={"error": f"Unknown action: {params.action}"},
                blocks=[TextBlock(text=f"Error: Unknown action '{params.action}'. Use 'insert', 'withdraw', or 'set_position'.")],
                reward=0.0,
                finished=False,
            )

        # Apply rod speed limitation
        max_change = self.sim.params.rod_speed_pct_per_s * self.sim.dt
        actual_change = min(abs(new_pos - old_pos), max_change)
        if new_pos < old_pos:
            s.control_rod_position_pct = old_pos - actual_change
        else:
            s.control_rod_position_pct = old_pos + actual_change

        # Update rod reactivity: scale the CHANGE by manual_rod_worth_fraction
        # to represent movement of a small rod group, not all rods.
        # Ref: RBMK operators moved 1-4 rods at a time (SIUR panel max: 4).
        inserting = s.control_rod_position_pct < old_pos
        old_rho = self.sim.neutronics.control_rod_reactivity(
            old_pos,
            inserting_from_withdrawn=inserting and old_pos > 80.0,
        )
        new_rho_full = self.sim.neutronics.control_rod_reactivity(
            s.control_rod_position_pct,
            inserting_from_withdrawn=inserting and old_pos > 80.0,
        )
        delta = new_rho_full - old_rho
        scaled_delta = delta * self.sim.params.manual_rod_worth_fraction
        s.reactivity_rods = old_rho + scaled_delta
        # Track offset so advance() preserves the manual scaling
        s.manual_rod_reactivity_offset = s.reactivity_rods - new_rho_full

        # Update ORM for RBMK
        if self.sim.reactor_type == "rbmk":
            s.orm_count = int(
                self.sim.params.num_control_rods * (1.0 - s.control_rod_position_pct / 100.0)
            )

        detail = (
            f"Rods {params.action}: {old_pos:.1f}% → {s.control_rod_position_pct:.1f}% "
            f"(requested {params.value}%, max rate limited to {actual_change:.1f}%)"
        )

        return self._advance_time_and_get_output("adjust_control_rods", detail)

    @tool
    async def operate_pump(self, params: OperatePumpParams) -> ToolOutput:
        """Start, stop, or adjust a coolant pump.
        pump_id: e.g. 'mcp_1', 'rcp_1', 'rcic', 'hpci', 'hpi_1'.
        action: 'start', 'stop', 'set_speed'.
        speed_pct: speed percentage (for 'set_speed' action).
        """
        assert self.sim is not None

        eq = self.sim.equipment.get(params.pump_id)
        if eq is None:
            return ToolOutput(
                metadata={"error": f"Unknown pump: {params.pump_id}"},
                blocks=[TextBlock(text=f"Error: No pump with ID '{params.pump_id}' exists in this reactor.")],
                reward=0.0,
                finished=False,
            )

        # Check AC power requirement
        ac_dependent = {"rcp_1", "rcp_2", "rcp_3", "rcp_4", "rcp_a", "rcp_b", "eccs_lp"}
        if params.pump_id in ac_dependent and not self.sim.equipment.has_ac_power():
            return ToolOutput(
                metadata={"error": "No AC power available"},
                blocks=[TextBlock(text=f"Error: {params.pump_id} requires AC power, which is unavailable.")],
                reward=0.0,
                finished=False,
            )

        if params.action == "start":
            ok, msg = eq.start()
        elif params.action == "stop":
            ok, msg = eq.stop()
        elif params.action == "set_speed":
            ok, msg = eq.set_speed(params.speed_pct)
        else:
            return ToolOutput(
                metadata={"error": f"Unknown action: {params.action}"},
                blocks=[TextBlock(text=f"Error: Unknown action '{params.action}'. Use 'start', 'stop', or 'set_speed'.")],
                reward=0.0,
                finished=False,
            )

        return self._advance_time_and_get_output("operate_pump", msg)

    @tool
    async def operate_valve(self, params: OperateValveParams) -> ToolOutput:
        """Open, close, or adjust a valve.
        valve_id: e.g. 'porv', 'block_valve', 'srv_1', 'srv_2', 'srv_3', 'msiv', 'containment_vent'.
        action: 'open', 'close', 'set_position'.
        position_pct: percentage open (for 'set_position' action).
        """
        assert self.sim is not None

        eq = self.sim.equipment.get(params.valve_id)
        if eq is None:
            return ToolOutput(
                metadata={"error": f"Unknown valve: {params.valve_id}"},
                blocks=[TextBlock(text=f"Error: No valve with ID '{params.valve_id}' exists in this reactor.")],
                reward=0.0,
                finished=False,
            )

        # SRVs require DC power
        if "srv" in params.valve_id and not self.sim.equipment.has_dc_power():
            return ToolOutput(
                metadata={"error": "No DC power for SRV solenoids"},
                blocks=[TextBlock(text=f"Error: {params.valve_id} solenoid requires DC power. Batteries depleted.")],
                reward=0.0,
                finished=False,
            )

        if isinstance(eq, Valve):
            if params.action == "open":
                ok, msg = eq.open()
            elif params.action == "close":
                ok, msg = eq.close()
            elif params.action == "set_position":
                ok, msg = eq.set_position(params.position_pct)
            else:
                return ToolOutput(
                    metadata={"error": f"Unknown action: {params.action}"},
                    blocks=[TextBlock(text=f"Error: Unknown action '{params.action}'.")],
                    reward=0.0,
                    finished=False,
                )
        else:
            # Treat non-Valve equipment as simple on/off
            if params.action == "open":
                ok, msg = eq.start()
            elif params.action == "close":
                ok, msg = eq.stop()
            else:
                return ToolOutput(
                    metadata={"error": f"{params.valve_id} is not a valve"},
                    blocks=[TextBlock(text=f"Error: {params.valve_id} cannot be operated as a valve.")],
                    reward=0.0,
                    finished=False,
                )

        return self._advance_time_and_get_output("operate_valve", msg)

    @tool
    async def activate_system(self, params: ActivateSystemParams) -> ToolOutput:
        """Activate or deactivate a safety or auxiliary system.
        system_id: e.g. 'eccs', 'rcic', 'hpci', 'diesel_1', 'fire_truck', 'water_injection'.
        action: 'activate' or 'deactivate'.
        """
        assert self.sim is not None

        eq = self.sim.equipment.get(params.system_id)
        if eq is None:
            return ToolOutput(
                metadata={"error": f"Unknown system: {params.system_id}"},
                blocks=[TextBlock(text=f"Error: No system with ID '{params.system_id}' exists.")],
                reward=0.0,
                finished=False,
            )

        if params.action == "activate":
            ok, msg = eq.start()
        elif params.action == "deactivate":
            ok, msg = eq.stop()
        else:
            return ToolOutput(
                metadata={"error": f"Unknown action: {params.action}"},
                blocks=[TextBlock(text=f"Error: Unknown action '{params.action}'. Use 'activate' or 'deactivate'.")],
                reward=0.0,
                finished=False,
            )

        return self._advance_time_and_get_output("activate_system", msg)

    @tool
    async def inject_coolant(self, params: InjectCoolantParams) -> ToolOutput:
        """Emergency coolant injection from various sources.
        source: 'borated_water', 'seawater', 'fire_truck', 'makeup_tank'.
        flow_rate_kg_s: desired injection flow rate.
        boron_ppm: boron concentration (for borated_water, adds negative reactivity).
        """
        assert self.sim is not None

        valid_sources = {"borated_water", "seawater", "fire_truck", "makeup_tank"}
        if params.source not in valid_sources:
            return ToolOutput(
                metadata={"error": f"Unknown source: {params.source}"},
                blocks=[TextBlock(text=f"Error: Unknown coolant source '{params.source}'. Available: {valid_sources}")],
                reward=0.0,
                finished=False,
            )

        # Check if source is available
        if params.source == "fire_truck":
            ft = self.sim.equipment.get("fire_truck")
            if ft is None or ft.status in (EquipmentStatus.UNAVAILABLE, EquipmentStatus.DESTROYED):
                return ToolOutput(
                    metadata={"error": "Fire truck not available"},
                    blocks=[TextBlock(text="Error: Fire truck is not available at this time.")],
                    reward=0.0,
                    finished=False,
                )
            # Fire truck can only inject at low pressure
            if self.sim.state.coolant_pressure_mpa > 1.5:
                return ToolOutput(
                    metadata={"error": "Pressure too high for fire truck"},
                    blocks=[TextBlock(text=f"Error: Reactor vessel pressure ({self.sim.state.coolant_pressure_mpa:.1f} MPa) is too high for fire truck injection. Must depressurize below ~1.5 MPa first.")],
                    reward=0.0,
                    finished=False,
                )
            ft.status = EquipmentStatus.RUNNING
            ft.flow_kg_s = params.flow_rate_kg_s

        # Set injection state
        self.sim.state.injection_rate_kg_s = params.flow_rate_kg_s
        self.sim.state.injection_boron_ppm = params.boron_ppm

        detail = (
            f"Injecting {params.source} at {params.flow_rate_kg_s:.1f} kg/s"
            + (f" with {params.boron_ppm:.0f} ppm boron" if params.boron_ppm > 0 else "")
        )

        return self._advance_time_and_get_output("inject_coolant", detail)

    @tool
    async def order_scram(self, params: OrderScramParams) -> ToolOutput:
        """Emergency reactor shutdown (SCRAM / AZ-5).
        Inserts all control rods simultaneously. Use with extreme caution —
        in RBMK reactors with low ORM, this can trigger positive reactivity
        insertion from graphite-tipped control rods.
        """
        assert self.sim is not None

        if not params.confirm:
            return ToolOutput(
                metadata={"error": "Scram not confirmed"},
                blocks=[TextBlock(text="SCRAM not executed. Set confirm=true to proceed.")],
                reward=0.0,
                finished=False,
            )

        s = self.sim.state
        s.scram_active = True
        s.rods_inserting = True

        warning = ""
        if self.sim.reactor_type == "rbmk" and s.control_rod_position_pct > 80.0:
            warning = (
                "\n⚠ WARNING: RBMK SCRAM from near-fully-withdrawn position! "
                f"Graphite displacers will enter core first. ORM={s.orm_count}. "
                "Positive reactivity insertion expected for first 8-10 seconds."
            )

        detail = f"AZ-5 / SCRAM initiated. Rods inserting from {s.control_rod_position_pct:.1f}%.{warning}"

        return self._advance_time_and_get_output("order_scram", detail)

    @tool
    async def vent_containment(self, params: VentContainmentParams) -> ToolOutput:
        """Vent containment pressure to reduce hydrogen explosion risk.
        vent_path: 'filtered' (reduces release), 'unfiltered' (direct), 'wetwell' (through suppression pool, scrubs fission products).
        WARNING: Venting releases some radioactivity to the environment.
        """
        assert self.sim is not None

        valid_paths = {"filtered", "unfiltered", "wetwell"}
        if params.vent_path not in valid_paths:
            return ToolOutput(
                metadata={"error": f"Unknown vent path: {params.vent_path}"},
                blocks=[TextBlock(text=f"Error: Unknown vent path '{params.vent_path}'. Available: {valid_paths}")],
                reward=0.0,
                finished=False,
            )

        # Open containment vent
        vent = self.sim.equipment.get("containment_vent")
        if vent is None:
            # Create one if not present
            from equipment import Valve
            vent = Valve("containment_vent", {"status": "running", "position_pct": 100.0})
            self.sim.equipment.equipment["containment_vent"] = vent
        else:
            if isinstance(vent, Valve):
                vent.open()
            else:
                vent.start()

        self.sim.equipment.containment_venting = True

        detail = (
            f"Containment venting initiated via {params.vent_path} path. "
            f"Current containment pressure: {self.sim.state.containment_pressure_mpa:.3f} MPa, "
            f"H₂: {self.sim.state.containment_hydrogen_pct:.1f}%."
        )

        return self._advance_time_and_get_output("vent_containment", detail)

    @tool
    async def submit_log(self, params: SubmitLogParams) -> ToolOutput:
        """Document your reasoning and observations. No simulation effect.
        Use this to record your diagnosis, decision rationale, or concerns.
        """
        self.action_log.append({
            "step": self.step_count,
            "action": "log",
            "detail": params.entry,
            "severity": params.severity,
        })

        return ToolOutput(
            metadata={"logged": True},
            blocks=[TextBlock(text=f"[{params.severity.upper()}] Log recorded: {params.entry}")],
            reward=0.0,
            finished=False,
        )

    @tool
    async def wait(self, params: WaitParams) -> ToolOutput:
        """Wait and monitor the reactor without taking any operational action.
        Advances simulation time by one timestep. Use this when conditions are
        stable and no intervention is needed.
        """
        assert self.sim is not None

        return self._advance_time_and_get_output(
            "wait", "Monitoring — no operational action taken."
        )

    # =========================================================================
    # Class methods for task/split enumeration
    # =========================================================================

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        return ScenarioRegistry.list_tasks(split)

    @classmethod
    def list_splits(cls) -> list[str]:
        return ScenarioRegistry.list_splits()
