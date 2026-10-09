"""Episode scoring tests.

The score is computed once, at the end of the episode: the step safety index
averaged over the full horizon plus the terminal outcome, minus the same
quantity for the plant left unattended. These tests check that doing nothing
scores exactly 0, that a scripted operator following each scenario's
procedure beats it, and that episode length alone earns nothing.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

import npp_sim as N
from reactor import ReactorState
from rewards import RewardCalculator
from scenarios import TRAIN_SCENARIOS

CRISIS_SCENARIOS = [
    "tmi_porv_stuck",
    "tmi_recovery",
    "fukushima_rcic_failure",
    "fukushima_hydrogen",
    "windscale_fire",
]

PARAMS = {
    "wait": N.WaitParams,
    "operate_valve": N.OperateValveParams,
    "operate_pump": N.OperatePumpParams,
    "activate_system": N.ActivateSystemParams,
    "inject_coolant": N.InjectCoolantParams,
    "vent_containment": N.VentContainmentParams,
}

# Procedures taken from each scenario's description.
PROCEDURES = {
    "tmi_porv_stuck": [
        ("operate_valve", {"valve_id": "block_valve", "action": "close"}),
    ],
    "tmi_recovery": [
        ("operate_pump", {"pump_id": "hpi_1", "action": "start"}),
        ("operate_pump", {"pump_id": "hpi_2", "action": "start"}),
    ],
    "fukushima_rcic_failure": [
        ("activate_system", {"system_id": "fire_truck", "action": "activate"}),
        ("operate_valve", {"valve_id": "srv_1", "action": "open"}),
        ("operate_valve", {"valve_id": "srv_2", "action": "open"}),
        ("inject_coolant", {"source": "fire_truck", "flow_rate_kg_s": 100.0}),
    ],
    "fukushima_hydrogen": [
        ("inject_coolant", {"source": "fire_truck", "flow_rate_kg_s": 30.0}),
        ("vent_containment", {"vent_path": "wetwell"}),
    ],
    "windscale_fire": [
        ("activate_system", {"system_id": "water_injection", "action": "activate"}),
        ("operate_pump", {"pump_id": "blower_1", "action": "stop"}),
    ],
}


def _task(scenario: str, seed: int = 0) -> dict:
    return next(
        t for t in N.NuclearPlantEnvironment.list_tasks("test")
        if t["scenario"] == scenario and t["seed"] == seed
    )


async def _play(scenario: str, actions: list, seed: int = 0):
    """Run the actions, then wait one step at a time until the episode ends.

    Returns every tool reward (the session score is their sum), the final
    output and the environment.
    """
    env = N.NuclearPlantEnvironment(task_spec=_task(scenario, seed))
    await env.setup()
    rewards = []
    for name, params in actions:
        out = await getattr(env, name)(PARAMS[name](**params))
        rewards.append(out.reward)
        if out.finished:
            return rewards, out, env
    for _ in range(1000):
        out = await env.wait(N.WaitParams())
        rewards.append(out.reward)
        if out.finished:
            return rewards, out, env
    pytest.fail("episode did not end")


def test_train_split_is_the_crisis_scenarios():
    assert sorted(TRAIN_SCENARIOS) == sorted(CRISIS_SCENARIOS)


@pytest.mark.asyncio
@pytest.mark.parametrize("seed", [0, 7])
@pytest.mark.parametrize("scenario", CRISIS_SCENARIOS)
async def test_wait_only_scores_exactly_zero(scenario, seed):
    rewards, out, _ = await _play(scenario, [], seed=seed)
    assert sum(rewards) == 0.0
    assert all(r == 0.0 for r in rewards)


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", CRISIS_SCENARIOS)
async def test_scenario_procedure_beats_waiting(scenario):
    wait_rewards, _, _ = await _play(scenario, [])
    rewards, out, _ = await _play(scenario, PROCEDURES[scenario])
    assert sum(rewards) > sum(wait_rewards) + 0.1


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", CRISIS_SCENARIOS)
async def test_only_the_final_step_is_rewarded(scenario):
    rewards, out, _ = await _play(scenario, PROCEDURES[scenario])
    assert all(r == 0.0 for r in rewards[:-1])
    assert out.finished and out.reward == sum(rewards)


@pytest.mark.asyncio
async def test_start_stable_plant_does_not_stabilize_while_waiting():
    # The TMI plant looks stable until the stuck PORV has drained the core.
    # Seed 1 takes over at minute 0.
    _, out, env = await _play("tmi_porv_stuck", [], seed=1)
    assert out.metadata["reason"] != "stabilized"
    assert env.step_count > 100


@pytest.mark.asyncio
async def test_stabilization_counts_only_after_unattended_failure_point():
    rewards, out, env = await _play("tmi_porv_stuck", PROCEDURES["tmi_porv_stuck"])
    assert out.metadata["reason"] == "stabilized"
    # The unattended plant fails 129 minutes into the transient; seed 0 takes
    # over at minute 90. Success then needs 30 further stable minutes.
    assert env.baseline.steps == 129 - 90
    assert env.step_count == env.baseline.steps + 30


def _porv_variant(seed: int) -> tuple[int, bool]:
    ic = _task("tmi_porv_stuck", seed)["initial_conditions"]
    stuck = ic["equipment"]["block_valve"]["status"] == "stuck_open"
    return ic.get("unattended_minutes", 0), stuck


def _porv_faults(seed: int) -> tuple[int, bool, int, tuple[str, ...]]:
    """(takeover minute, block valve stuck, HPI pumps out, pumps with lost seal cooling)."""
    ic = _task("tmi_porv_stuck", seed)["initial_conditions"]
    eq = ic["equipment"]
    hpi_out = sum(eq[h]["status"] != "running" for h in ("hpi_1", "hpi_2"))
    seals = tuple(r for r in ("rcp_1", "rcp_2", "rcp_3", "rcp_4") if eq[r].get("seal_cooling_lost"))
    return ic.get("unattended_minutes", 0), eq["block_valve"]["status"] == "stuck_open", hpi_out, seals


@pytest.mark.asyncio
async def test_porv_stability_start_is_on_the_plant_clock():
    # Seed 9 takes over at minute 110; the unattended plant fails at minute 129.
    env = N.NuclearPlantEnvironment(task_spec=_task("tmi_porv_stuck", 9))
    await env.setup()
    prompt = (await env.get_prompt())[0].text
    assert "Stability is assessed from minute 129 onward" in prompt
    readings = env.sim.get_instrument_readings()
    assert readings["time"]["elapsed_minutes"] == 110.0


def test_porv_seeds_vary_takeover_time_and_equipment():
    variants = [_porv_faults(seed) for seed in range(10)]
    assert len(set(variants)) == 10
    assert any(stuck for _, stuck, _, _ in variants)
    assert any(not stuck and minutes >= 90 for minutes, stuck, _, _ in variants)
    # Both HPI pumps out, with different pumps left with intact seals.
    no_hpi = [seals for _, _, hpi_out, seals in variants if hpi_out == 2]
    assert len(no_hpi) >= 2 and len(set(no_hpi)) >= 2
    assert any(hpi_out == 1 for _, _, hpi_out, _ in variants)


def test_takeover_time_is_the_unattended_plant_at_that_minute():
    from reactor import ReactorSimulation
    from scenarios import ALL_SCENARIOS

    scenario = ALL_SCENARIOS["tmi_porv_stuck"]
    late = ReactorSimulation(
        "pwr", {**scenario.initial_conditions, "unattended_minutes": 45}, 1.0
    )
    plain = ReactorSimulation("pwr", scenario.initial_conditions, 1.0)
    for _ in range(45):
        plain.advance()
    assert late.state == plain.state


INJECT = ("inject_coolant", {"source": "borated_water", "flow_rate_kg_s": 30.0})
CLOSE_BLOCK = ("operate_valve", {"valve_id": "block_valve", "action": "close"})
START_RCPS = [("operate_pump", {"pump_id": f"rcp_{i}", "action": "start"}) for i in range(1, 5)]


@pytest.mark.asyncio
async def test_porv_block_valve_stuck_needs_makeup_injection():
    # Seed 1: block valve stuck open, so isolating the leak is impossible.
    assert _porv_variant(1) == (0, True)
    isolate, _, _ = await _play("tmi_porv_stuck", [CLOSE_BLOCK] + START_RCPS, seed=1)
    makeup, out, _ = await _play("tmi_porv_stuck", [INJECT, CLOSE_BLOCK] + START_RCPS, seed=1)
    assert sum(isolate) < 0.1
    assert out.metadata["reason"] == "stabilized"
    assert sum(makeup) > 2.0


@pytest.mark.asyncio
async def test_porv_late_takeover_rewards_inventory_makeup():
    # Seed 0: takeover at minute 90, block valve works but the core is heating.
    assert _porv_faults(0) == (90, False, 0, ())
    isolate, _, _ = await _play("tmi_porv_stuck", [CLOSE_BLOCK] + START_RCPS, seed=0)
    makeup, _, _ = await _play("tmi_porv_stuck", [INJECT, CLOSE_BLOCK] + START_RCPS, seed=0)
    late, _, _ = await _play(
        "tmi_porv_stuck", [("wait", {"duration_steps": 10}), CLOSE_BLOCK] + START_RCPS, seed=0
    )
    assert sum(makeup) > sum(isolate) + 0.04
    assert sum(isolate) > sum(late)


@pytest.mark.asyncio
async def test_porv_injection_is_limited_by_running_hpi_pumps():
    # Seed 3: hpi_2 is out, so injection delivers at most hpi_1's 30 kg/s.
    assert _porv_faults(3)[2] == 1
    env = N.NuclearPlantEnvironment(task_spec=_task("tmi_porv_stuck", 3))
    await env.setup()
    out = await env.inject_coolant(N.InjectCoolantParams(source="borated_water", flow_rate_kg_s=100.0))
    assert "at most 30.0 kg/s" in out.blocks[0].text
    assert env.sim.effective_injection_kg_s() == 30.0
    # Stopping the last HPI pump stops the injection with it.
    await env.operate_pump(N.OperatePumpParams(pump_id="hpi_1", action="stop"))
    assert env.sim.effective_injection_kg_s() == 0.0


@pytest.mark.asyncio
async def test_porv_injection_needs_a_running_hpi_pump():
    # Seed 2: both HPI pumps are out and cannot be restarted.
    assert _porv_faults(2)[2] == 2
    env = N.NuclearPlantEnvironment(task_spec=_task("tmi_porv_stuck", 2))
    await env.setup()
    out = await env.inject_coolant(N.InjectCoolantParams(source="borated_water", flow_rate_kg_s=30.0))
    assert out.metadata.get("error") and not out.finished
    assert env.step_count == 0
    out = await env.operate_pump(N.OperatePumpParams(pump_id="hpi_1", action="start"))
    assert "cannot be restarted" in out.blocks[0].text
    assert env.sim.equipment.get("hpi_1").status.value == "failed"


@pytest.mark.asyncio
async def test_pump_with_lost_seal_cooling_fails_and_leaks_when_started():
    # Seed 6: rcp_1..rcp_3 lost seal cooling; rcp_4 is intact.
    assert _porv_faults(6)[3] == ("rcp_1", "rcp_2", "rcp_3")
    env = N.NuclearPlantEnvironment(task_spec=_task("tmi_porv_stuck", 6))
    await env.setup()
    text = env.sim.format_readings(env.sim.get_instrument_readings())
    assert "rcp_1: tripped (seals: seal cooling lost)" in text
    assert "rcp_4: tripped\n" in text
    out = await env.operate_pump(N.OperatePumpParams(pump_id="rcp_1", action="start"))
    assert "seals" in out.blocks[0].text and "leaking" in out.blocks[0].text
    pressure = env.sim.state.coolant_pressure_mpa
    assert env.sim.rcs_leak_rate(pressure) > env.sim.equipment.get_porv_leak_rate(pressure)
    assert env.sim.equipment.get_effective_coolant_flow() == 0.0
    for retry in (
        N.OperatePumpParams(pump_id="rcp_1", action="start"),
        N.OperatePumpParams(pump_id="rcp_1", action="set_speed", speed_pct=100.0),
    ):
        out = await env.operate_pump(retry)
        assert env.sim.equipment.get("rcp_1").status.value == "failed"
    out = await env.operate_pump(N.OperatePumpParams(pump_id="rcp_2", action="set_speed", speed_pct=50.0))
    assert env.sim.equipment.get("rcp_2").seal_failed


@pytest.mark.asyncio
async def test_porv_no_single_sequence_wins_every_seed():
    # The sequence that wins every seed on a plant without these faults:
    # isolate, start every RCP, inject.
    modal = [CLOSE_BLOCK] + START_RCPS + [INJECT]
    no_rcp = [CLOSE_BLOCK, INJECT]
    rcp_4 = [CLOSE_BLOCK, ("operate_pump", {"pump_id": "rcp_4", "action": "start"})]
    rcp_1 = [CLOSE_BLOCK, ("operate_pump", {"pump_id": "rcp_1", "action": "start"})]
    # Seed 6: no HPI, only rcp_4 has intact seals.
    modal_6, _, _ = await _play("tmi_porv_stuck", modal, seed=6)
    best_6, out, _ = await _play("tmi_porv_stuck", rcp_4, seed=6)
    assert out.metadata["reason"] == "stabilized"
    assert sum(best_6) > sum(modal_6) + 2.0
    # Seed 2: no HPI, so injection is impossible and an RCP must run.
    no_rcp_2, _, _ = await _play("tmi_porv_stuck", no_rcp, seed=2)
    best_2, out, _ = await _play("tmi_porv_stuck", rcp_1, seed=2)
    assert out.metadata["reason"] == "stabilized"
    assert sum(best_2) > sum(no_rcp_2) + 2.0
    # Seed 1: block valve stuck, so only injection saves the core.
    rcps_1, _, _ = await _play("tmi_porv_stuck", [CLOSE_BLOCK] + START_RCPS, seed=1)
    modal_1, _, _ = await _play("tmi_porv_stuck", modal, seed=1)
    assert sum(modal_1) > sum(rcps_1) + 2.0


def test_pwr_is_not_stable_while_losing_inventory():
    calc = RewardCalculator("pwr", "stabilize", time_step_minutes=10.0)
    state = _healthy_state()
    state.coolant_inventory_fraction = 0.95
    for step in range(1, 10):
        state.coolant_inventory_fraction -= 0.001
        done, _ = calc.is_terminal(state, step, 200)
        assert not done
    for step in range(10, 13):
        done, reason = calc.is_terminal(state, step, 200)
    assert done and reason == "stabilized"


@pytest.mark.asyncio
@pytest.mark.parametrize("seed", range(10))
async def test_porv_scores_stay_within_scale(seed):
    for actions in ([], [INJECT, CLOSE_BLOCK] + START_RCPS,
                    [("operate_pump", {"pump_id": "hpi_1", "action": "stop"})]):
        rewards, _, _ = await _play("tmi_porv_stuck", actions, seed=seed)
        assert -4.0 <= sum(rewards) <= 4.0


def _healthy_state() -> ReactorState:
    state = ReactorState()
    state.cladding_temp_c = 350.0
    state.containment_pressure_mpa = 0.1
    return state


def test_early_stabilized_end_scores_like_holding_the_final_state():
    from rewards import EpisodeTracker

    state = _healthy_state()
    short = EpisodeTracker(RewardCalculator("pwr", "stabilize"), max_steps=200)
    full = EpisodeTracker(RewardCalculator("pwr", "stabilize"), max_steps=200)
    for _ in range(10):
        short.record(state, state)
    for _ in range(200):
        full.record(state, state)
    a = short.result(state, "stabilized")
    b = full.result(state, "max_steps_reached")
    assert a.value == pytest.approx(b.value, abs=1e-12)


def test_catastrophic_end_is_padded_with_worst_index():
    from rewards import EpisodeTracker

    calc = RewardCalculator("pwr", "stabilize")
    state = _healthy_state()
    tracker = EpisodeTracker(calc, max_steps=200)
    for _ in range(3):
        tracker.record(state, state)
    state.containment_hydrogen_pct = 19.0
    result = tracker.result(state, "hydrogen_detonation")
    expected = (3 * calc.step_reward(_healthy_state(), _healthy_state()) - 197.0) / 200
    assert result.mean_safety_index == pytest.approx(expected)
    # Surviving longer before the same catastrophe is worth more.
    later = EpisodeTracker(calc, max_steps=200)
    for _ in range(50):
        later.record(_healthy_state(), _healthy_state())
    assert later.result(state, "hydrogen_detonation").value > result.value


@pytest.mark.asyncio
async def test_wait_duration_steps_advances_several_steps():
    env = N.NuclearPlantEnvironment(task_spec=_task("windscale_fire"))
    await env.setup()
    out = await env.wait(N.WaitParams(duration_steps=10))
    assert env.step_count == 10
    assert not out.finished and out.reward == 0.0


@pytest.mark.asyncio
async def test_wait_duration_is_capped_at_the_horizon():
    env = N.NuclearPlantEnvironment(task_spec=_task("windscale_fire"))
    await env.setup()
    out = await env.wait(N.WaitParams(duration_steps=10_000))
    assert out.finished
    assert env.step_count == env.config.max_steps
    assert out.reward == 0.0  # identical to the unattended plant


@pytest.mark.asyncio
async def test_venting_through_an_existing_vent_does_not_raise():
    # fukushima_hydrogen defines a containment_vent valve up front.
    env = N.NuclearPlantEnvironment(task_spec=_task("fukushima_hydrogen"))
    await env.setup()
    out = await env.vent_containment(N.VentContainmentParams(vent_path="wetwell"))
    assert out.metadata["step"] == 1
    out = await env.vent_containment(N.VentContainmentParams(vent_path="filtered"))
    assert out.metadata["step"] == 2


FIRE_TRUCK = ("inject_coolant", {"source": "fire_truck", "flow_rate_kg_s": 30.0})


@pytest.mark.asyncio
async def test_scrubbed_vent_path_releases_less_than_unfiltered():
    scrubbed, out_s, env_s = await _play(
        "fukushima_hydrogen", [FIRE_TRUCK, ("vent_containment", {"vent_path": "wetwell"})]
    )
    direct, out_d, env_d = await _play(
        "fukushima_hydrogen", [FIRE_TRUCK, ("vent_containment", {"vent_path": "unfiltered"})]
    )
    assert out_d.metadata["reason"] == "catastrophic_release"
    assert out_s.metadata["reason"] not in ("catastrophic_release", "hydrogen_detonation", "core_meltdown")
    assert sum(scrubbed) > sum(direct) + 0.5


@pytest.mark.asyncio
async def test_closing_the_vent_stops_venting():
    env = N.NuclearPlantEnvironment(task_spec=_task("fukushima_hydrogen"))
    await env.setup()
    await env.vent_containment(N.VentContainmentParams(vent_path="wetwell"))
    await env.operate_valve(N.OperateValveParams(valve_id="containment_vent", action="close"))
    before = env.sim.state.containment_pressure_mpa
    await env.wait(N.WaitParams(duration_steps=3))
    assert env.sim.state.containment_pressure_mpa > before


@pytest.mark.asyncio
async def test_windscale_release_stops_once_the_fire_is_out():
    water_off = [
        ("activate_system", {"system_id": "water_injection", "action": "activate"}),
        ("operate_pump", {"pump_id": "blower_1", "action": "stop"}),
    ]
    passive, _, _ = await _play("windscale_fire", [])
    prompt, _, env = await _play("windscale_fire", water_off)
    delayed, _, _ = await _play("windscale_fire", [("wait", {"duration_steps": 20})] + water_off)
    assert sum(prompt) > sum(delayed) + 0.05
    assert sum(delayed) > sum(passive) + 0.05
    assert env.sim.state.environmental_release_tbq < 5.0
    assert env.sim.state.graphite_temp_c >= 20.0


@pytest.mark.asyncio
@pytest.mark.parametrize("seed", range(10))
async def test_intermittent_offscale_reading_does_not_raise(seed):
    # Core exit thermocouples read OFFSCALE HIGH intermittently in this scenario.
    env = N.NuclearPlantEnvironment(task_spec=_task("tmi_loss_of_coolant", seed))
    await env.setup()
    seen = set()
    for _ in range(30):
        value = env.sim.get_instrument_readings()["thermal"]["cladding_temp_c"]
        seen.add(type(value))
    assert seen <= {float, str}
