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
    _, out, env = await _play("tmi_porv_stuck", [])
    assert out.metadata["reason"] != "stabilized"
    assert env.step_count > 100


@pytest.mark.asyncio
async def test_stabilization_counts_only_after_unattended_failure_point():
    rewards, out, env = await _play("tmi_porv_stuck", PROCEDURES["tmi_porv_stuck"])
    assert out.metadata["reason"] == "stabilized"
    # The unattended plant fails at step 129 (1-minute steps); success then
    # needs 30 further stable minutes.
    assert env.step_count == 129 + 30


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
