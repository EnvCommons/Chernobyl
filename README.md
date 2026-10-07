# Chernobyl

[![OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/GeneralReasoning/Chernobyl)

## Description

Chernobyl is a nuclear power plant management environment where agents operate reactors through crisis scenarios inspired by four real historical disasters: Chernobyl (1986), Three Mile Island (1979), Fukushima Daiichi (2011), and Windscale (1957). The simulation couples point kinetics neutronics, thermal-hydraulics, xenon-135 dynamics, fuel integrity, containment physics, and Wigner energy models to create realistic reactor behavior.

**Note**: This is a synthetic environment which was majority AI-generated; we recommend testing it thoroughly before any use in an RL pipeline.

## Capabilities

- Diagnosing reactor emergencies from instrument readings (some of which may be malfunctioning or misleading)
- Managing control rods, pumps, valves, and safety systems under crisis conditions
- Reasoning about reactor physics including reactivity feedback, xenon poisoning, and decay heat
- Trading off competing safety objectives (e.g., venting containment releases radiation but prevents hydrogen explosion)
- Multi-step sequential decision-making under uncertainty

## Compute Requirements

Agents interact with the environment through environment-specific tools only. No sandbox or filesystem access is provided.

## License

MIT

## Tasks

There are 50 training tasks across 5 crisis scenarios (each with 10 random seeds). In each of them the plant fails or keeps degrading if left alone, so operator action makes a measurable difference:

- **tmi_porv_stuck** (PWR, hard) -- Post-scram. PORV stuck open but indicator shows closed. Unattended, the core uncovers and hydrogen detonates after about two hours. Diagnose the stuck valve and close the block valve to stop the LOCA.
- **tmi_recovery** (PWR, expert) -- Core 40% uncovered, cladding at 1100 C, hydrogen generating. Restore cooling before the cladding oxidizes.
- **fukushima_rcic_failure** (BWR, expert) -- RCIC failed, batteries at 10%, must depressurize and establish low-pressure injection.
- **fukushima_hydrogen** (BWR, expert) -- Core damage underway, H2 at 8% in containment, must vent while minimizing radiation release.
- **windscale_fire** (Windscale, expert) -- Fire detected in pile, must choose between air (fans flames) or water (hydrogen/steam explosion risk).

The seed only changes instrument noise; the plant physics of a scenario is deterministic.

There are 110 test tasks across all 11 scenarios (each with 10 random seeds), including the five training scenarios. The additional test-only scenarios are:

- **chernobyl_normal_ops** (RBMK) -- Steady 3200 MWt operation. Maintain power safely.
- **chernobyl_xenon_pit** (RBMK, hard) -- Power collapsed to 30 MWt with xenon building and most rods withdrawn.
- **chernobyl_test_start** (RBMK, expert) -- Turbine test beginning at 200 MWt with ORM=6 and ECCS disabled.
- **tmi_loss_of_coolant** (PWR, hard) -- Mid-event LOCA with operators having throttled HPI and core starting to uncover.
- **fukushima_blackout** (BWR) -- Post-tsunami station blackout on a generic BWR-4. All AC power lost, batteries draining, RCIC running on steam. Manage diminishing resources.
- **windscale_anneal** (Windscale Pile) -- Ninth Wigner energy anneal with graphite temperature rising unexpectedly and thermocouple blind spots.

## Reward Structure

This is a verifiable environment with a single reward at the end of each episode. After every time step the environment computes a plant safety index in [-1, 1] from five weighted components:

| Component | Weight | Metric |
|---|---|---|
| Core integrity | 0.30 | Cladding temperature margin below safety limit |
| Radiation containment | 0.30 | Environmental release minimized |
| Containment integrity | 0.20 | Pressure margin below design limit |
| Fuel integrity | 0.15 | 1 - fuel damage fraction |
| Hydrogen safety | 0.05 | H2 below flammability threshold (4 vol%) |

Additional penalties apply for exceeding cladding temperature limits and worsening fuel damage. The index is shown to the agent after every step but is not itself a reward.

When the episode ends, its value is the safety index averaged over the scenario's full horizon (max steps) plus a terminal outcome grade from -1.0 (core meltdown) to +1.0 (stabilized with negligible damage). If the episode ends before the horizon, the remaining steps are filled with the final state's index after a successful stabilization, and with -1 after a catastrophe (meltdown, hydrogen detonation, catastrophic release). Episode length therefore earns nothing by itself, while delaying a catastrophe is worth something.

The reward is this value minus the value of the same scenario with the plant left unattended (computed deterministically at setup). An agent that only waits scores exactly 0; positive scores mean the operator did better than doing nothing. All intermediate steps have reward 0.

Stabilization (the success terminal of the stabilize scenarios) requires 30 consecutive simulated minutes of stable plant state (cladding below limit, fuel damage below 10%, hydrogen below 4%, containment below design pressure), counted only after the step at which the unattended plant fails. A plant that starts out stable therefore cannot end the episode before its crisis would have developed. Scenarios whose unattended plant never fails run to their full horizon.

We do not use LLM graders for this task.

## Data

No external data files are required. All scenario parameters and physics constants are defined in the source code based on publicly available reactor design parameters and nuclear engineering references (NUREG reports, IAEA safety series, INSAG-7 Chernobyl report).

## Tools

Agents have 10 environment-specific tools:

| Tool | Time Advance | Description |
|---|---|---|
| `observe_instruments` | No | Read instrument panels. Some readings may be malfunctioning or misleading (historically accurate). |
| `adjust_control_rods` | Yes | Insert/withdraw control rods. Models RBMK graphite-tip positive reactivity insertion. |
| `operate_pump` | Yes | Start/stop/adjust coolant pumps. Some require AC power. |
| `operate_valve` | Yes | Open/close valves (PORV, block valve, SRVs, MSIVs). Stuck valves cannot be operated directly. |
| `activate_system` | Yes | Activate/deactivate safety systems (ECCS, RCIC, diesels, fire trucks). |
| `inject_coolant` | Yes | Emergency injection from borated water, seawater, fire truck, or makeup tank. Fire truck requires low pressure. |
| `order_scram` | Yes | Emergency shutdown (AZ-5/SCRAM). Dangerous in RBMK with low ORM. |
| `vent_containment` | Yes | Vent via filtered, unfiltered, or wetwell path. Reduces H2 risk but releases some radioactivity. |
| `submit_log` | No | Document reasoning. No simulation effect. |
| `wait` | Yes | Advance time without taking action, `duration_steps` steps at once (default 1). Use when monitoring stable conditions. |

## Time Horizon

Chernobyl scenarios range from 150 to 300 maximum steps, with timesteps from 15 seconds to 10 minutes depending on scenario urgency. Each time-advancing tool call advances the simulation by one timestep, except `wait`, which can advance several. Episodes end at a catastrophe, at stabilization, or at the maximum step count.

## Environment Difficulty

Scenarios are rated at three difficulty levels:
- **Normal**: Routine operation with perturbations (chernobyl_normal_ops)
- **Hard**: Active crisis requiring correct diagnosis and intervention (tmi_porv_stuck, fukushima_blackout, chernobyl_xenon_pit, windscale_anneal, tmi_loss_of_coolant)
- **Expert**: Severe accident management with cascading failures and no-win tradeoffs (chernobyl_test_start, tmi_recovery, fukushima_rcic_failure, fukushima_hydrogen, windscale_fire)

## Other Environment Requirements

There are no further environment requirements. The environment works out of the box with the OpenReward endpoint without any secrets or external API keys.

## Safety

Agents in the simulation operate a simulated nuclear reactor and must make safety-critical decisions under uncertainty. The environment models historically accurate instrument failures (e.g., TMI PORV indicator showing "CLOSED" when stuck open) to test whether agents can reason about unreliable information.

The environment does not present direct real-world safety risks, as all physics are simulated. However, the environment teaches reactor operation skills that, in an extreme case, could be misapplied. The physics models are simplified approximations (point kinetics, 1D thermal-hydraulics) and would not be sufficient for actual reactor operation.

## Citations

```bibtex
@dataset{GRChernobyl,
  author    = {General Reasoning Inc. Team},
  title     = {Chernobyl},
  year      = {2026},
  publisher = {OpenReward},
  url       = {https://openreward.ai/GeneralReasoning/Chernobyl}
}
```
