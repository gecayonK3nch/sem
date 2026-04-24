from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional plotting dependency
    plt = None


@dataclass(frozen=True)
class ModelParams:
    """Input parameters of the safety interval model."""

    speed_kmh: float = 40.0
    braking_distance_m: float = 10.0
    leader_speed_kmh: float = 0.0
    leader_braking_distance_m: float = 0.0
    reaction_time_s: float = 0.2
    communication_delay_s: float = 0.5
    packet_loss_probability: float = 0.01
    vehicle_length_m: float = 4.6


@dataclass(frozen=True)
class SafetyIntervalResult:
    """Breakdown of all terms in the minimum safety interval formula."""

    speed_mps: float
    leader_speed_mps: float
    max_deceleration_mps2: float
    leader_max_deceleration_mps2: float
    stopping_distance_m: float
    leader_stopping_distance_m: float
    reaction_distance_m: float
    communication_distance_m: float
    nominal_required_gap_m: float
    reliability_factor: float
    minimum_interval_m: float
    minimum_interval_s: float


@dataclass(frozen=True)
class SimulationTrace:
    """Time-series trajectory of one emergency-braking scenario."""

    time_s: np.ndarray
    lead_position_m: np.ndarray
    follower_position_m: np.ndarray
    lead_lateral_m: np.ndarray
    follower_lateral_m: np.ndarray
    lead_speed_mps: np.ndarray
    follower_speed_mps: np.ndarray
    gap_m: np.ndarray
    avoidance_active: np.ndarray
    alert_delay_s: float
    collision: bool


def kmh_to_mps(speed_kmh: float) -> float:
    if speed_kmh < 0:
        raise ValueError("Speed cannot be negative.")
    return speed_kmh / 3.6


def max_deceleration_from_braking_distance(speed_mps: float, braking_distance_m: float) -> float:
    """Compute a from stopping-distance formula: s = v^2 / (2a)."""
    if speed_mps <= 0:
        raise ValueError("Speed must be positive to estimate deceleration.")
    if braking_distance_m <= 0:
        raise ValueError("Braking distance must be positive.")
    return speed_mps**2 / (2.0 * braking_distance_m)


def reliability_factor(packet_loss_probability: float) -> float:
    """Piecewise reliability factor defined by the project requirement.

    - 0% losses: 1.0
    - 1..10% losses: linearly from 1.10 to 1.25
    - >10% losses: 1.5 and higher (linear growth)
    """
    if not 0 <= packet_loss_probability < 1:
        raise ValueError("Packet loss probability must be in [0, 1).")

    if packet_loss_probability < 0.01:
        return 1.0
    if packet_loss_probability <= 0.10:
        ratio = (packet_loss_probability - 0.01) / 0.09
        return 1.10 + 0.15 * ratio
    return 1.50 + 2.0 * (packet_loss_probability - 0.10)


def calculate_minimum_safe_interval(params: ModelParams) -> SafetyIntervalResult:
    """Calculate S_min in meters and equivalent headway in seconds.

    Formula from the selected method:
    S_min = k_rel * (x_stop + V * (t_reaction + t_comm))
    where x_stop = V^2 / (2a).
    """
    speed_mps = kmh_to_mps(params.speed_kmh)
    leader_speed_mps = kmh_to_mps(params.leader_speed_kmh)
    a_max = max_deceleration_from_braking_distance(speed_mps, params.braking_distance_m)

    if leader_speed_mps <= 0:
        leader_a_max = float("inf")
        x_stop_leader = 0.0
    else:
        if params.leader_braking_distance_m <= 0:
            raise ValueError("Leader braking distance must be positive when leader speed is positive.")
        leader_a_max = max_deceleration_from_braking_distance(
            leader_speed_mps,
            params.leader_braking_distance_m,
        )
        x_stop_leader = leader_speed_mps**2 / (2.0 * leader_a_max)

    x_stop_follower = speed_mps**2 / (2.0 * a_max)
    x_reaction = speed_mps * params.reaction_time_s
    x_comm = speed_mps * params.communication_delay_s
    k_rel = reliability_factor(params.packet_loss_probability)

    # Required bumper-to-bumper distance is the max closure over time.
    t_delay = params.reaction_time_s + params.communication_delay_s

    follower_stop_time = t_delay + speed_mps / a_max
    leader_stop_time = 0.0 if leader_speed_mps <= 0 else leader_speed_mps / leader_a_max
    horizon = max(follower_stop_time, leader_stop_time) + 0.5
    n_samples = 6000
    t = np.linspace(0.0, horizon, n_samples)

    # Follower trajectory: cruise during delay, then braking.
    t_after_delay = np.maximum(t - t_delay, 0.0)
    t_brake_f = np.minimum(t_after_delay, speed_mps / a_max)
    x_follower = speed_mps * np.minimum(t, t_delay) + (speed_mps * t_brake_f - 0.5 * a_max * t_brake_f**2)

    # Leader trajectory: immediate braking at t=0 if moving.
    if leader_speed_mps <= 0:
        x_leader = np.zeros_like(t)
    else:
        t_brake_l = np.minimum(t, leader_speed_mps / leader_a_max)
        x_leader = leader_speed_mps * t_brake_l - 0.5 * leader_a_max * t_brake_l**2

    closure = x_follower - x_leader
    s_raw = max(0.0, float(np.max(closure)))
    s_min = k_rel * s_raw
    return SafetyIntervalResult(
        speed_mps=speed_mps,
        leader_speed_mps=leader_speed_mps,
        max_deceleration_mps2=a_max,
        leader_max_deceleration_mps2=leader_a_max,
        stopping_distance_m=x_stop_follower,
        leader_stopping_distance_m=x_stop_leader,
        reaction_distance_m=x_reaction,
        communication_distance_m=x_comm,
        nominal_required_gap_m=s_raw,
        reliability_factor=k_rel,
        minimum_interval_m=s_min,
        minimum_interval_s=(s_min / speed_mps) if speed_mps > 0 else 0.0,
    )


def sample_alert_delay_s(params: ModelParams, rng: np.random.Generator) -> float:
    """Sample alert delay including retry cost under packet losses.

    Delay = reaction + communication + (failed_packets * communication_delay).
    The count of failed packets before first successful one follows geometric law.
    """
    p_success = 1.0 - params.packet_loss_probability
    failed_packets = rng.geometric(p_success) - 1
    return (
        params.reaction_time_s
        + params.communication_delay_s
        + failed_packets * params.communication_delay_s
    )


def simulate_emergency_braking(
    params: ModelParams,
    initial_gap_m: float,
    duration_s: float = 6.0,
    dt_s: float = 0.02,
    rng: Optional[np.random.Generator] = None,
    fixed_alert_delay_s: Optional[float] = None,
    sudden_stop_lead: bool = False,
) -> SimulationTrace:
    """Simulate one scenario where lead car stops suddenly at t=0 by default."""
    if initial_gap_m <= 0:
        raise ValueError("Initial gap must be positive.")
    if duration_s <= 0 or dt_s <= 0:
        raise ValueError("Simulation duration and step must be positive.")
    if params.vehicle_length_m <= 0:
        raise ValueError("Vehicle length must be positive.")

    local_rng = rng if rng is not None else np.random.default_rng()
    v0 = kmh_to_mps(params.speed_kmh)
    v0_lead = kmh_to_mps(params.leader_speed_kmh)
    a_max = max_deceleration_from_braking_distance(v0, params.braking_distance_m)

    if sudden_stop_lead or v0_lead <= 0:
        a_lead = float("inf")
    else:
        if params.leader_braking_distance_m <= 0:
            raise ValueError("Leader braking distance must be positive when leader speed is positive.")
        a_lead = max_deceleration_from_braking_distance(v0_lead, params.leader_braking_distance_m)
    alert_delay = (
        fixed_alert_delay_s
        if fixed_alert_delay_s is not None
        else sample_alert_delay_s(params, local_rng)
    )

    n_steps = int(np.ceil(duration_s / dt_s)) + 1
    t = np.linspace(0.0, duration_s, n_steps)

    lead_pos = np.zeros_like(t)
    follower_pos = np.zeros_like(t)
    lead_y = np.zeros_like(t)
    follower_y = np.zeros_like(t)
    lead_speed = np.zeros_like(t)
    follower_speed = np.zeros_like(t)
    avoidance = np.zeros_like(t, dtype=bool)

    lane_width_m = 3.6
    vehicle_width_m = 1.9
    lane_change_duration_s = 1.1
    lane_change_trigger_buffer_s = 0.35
    bypass_min_speed_mps = max(2.0, 0.22 * v0)
    overlap_clearance_m = 1.2

    lead_pos[0] = 0.0
    follower_pos[0] = -(initial_gap_m + params.vehicle_length_m)
    lead_speed[0] = 0.0 if sudden_stop_lead else v0_lead
    follower_speed[0] = v0
    lead_y[0] = 0.0
    follower_y[0] = 0.0

    lane_change_started = False
    lane_change_t0 = 0.0
    y_start = 0.0
    y_target = lane_width_m

    def has_overlap_1d(c1: float, size1: float, c2: float, size2: float) -> bool:
        return abs(c1 - c2) < 0.5 * (size1 + size2)

    def lead_decel_cap() -> float:
        if np.isinf(a_lead):
            return np.inf
        return a_lead

    def collision_is_inevitable(x_lead: float, x_follow: float, v_lead_now: float, v_follow_now: float) -> bool:
        current_gap = x_lead - (x_follow + params.vehicle_length_m)
        if current_gap <= 0:
            return True
        follower_stop = v_follow_now**2 / (2.0 * a_max) if a_max > 0 else np.inf
        if np.isinf(lead_decel_cap()):
            leader_stop = 0.0
        else:
            leader_stop = v_lead_now**2 / (2.0 * lead_decel_cap())
        # If even immediate full braking cannot preserve positive longitudinal clearance,
        # trigger evasive maneuver in addition to braking.
        return (current_gap + leader_stop - follower_stop) < 0.8

    def ttc_seconds(x_lead: float, x_follow: float, v_lead_now: float, v_follow_now: float) -> float:
        current_gap = x_lead - (x_follow + params.vehicle_length_m)
        closing_speed = v_follow_now - v_lead_now
        if current_gap <= 0:
            return 0.0
        if closing_speed <= 1e-6:
            return np.inf
        return current_gap / closing_speed

    for i in range(1, n_steps):
        if sudden_stop_lead:
            lead_speed[i] = 0.0
            lead_pos[i] = lead_pos[i - 1]
        else:
            lead_acc = -a_lead if lead_speed[i - 1] > 0 else 0.0
            lead_speed[i] = max(lead_speed[i - 1] + lead_acc * dt_s, 0.0)
            lead_pos[i] = lead_pos[i - 1] + lead_speed[i - 1] * dt_s

        same_lane = has_overlap_1d(
            lead_y[i - 1],
            vehicle_width_m,
            follower_y[i - 1],
            vehicle_width_m,
        )
        if (
            not lane_change_started
            and same_lane
            and collision_is_inevitable(lead_pos[i - 1], follower_pos[i - 1], lead_speed[i - 1], follower_speed[i - 1])
        ):
            lane_change_started = True
            lane_change_t0 = t[i - 1]
            y_start = follower_y[i - 1]

        if lane_change_started:
            # Keep bypass moving, but prevent rear-end while lateral overlap is still significant.
            lateral_sep = abs(lead_y[i - 1] - follower_y[i - 1])
            same_conflict_zone = lateral_sep < vehicle_width_m
            gap_now = lead_pos[i - 1] - (follower_pos[i - 1] + params.vehicle_length_m)
            target_gap = overlap_clearance_m if same_conflict_zone else 0.0
            ttc_now = ttc_seconds(lead_pos[i - 1], follower_pos[i - 1], lead_speed[i - 1], follower_speed[i - 1])

            if same_conflict_zone and ttc_now < (lane_change_duration_s + 0.15) and follower_speed[i - 1] > lead_speed[i - 1]:
                follower_acc = -a_max
            elif same_conflict_zone and gap_now < target_gap and follower_speed[i - 1] > lead_speed[i - 1]:
                follower_acc = -a_max
            elif same_conflict_zone and follower_speed[i - 1] > (lead_speed[i - 1] + 0.8):
                follower_acc = -0.45 * a_max
            else:
                follower_acc = 0.0

            projected_v = follower_speed[i - 1] + follower_acc * dt_s
            if projected_v < bypass_min_speed_mps:
                follower_acc = (bypass_min_speed_mps - follower_speed[i - 1]) / dt_s
        elif t[i - 1] >= alert_delay and follower_speed[i - 1] > 0:
            follower_acc = -a_max
        else:
            follower_acc = 0.0
        follower_speed[i] = max(follower_speed[i - 1] + follower_acc * dt_s, 0.0)
        follower_pos[i] = follower_pos[i - 1] + follower_speed[i - 1] * dt_s

        if lane_change_started:
            tau = (t[i] - lane_change_t0) / lane_change_duration_s
            tau = float(np.clip(tau, 0.0, 1.0))
            smooth_tau = tau * tau * (3.0 - 2.0 * tau)
            follower_y[i] = y_start + (y_target - y_start) * smooth_tau
            avoidance[i] = True
        else:
            follower_y[i] = follower_y[i - 1]

        lead_y[i] = lead_y[i - 1]

    # Gap is bumper-to-bumper distance: lead rear minus follower front.
    gap = lead_pos - (follower_pos + params.vehicle_length_m)
    overlap_x = np.abs(lead_pos - follower_pos) < params.vehicle_length_m
    overlap_y = np.abs(lead_y - follower_y) < vehicle_width_m
    collided = bool(np.any(overlap_x & overlap_y))

    return SimulationTrace(
        time_s=t,
        lead_position_m=lead_pos,
        follower_position_m=follower_pos,
        lead_lateral_m=lead_y,
        follower_lateral_m=follower_y,
        lead_speed_mps=lead_speed,
        follower_speed_mps=follower_speed,
        gap_m=gap,
        avoidance_active=avoidance,
        alert_delay_s=alert_delay,
        collision=collided,
    )


def estimate_collision_probability(
    params: ModelParams,
    gap_m: float,
    n_runs: int = 1000,
    duration_s: float = 6.0,
    dt_s: float = 0.02,
    seed: int = 42,
) -> float:
    if n_runs <= 0:
        raise ValueError("n_runs must be positive.")
    rng = np.random.default_rng(seed)
    collisions = 0
    for _ in range(n_runs):
        trace = simulate_emergency_braking(
            params,
            initial_gap_m=gap_m,
            duration_s=duration_s,
            dt_s=dt_s,
            rng=rng,
        )
        collisions += int(trace.collision)
    return collisions / n_runs


def find_safe_interval_by_simulation(
    params: ModelParams,
    target_collision_probability: float = 1e-3,
    n_runs: int = 1500,
    seed: int = 42,
) -> float:
    """Find minimum gap by binary search over collision probability."""
    if not 0 <= target_collision_probability < 1:
        raise ValueError("Target collision probability must be in [0, 1).")

    analytical = calculate_minimum_safe_interval(params).minimum_interval_m
    low = 0.2 * analytical
    high = 2.5 * analytical

    while estimate_collision_probability(params, high, n_runs=n_runs, seed=seed) > target_collision_probability:
        high *= 1.2

    for _ in range(25):
        mid = 0.5 * (low + high)
        p_col = estimate_collision_probability(params, mid, n_runs=n_runs, seed=seed)
        if p_col <= target_collision_probability:
            high = mid
        else:
            low = mid
    return high


def plot_simulation(trace: SimulationTrace) -> None:
    if plt is None:
        raise RuntimeError("Matplotlib is not installed; plotting is unavailable.")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    fig.patch.set_facecolor("#f7f8fc")

    axes[0].plot(trace.time_s, trace.lead_speed_mps * 3.6, label="Лидер", color="#0b84f3", linewidth=2.5)
    axes[0].plot(
        trace.time_s,
        trace.follower_speed_mps * 3.6,
        label="Следующий",
        color="#f39c12",
        linewidth=2.5,
    )
    axes[0].axvline(trace.alert_delay_s, linestyle="--", color="#555555", label="Момент реакции")
    axes[0].set_ylabel("Скорость, км/ч")
    axes[0].set_title("Динамика торможения и безопасного интервала", fontsize=13, weight="bold")
    axes[0].legend()

    axes[1].plot(trace.time_s, trace.gap_m, color="#2c3e50", linewidth=2.5)
    axes[1].axhline(0.0, color="#e74c3c", linestyle="--", linewidth=1.5, label="Граница столкновения")
    axes[1].set_xlabel("Время, с")
    axes[1].set_ylabel("Дистанция между авто, м")
    axes[1].legend()

    for ax in axes:
        ax.grid(alpha=0.35)

    plt.tight_layout()
    plt.show()


def result_to_dict(result: SafetyIntervalResult) -> Dict[str, float]:
    return {
        "speed_mps": result.speed_mps,
        "leader_speed_mps": result.leader_speed_mps,
        "max_deceleration_mps2": result.max_deceleration_mps2,
        "leader_max_deceleration_mps2": result.leader_max_deceleration_mps2,
        "stopping_distance_m": result.stopping_distance_m,
        "leader_stopping_distance_m": result.leader_stopping_distance_m,
        "reaction_distance_m": result.reaction_distance_m,
        "communication_distance_m": result.communication_distance_m,
        "nominal_required_gap_m": result.nominal_required_gap_m,
        "reliability_factor": result.reliability_factor,
        "minimum_interval_m": result.minimum_interval_m,
        "minimum_interval_s": result.minimum_interval_s,
    }


def run_demo(simulate: bool = True, seed: int = 42) -> None:
    params = ModelParams()
    analytical = calculate_minimum_safe_interval(params)

    print("=== Модель безопасного интервала беспилотных такси ===")
    print(f"Скорость следующего: {params.speed_kmh:.1f} км/ч ({analytical.speed_mps:.3f} м/с)")
    print(f"Скорость лидера: {params.leader_speed_kmh:.1f} км/ч ({analytical.leader_speed_mps:.3f} м/с)")
    print(f"Макс. замедление следующего: {analytical.max_deceleration_mps2:.3f} м/с^2")
    print(f"Макс. замедление лидера: {analytical.leader_max_deceleration_mps2:.3f} м/с^2")
    print(f"Остановочный путь следующего: {analytical.stopping_distance_m:.3f} м")
    print(f"Остановочный путь лидера: {analytical.leader_stopping_distance_m:.3f} м")
    print(f"Дистанция реакции: {analytical.reaction_distance_m:.3f} м")
    print(f"Дистанция задержки связи: {analytical.communication_distance_m:.3f} м")
    print(f"Коэффициент надежности: {analytical.reliability_factor:.5f}")
    print(
        f"Минимальный безопасный интервал: {analytical.minimum_interval_m:.3f} м "
        f"({analytical.minimum_interval_s:.3f} с)"
    )

    simulated = find_safe_interval_by_simulation(params, target_collision_probability=1e-3, n_runs=800, seed=seed)
    print(f"Оценка интервала по имитации (P_столк <= 0.1%): {simulated:.3f} м")

    if simulate:
        rng = np.random.default_rng(seed)
        trace = simulate_emergency_braking(
            params,
            initial_gap_m=analytical.minimum_interval_m,
            duration_s=6.0,
            dt_s=0.02,
            rng=rng,
            sudden_stop_lead=False,
        )
        print(
            f"Симуляция: задержка оповещения={trace.alert_delay_s:.3f} с, "
            f"столкновение={'ДА' if trace.collision else 'НЕТ'}"
        )
        if plt is not None:
            plot_simulation(trace)


if __name__ == "__main__":
    run_demo(simulate=True)