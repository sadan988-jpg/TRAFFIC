"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         EcoSync Hackathon — Trigger V.1                                      ║
║         Phase 4: Deep RL Agent — "The Strategist"                            ║
║                                                                              ║
║  Role   : Traffic signal controller via Deep Q-Network (DQN)                 ║
║  Inputs : TraCI state (Phase 1) + YOLOv8 vision (Phase 2)                   ║
║           + LSTM predictions (Phase 3)                                       ║
║  Output : Optimal traffic phase decisions → traci.trafficlight.setPhase()   ║
║  Stack  : Stable Baselines3 · Gymnasium · PyTorch · TraCI                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

DESIGN PHILOSOPHY
─────────────────
The Strategist treats the entire intersection as a single Markov Decision
Process. At each decision step it receives a rich observation vector fusing
*what is happening now* (vehicle counts, current phase) with *what The Oracle
predicts will happen* (LSTM density forecasts) — then outputs a discrete phase
selection that minimises a weighted penalty on waiting time, CO2 emissions, and
predicted jam risk.

The reward function directly encodes the EcoSync mission:
  R = -(w1·WaitTime + w2·Emissions + w3·PredictedJam)
making every green second earned from a less-congested lane and every vehicle
that doesn't idle a positive learning signal.

PHASE 4 MODULE MAP
──────────────────
  TrafficEnv          — Gymnasium environment wrapping TraCI + Phase 2/3 data
  DQNNetwork          — PyTorch neural net (dueling DQN architecture)
  ReplayBuffer        — Prioritised experience replay
  DQNAgent            — Epsilon-greedy DQN with target network
  EcoSyncStrategist   — Top-level coordinator (train / deploy / evaluate)
  main()              — Entry point with TRAIN / DEPLOY toggle
"""

import os
import sys
import time
import random
import logging
import argparse
from collections import deque, namedtuple
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
from gymnasium import spaces

# ── Conditional TraCI import (graceful mock for offline dev) ──────────────────
try:
    import traci
    import traci.constants as tc
    TRACI_AVAILABLE = True
except ImportError:
    TRACI_AVAILABLE = False
    print("[WARNING] TraCI not found — running in MOCK mode for development.")

# ── Phase 3 Oracle import ─────────────────────────────────────────────────────
try:
    from traffic_oracle import TrafficOracle
    ORACLE_AVAILABLE = True
except ImportError:
    ORACLE_AVAILABLE = False
    print("[WARNING] traffic_oracle.py not found — Oracle predictions disabled.")

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [Strategist] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("EcoSync.Strategist")


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
class Config:
    """Central configuration — tweak here for the hackathon."""

    # ── Simulation ────────────────────────────────────────────────────────
    SUMO_CFG        = "ecosync_env/ecosync.sumocfg"   # path to your SUMO config
    SUMO_BINARY     = "sumo"                   # "sumo-gui" for visual debugging
    STEP_LENGTH     = 1.0                      # seconds per RL decision step
    SIM_STEPS       = 3600                     # 1 hour per episode
    TL_ID           = "J0"                     # traffic light junction ID in SUMO

    # ── Lanes (must match SUMO network & Phase 2 lane IDs) ───────────────
    LANE_IDS        = [
        "N_in_0", "N_in_1", "N_in_2",
        "S_in_0", "S_in_1", "S_in_2",
        "E_in_0", "E_in_1", "E_in_2",
        "W_in_0", "W_in_1", "W_in_2",
    ]
    NUM_LANES       = len(LANE_IDS)

    # ── Traffic light phases ─────────────────────────────────────────────
    # Index maps 1:1 to traci.trafficlight.setPhase() argument
    PHASES = {
        0: "Green NS  / Red EW",     # North-South green
        1: "Yellow NS / Red EW",     # North-South yellow
        2: "Red NS   / Green EW",    # East-West green
        3: "Red NS   / Yellow EW",   # East-West yellow
    }
    # Actions the RL agent can choose (only "meaningful" phases — no yellow)
    # Yellow transitions are handled automatically by the env step logic
    ACTION_PHASES   = [0, 2]          # agent picks NS-green or EW-green
    NUM_ACTIONS     = len(ACTION_PHASES)

    # Phase durations (seconds)
    MIN_GREEN_STEPS = 5               # safety minimum green
    MAX_GREEN_STEPS = 60              # cap to prevent starvation
    YELLOW_STEPS    = 3               # fixed yellow duration

    # ── Observation ───────────────────────────────────────────────────────
    FORECAST_HORIZON = 5              # steps from Phase 3 Oracle
    # obs vector size = num_lanes*(1 count + horizon predicted counts +
    #                              1 emission + horizon predicted emissions)
    #                 + 1 (current phase) + 1 (phase elapsed ratio)
    OBS_DIM = NUM_LANES * (1 + FORECAST_HORIZON + 1 + FORECAST_HORIZON) + 2

    # ── Reward weights ────────────────────────────────────────────────────
    W1_WAIT_TIME    = 0.50   # penalise total waiting time
    W2_EMISSIONS    = 0.30   # penalise CO2/NOx emissions
    W3_JAM_PENALTY  = 0.20   # penalise predicted future jam risk

    # Normalisation denominators (empirical maximums for this intersection)
    MAX_WAIT_TIME   = 300.0  # seconds total across all lanes
    MAX_EMISSIONS   = 10.0   # emission score sum across all lanes
    MAX_JAM_PENALTY = 4.0    # sum of congestion risks (1 per lane × 4)

    # ── DQN Hyperparameters ───────────────────────────────────────────────
    HIDDEN_DIM      = 128
    REPLAY_CAPACITY = 20_000
    BATCH_SIZE      = 64
    GAMMA           = 0.95
    LR              = 5e-4
    TAU             = 0.005            # soft target update coefficient
    EPS_START       = 1.0
    EPS_END         = 0.05
    EPS_DECAY       = 0.995            # per episode
    TARGET_UPDATE   = 10               # hard update every N episodes
    WARMUP_EPISODES = 5                # pure exploration before learning

    # ── Training ──────────────────────────────────────────────────────────
    TRAIN_EPISODES  = 200
    EVAL_INTERVAL   = 20               # evaluate every N episodes
    CHECKPOINT_DIR  = "checkpoints"
    BEST_MODEL_PATH = "checkpoints/strategist_best.pt"

    # ── Deployment ────────────────────────────────────────────────────────
    DEPLOY_MODEL    = "checkpoints/strategist_best.pt"
    DEPLOY_EPISODES = 1                # run one episode for demo


# ─────────────────────────────────────────────────────────────────────────────
# MOCK TRACI — for offline development without SUMO installed
# ─────────────────────────────────────────────────────────────────────────────
class MockTraCI:
    """
    Minimal TraCI mock so the full pipeline can be unit-tested without SUMO.
    Replace with real `traci` in production.
    """
    class trafficlight:
        @staticmethod
        def setPhase(tl_id, phase): pass
        @staticmethod
        def getPhase(tl_id): return random.randint(0, 3)

    class lane:
        @staticmethod
        def getLastStepVehicleNumber(lane_id):
            return random.randint(0, 10)
        @staticmethod
        def getWaitingTime(lane_id):
            return random.uniform(0, 60)
        @staticmethod
        def getLastStepMeanSpeed(lane_id):
            return random.uniform(0, 13.9)

    class simulation:
        @staticmethod
        def getTime(): return time.time()
        @staticmethod
        def getDeltaT(): return 1.0
        @staticmethod
        def step(): time.sleep(0.001)
        @staticmethod
        def getMinExpectedNumber(): return random.randint(10, 50)

    class vehicle:
        @staticmethod
        def getIDList(): return [f"veh_{i}" for i in range(random.randint(5, 20))]
        @staticmethod
        def getCO2Emission(veh_id): return random.uniform(0, 200)
        @staticmethod
        def getNOxEmission(veh_id): return random.uniform(0, 50)

    @staticmethod
    def start(cmd): log.info("[MockTraCI] start: %s", cmd)
    @staticmethod
    def close(): log.info("[MockTraCI] closed")
    @staticmethod
    def isLoaded(): return True


if not TRACI_AVAILABLE:
    traci = MockTraCI()


# ─────────────────────────────────────────────────────────────────────────────
# MOCK YOLO OUTPUT — simulates Phase 2 live_traffic_data
# ─────────────────────────────────────────────────────────────────────────────
def get_yolo_data(lane_ids: List[str]) -> Dict[str, Dict]:
    """
    Production: replace this with your Phase 2 YOLOv8 live output.
    Returns the same format as Phase 2:
        {lane_id: {"count": int, "emissions_score": float}}
    """
    return {
        lid: {
            "count":           int(traci.lane.getLastStepVehicleNumber(lid)
                                   if TRACI_AVAILABLE else random.randint(0, 10)),
            "emissions_score": float(random.uniform(0.1, 0.8)),
        }
        for lid in lane_ids
    }


# ─────────────────────────────────────────────────────────────────────────────
# GYMNASIUM ENVIRONMENT — TrafficEnv
# ─────────────────────────────────────────────────────────────────────────────
class TrafficEnv(gym.Env):
    """
    Gymnasium-compatible environment wrapping:
      • Phase 1 — SUMO/TraCI simulation
      • Phase 2 — YOLOv8 perception data (vehicle counts & emissions)
      • Phase 3 — LSTM Oracle predictions (future density forecasts)

    Observation Space
    ─────────────────
    A flat float32 vector containing, per lane:
        [current_count, pred_count×5, current_emission, pred_emission×5]
    Plus global signals:
        [current_phase_normalised, phase_elapsed_ratio]

    Total dim: NUM_LANES × 12 + 2

    Action Space
    ────────────
    Discrete(2):
        0 → Set Green for North-South (Phase 0)
        1 → Set Green for East-West   (Phase 2)

    Reward
    ──────
    R = -(w1·norm_wait + w2·norm_emissions + w3·norm_jam)
    Clipped to [-1, 0] to keep training stable.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config: Config = None,
        oracle: Optional[Any] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.cfg         = config or Config()
        self.oracle      = oracle
        self.render_mode = render_mode

        # ── Spaces ────────────────────────────────────────────────────────
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self.cfg.OBS_DIM,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.cfg.NUM_ACTIONS)

        # ── Internal state ─────────────────────────────────────────────────
        self._sim_step          = 0
        self._current_phase     = 0
        self._phase_step_count  = 0
        self._episode_reward    = 0.0
        self._in_yellow         = False
        self._yellow_remaining  = 0
        self._pending_phase     = None
        self._episode_metrics: Dict = {}

    # ── Gymnasium API ──────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        self._start_sumo()
        self._sim_step         = 0
        self._current_phase    = 0
        self._phase_step_count = 0
        self._episode_reward   = 0.0
        self._in_yellow        = False
        self._yellow_remaining = 0
        self._pending_phase    = None
        self._episode_metrics  = {
            "total_wait":      0.0,
            "total_emissions": 0.0,
            "total_jam_risk":  0.0,
            "steps":           0,
        }
        self._emergency_wait_tracker = {lid: 0 for lid in self.cfg.LANE_IDS}
        self._emergency_triggered = False

        obs = self._get_observation()
        return obs, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one RL decision step.

        If in yellow transition: ignore agent action, count down yellow.
        Otherwise: check if the requested phase differs → insert yellow.
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"

        desired_phase = self.cfg.ACTION_PHASES[action]

        # ── Yellow transition logic ────────────────────────────────────────
        if self._in_yellow:
            self._yellow_remaining -= 1
            self._advance_simulation(1)
            if self._yellow_remaining <= 0:
                self._in_yellow = False
                self._set_phase(self._pending_phase)
        else:
            if desired_phase != self._current_phase:
                # Insert yellow before switching
                yellow_phase = self._current_phase + 1  # e.g. 0→1 or 2→3
                self._set_phase(yellow_phase)
                self._in_yellow        = True
                self._yellow_remaining = self.cfg.YELLOW_STEPS
                self._pending_phase    = desired_phase
                self._advance_simulation(1)
            else:
                self._phase_step_count += 1
                # Enforce max green to prevent lane starvation
                if self._phase_step_count >= self.cfg.MAX_GREEN_STEPS:
                    forced = 2 if self._current_phase == 0 else 0
                    self._set_phase(forced)
                    self._phase_step_count = 0
                self._advance_simulation(1)

        self._sim_step += 1

        # ── Gather metrics ─────────────────────────────────────────────────
        obs    = self._get_observation()
        reward = self._compute_reward()

        self._episode_reward                 += reward
        self._episode_metrics["total_wait"]  += self._last_wait
        self._episode_metrics["total_emissions"] += self._last_emissions
        self._episode_metrics["total_jam_risk"]  += self._last_jam_risk
        self._episode_metrics["steps"]       += 1

        terminated = self._sim_step >= self.cfg.SIM_STEPS
        truncated  = False

        info = {
            "sim_step":    self._sim_step,
            "phase":       self._current_phase,
            "wait_time":   self._last_wait,
            "emissions":   self._last_emissions,
            "jam_risk":    self._last_jam_risk,
            "ep_reward":   self._episode_reward,
            "emergency_corridor_active": self._emergency_triggered,
        }

        if terminated:
            traci.close()
            log.info(
                "Episode done | reward=%.2f | wait=%.1fs | emissions=%.2f",
                self._episode_reward,
                self._episode_metrics["total_wait"],
                self._episode_metrics["total_emissions"],
            )
            info["episode"] = self._episode_metrics.copy()

        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        if self.render_mode == "human":
            log.info(
                "Step=%4d | Phase=%d | Reward=%.3f | Wait=%.1f | Emissions=%.2f",
                self._sim_step, self._current_phase,
                self._episode_reward,
                getattr(self, "_last_wait", 0),
                getattr(self, "_last_emissions", 0),
            )

    def close(self) -> None:
        try:
            traci.close()
        except Exception:
            pass

    # ── Observation builder ────────────────────────────────────────────────

    def _get_observation(self) -> np.ndarray:
        """
        Fuse Phase 2 (YOLOv8) and Phase 3 (Oracle) data into a flat obs vector.
        All values normalised to [0, 1].
        """
        yolo_data = get_yolo_data(self.cfg.LANE_IDS)

        # Update Oracle if available
        if self.oracle is not None:
            self.oracle.update(yolo_data, self._get_signal_states())
            pred_state = self.oracle.get_predicted_state()
        else:
            pred_state = None

        obs_parts = []

        for lid in self.cfg.LANE_IDS:
            lane_data = yolo_data.get(lid, {})
            count     = lane_data.get("count", 0)
            emission  = lane_data.get("emissions_score", 0.0)

            # Normalise current observation
            norm_count    = min(count / 15.0, 1.0)    # cap at 15 vehicles
            norm_emission = float(np.clip(emission, 0, 1))

            # Predicted counts and emissions (from Oracle)
            if pred_state and lid in pred_state:
                lane_pred     = pred_state[lid]
                pred_counts   = [min(v / 15.0, 1.0) for v in lane_pred["predicted_counts"]]
                pred_emissions= [float(np.clip(v, 0, 1)) for v in lane_pred["predicted_emissions"]]
            else:
                pred_counts    = [norm_count]    * self.cfg.FORECAST_HORIZON
                pred_emissions = [norm_emission] * self.cfg.FORECAST_HORIZON

            obs_parts.extend([norm_count] + pred_counts + [norm_emission] + pred_emissions)

        # Global features
        phase_norm    = self._current_phase / max(len(self.cfg.PHASES) - 1, 1)
        elapsed_ratio = min(self._phase_step_count / self.cfg.MAX_GREEN_STEPS, 1.0)
        obs_parts.extend([phase_norm, elapsed_ratio])

        obs = np.array(obs_parts, dtype=np.float32)

        # Safety check — pad/trim to exact OBS_DIM
        if len(obs) < self.cfg.OBS_DIM:
            obs = np.pad(obs, (0, self.cfg.OBS_DIM - len(obs)))
        elif len(obs) > self.cfg.OBS_DIM:
            obs = obs[:self.cfg.OBS_DIM]

        return obs

    # ── Reward function ────────────────────────────────────────────────────

    def _compute_reward(self) -> float:
        """
        R = -(w1·norm_wait + w2·norm_emissions + w3·norm_jam)

        Each component is normalised to [0,1] before weighting so that
        no single penalty dominates training on its raw scale.

        Urban air quality mission is encoded by w2 (emissions) and w3 (jam
        penalty) — idling vehicles and future congestion directly raise both.
        """
        # ── Component 1: Total waiting time across all lanes ───────────────
        total_wait = sum(
            traci.lane.getWaitingTime(lid) if TRACI_AVAILABLE else random.uniform(0, 30)
            for lid in self.cfg.LANE_IDS
        )
        norm_wait = min(total_wait / self.cfg.MAX_WAIT_TIME, 1.0)

        # ── Component 2: Emissions proxy (CO2 + NOx from TraCI or Phase 2) ─
        if TRACI_AVAILABLE:
            try:
                veh_ids = traci.vehicle.getIDList()
                co2_sum = sum(traci.vehicle.getCO2Emission(v) for v in veh_ids)
                nox_sum = sum(traci.vehicle.getNOxEmission(v) for v in veh_ids)
                emission_score = (co2_sum / 1000.0 + nox_sum * 2.0) / max(len(veh_ids), 1)
            except Exception:
                emission_score = 0.0
        else:
            # Fallback: use Phase 2 emission scores from YOLO
            yolo = get_yolo_data(self.cfg.LANE_IDS)
            emission_score = sum(d.get("emissions_score", 0) for d in yolo.values())

        norm_emissions = min(emission_score / self.cfg.MAX_EMISSIONS, 1.0)

        # ── Component 3: Predicted jam penalty from Oracle ─────────────────
        jam_penalty = 0.0
        if self.oracle is not None:
            pred = self.oracle.get_predicted_state()
            if pred:
                jam_penalty = sum(
                    pred.get(lid, {}).get("congestion_risk", 0.0)
                    for lid in self.cfg.LANE_IDS
                )
        norm_jam = min(jam_penalty / self.cfg.MAX_JAM_PENALTY, 1.0)

        # ── Weighted penalty reward ────────────────────────────────────────
        reward = -(
            self.cfg.W1_WAIT_TIME  * norm_wait     +
            self.cfg.W2_EMISSIONS  * norm_emissions +
            self.cfg.W3_JAM_PENALTY * norm_jam
        )

        # ── Emergency Corridor Penalty ─────────────────────────────────────
        yolo_data = get_yolo_data(self.cfg.LANE_IDS)
        signal_states = self._get_signal_states()
        self._emergency_triggered = False
        for lid in self.cfg.LANE_IDS:
            counts = yolo_data.get(lid, {}).get("class_counts", {})
            if counts.get("emergency", 0) > 0:
                if signal_states.get(lid) in ["red", "yellow"]:
                    self._emergency_wait_tracker[lid] += 1
                else:
                    self._emergency_wait_tracker[lid] = 0
            else:
                self._emergency_wait_tracker[lid] = 0

        # If any emergency vehicle has waited > 1 tick (approx 1s)
        if any(w > 1 for w in self._emergency_wait_tracker.values()):
            reward = -100.0
            self._emergency_triggered = True

        # Cache for info dict
        self._last_wait      = total_wait
        self._last_emissions = emission_score
        self._last_jam_risk  = jam_penalty

        return float(np.clip(reward, -100.0, 0.0))

    # ── Helpers ────────────────────────────────────────────────────────────

    def _start_sumo(self) -> None:
        """Launch (or re-launch) SUMO simulation via TraCI."""
        try:
            if TRACI_AVAILABLE:
                if traci.isLoaded():
                    traci.close()
                sumo_cmd = [
                    self.cfg.SUMO_BINARY,
                    "-c", self.cfg.SUMO_CFG,
                    "--step-length", str(self.cfg.STEP_LENGTH),
                    "--no-warnings", "true",
                    "--time-to-teleport", "-1",
                ]
                traci.start(sumo_cmd)
        except Exception as e:
            log.warning("TraCI start failed (%s) — using mock.", e)

    def _advance_simulation(self, steps: int = 1) -> None:
        for _ in range(steps):
            try:
                traci.simulation.step()
            except Exception:
                pass

    def _set_phase(self, phase: int) -> None:
        try:
            traci.trafficlight.setPhase(self.cfg.TL_ID, phase)
        except Exception:
            pass
        self._current_phase = phase

    def _get_signal_states(self) -> Dict[str, str]:
        phase_to_lane_signal = {
            0: {"lane_N": "green", "lane_S": "green", "lane_E": "red",   "lane_W": "red"},
            1: {"lane_N": "yellow","lane_S": "yellow","lane_E": "red",   "lane_W": "red"},
            2: {"lane_N": "red",   "lane_S": "red",   "lane_E": "green", "lane_W": "green"},
            3: {"lane_N": "red",   "lane_S": "red",   "lane_E": "yellow","lane_W": "yellow"},
        }
        return phase_to_lane_signal.get(self._current_phase, {lid: "red" for lid in self.cfg.LANE_IDS})


# ─────────────────────────────────────────────────────────────────────────────
# DUELING DQN NETWORK
# ─────────────────────────────────────────────────────────────────────────────
class DQNNetwork(nn.Module):
    """
    Dueling Double DQN architecture.

    Dueling streams separate the estimation of:
        V(s)       — how good is this state regardless of action
        A(s,a)     — advantage of each action over average

    Q(s,a) = V(s) + A(s,a) - mean(A(s,·))

    This improves learning in traffic scenarios where many actions have
    similar Q-values (e.g. when both lanes are empty — either green is fine).
    """

    def __init__(self, obs_dim: int, num_actions: int, hidden_dim: int = 128):
        super().__init__()

        # Shared feature extractor
        self.feature_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features   = self.feature_net(x)
        value      = self.value_stream(features)         # (B, 1)
        advantages = self.advantage_stream(features)     # (B, A)
        q_values   = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values


# ─────────────────────────────────────────────────────────────────────────────
# PRIORITISED EXPERIENCE REPLAY
# ─────────────────────────────────────────────────────────────────────────────
Transition = namedtuple("Transition", ["obs", "action", "reward", "next_obs", "done"])


class ReplayBuffer:
    """
    Uniform experience replay with efficient numpy storage.
    Upgrade to PER (Prioritised Experience Replay) for Phase 5 if needed.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append(Transition(obs, action, reward, next_obs, done))

    def sample(self, batch_size: int) -> Optional[Transition]:
        if len(self.buffer) < batch_size:
            return None
        batch = random.sample(self.buffer, batch_size)
        return Transition(*[np.array(x) for x in zip(*batch)])

    def __len__(self) -> int:
        return len(self.buffer)


# ─────────────────────────────────────────────────────────────────────────────
# DQN AGENT
# ─────────────────────────────────────────────────────────────────────────────
class DQNAgent:
    """
    Double DQN agent with soft target network updates.

    Key design choices for the hackathon:
    ─────────────────────────────────────
    • Soft target updates (τ=0.005) are more stable than hard updates for
      non-stationary traffic environments.
    • Epsilon decays per episode, not per step — giving full exploration
      within each episode's context before reducing.
    • Gradient clipping at 1.0 prevents explosive gradients during rare
      high-density events (accidents, platoons).
    """

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        config: Config,
        device: Optional[str] = None,
    ):
        self.cfg         = config
        self.num_actions = num_actions

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Online and target networks
        self.online_net = DQNNetwork(obs_dim, num_actions, config.HIDDEN_DIM).to(self.device)
        self.target_net = DQNNetwork(obs_dim, num_actions, config.HIDDEN_DIM).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimiser = optim.Adam(self.online_net.parameters(), lr=config.LR)
        self.replay     = ReplayBuffer(config.REPLAY_CAPACITY)

        self.epsilon    = config.EPS_START
        self.steps_done = 0

        log.info("DQNAgent ready | obs_dim=%d | actions=%d | device=%s",
                 obs_dim, num_actions, self.device)

    def select_action(self, obs: np.ndarray, training: bool = True) -> int:
        """Epsilon-greedy action selection."""
        if training and random.random() < self.epsilon:
            return self.num_actions - 1 if random.random() > 0.5 else 0

        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        self.online_net.eval()
        with torch.no_grad():
            q_values = self.online_net(obs_t)
        return int(q_values.argmax(dim=1).item())

    def store(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self.replay.push(obs, action, reward, next_obs, done)

    def learn(self) -> Optional[float]:
        """One gradient update step. Returns loss or None if buffer not ready."""
        batch = self.replay.sample(self.cfg.BATCH_SIZE)
        if batch is None:
            return None

        obs_t      = torch.tensor(batch.obs,      dtype=torch.float32).to(self.device)
        action_t   = torch.tensor(batch.action,   dtype=torch.long).to(self.device)
        reward_t   = torch.tensor(batch.reward,   dtype=torch.float32).to(self.device)
        next_obs_t = torch.tensor(batch.next_obs, dtype=torch.float32).to(self.device)
        done_t     = torch.tensor(batch.done,     dtype=torch.float32).to(self.device)

        # Current Q values
        self.online_net.train()
        q_values = self.online_net(obs_t).gather(1, action_t.unsqueeze(1)).squeeze(1)

        # Double DQN target: action selected by online, evaluated by target
        with torch.no_grad():
            next_actions = self.online_net(next_obs_t).argmax(dim=1)
            next_q       = self.target_net(next_obs_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q     = reward_t + self.cfg.GAMMA * next_q * (1 - done_t)

        loss = F.smooth_l1_loss(q_values, target_q)

        self.optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 1.0)
        self.optimiser.step()

        # Soft target update
        for p_online, p_target in zip(
            self.online_net.parameters(), self.target_net.parameters()
        ):
            p_target.data.copy_(
                self.cfg.TAU * p_online.data + (1 - self.cfg.TAU) * p_target.data
            )

        return loss.item()

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.cfg.EPS_END, self.epsilon * self.cfg.EPS_DECAY)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "online_state":    self.online_net.state_dict(),
            "target_state":    self.target_net.state_dict(),
            "optimiser_state": self.optimiser.state_dict(),
            "epsilon":         self.epsilon,
            "steps_done":      self.steps_done,
        }, path)
        log.info("Checkpoint saved → %s", path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online_state"])
        self.target_net.load_state_dict(ckpt["target_state"])
        self.optimiser.load_state_dict(ckpt["optimiser_state"])
        self.epsilon    = ckpt.get("epsilon", self.cfg.EPS_END)
        self.steps_done = ckpt.get("steps_done", 0)
        log.info("Loaded checkpoint from %s | ε=%.3f", path, self.epsilon)


# ─────────────────────────────────────────────────────────────────────────────
# ECOSYNC STRATEGIST — Top-Level Coordinator
# ─────────────────────────────────────────────────────────────────────────────
class EcoSyncStrategist:
    """
    Coordinates the full Phase 4 pipeline:

    TRAINING MODE
    ─────────────
    for episode in range(TRAIN_EPISODES):
        obs = env.reset()
        while not done:
            action = agent.select_action(obs, training=True)
            next_obs, reward, done, _, info = env.step(action)
            agent.store(obs, action, reward, next_obs, done)
            loss = agent.learn()
            obs = next_obs
        agent.decay_epsilon()
        save_best_checkpoint_if_improved()

    DEPLOYMENT MODE
    ───────────────
    agent.load(BEST_MODEL_PATH)
    agent.epsilon = 0           # pure exploitation
    run one episode, log all metrics for the judges dashboard.
    """

    def __init__(self, config: Config = None):
        self.cfg = config or Config()
        os.makedirs(self.cfg.CHECKPOINT_DIR, exist_ok=True)

        # ── Initialise Phase 3 Oracle ──────────────────────────────────────
        if ORACLE_AVAILABLE:
            self.oracle = TrafficOracle(
                lane_ids         = self.cfg.LANE_IDS,
                window_size      = 60,
                forecast_horizon = self.cfg.FORECAST_HORIZON,
                warmup_steps     = 60,  # shorter for hackathon speed
            )
            log.info("Phase 3 Oracle attached.")
        else:
            self.oracle = None
            log.warning("Oracle not available — predictions disabled.")

        # ── Environment ────────────────────────────────────────────────────
        self.env = TrafficEnv(config=self.cfg, oracle=self.oracle)

        # ── Agent ──────────────────────────────────────────────────────────
        self.agent = DQNAgent(
            obs_dim     = self.cfg.OBS_DIM,
            num_actions = self.cfg.NUM_ACTIONS,
            config      = self.cfg,
        )

        # ── Tracking ──────────────────────────────────────────────────────
        self._best_reward   = -float("inf")
        self._reward_history: List[float] = []
        self._loss_history:   List[float] = []

    # ── TRAINING MODE ──────────────────────────────────────────────────────

    def train(self) -> None:
        """
        Full training loop for TRAIN_EPISODES episodes.
        Best model (by episode reward) is checkpointed automatically.
        """
        log.info("═══ TRAINING MODE — %d episodes ═══", self.cfg.TRAIN_EPISODES)

        for ep in range(1, self.cfg.TRAIN_EPISODES + 1):
            obs, _      = self.env.reset()
            done        = False
            ep_reward   = 0.0
            ep_losses   = []
            step        = 0

            while not done:
                action = self.agent.select_action(obs, training=True)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                self.agent.store(obs, action, reward, next_obs, done)

                # Start learning after warmup episodes
                if ep > self.cfg.WARMUP_EPISODES:
                    loss = self.agent.learn()
                    if loss is not None:
                        ep_losses.append(loss)

                obs        = next_obs
                ep_reward += reward
                step      += 1

            self.agent.decay_epsilon()
            self._reward_history.append(ep_reward)
            mean_loss = np.mean(ep_losses) if ep_losses else 0.0
            self._loss_history.append(mean_loss)

            # ── Checkpoint best model ──────────────────────────────────────
            if ep_reward > self._best_reward:
                self._best_reward = ep_reward
                self.agent.save(self.cfg.BEST_MODEL_PATH)
                log.info("★ New best | ep=%d | reward=%.3f | ε=%.3f",
                         ep, ep_reward, self.agent.epsilon)

            # ── Periodic evaluation log ────────────────────────────────────
            if ep % self.cfg.EVAL_INTERVAL == 0:
                recent_mean = np.mean(self._reward_history[-self.cfg.EVAL_INTERVAL:])
                log.info(
                    "Episode %4d/%d | reward=%.3f | mean(last%d)=%.3f | ε=%.3f | loss=%.4f",
                    ep, self.cfg.TRAIN_EPISODES,
                    ep_reward,
                    self.cfg.EVAL_INTERVAL, recent_mean,
                    self.agent.epsilon,
                    mean_loss,
                )

            # ── Periodic checkpoint (every 50 episodes) ────────────────────
            if ep % 50 == 0:
                self.agent.save(
                    os.path.join(self.cfg.CHECKPOINT_DIR, f"strategist_ep{ep}.pt")
                )

        log.info("Training complete. Best reward: %.3f", self._best_reward)
        self._save_training_summary()

    # ── DEPLOYMENT MODE ────────────────────────────────────────────────────

    def deploy(self) -> Dict:
        """
        Run one full episode in pure exploitation mode (ε=0).
        Logs granular per-step metrics for the judges dashboard.
        Returns a summary dict for dashboard integration.
        """
        log.info("═══ DEPLOYMENT MODE — loading %s ═══", self.cfg.DEPLOY_MODEL)

        if not os.path.exists(self.cfg.DEPLOY_MODEL):
            log.warning("No trained model found. Running with untrained agent.")
        else:
            self.agent.load(self.cfg.DEPLOY_MODEL)

        self.agent.epsilon = 0.0  # pure exploitation — no random exploration

        obs, _    = self.env.reset()
        done      = False
        step      = 0
        ep_reward = 0.0

        step_log: List[Dict] = []

        log.info("Deployment loop started. Control handed to The Strategist.")

        while not done:
            action = self.agent.select_action(obs, training=False)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            ep_reward += reward
            obs        = next_obs
            step      += 1

            # Log every step for the dashboard
            step_entry = {
                "step":     step,
                "action":   self.cfg.ACTION_PHASES[action],
                "phase":    info["phase"],
                "reward":   round(reward, 4),
                "wait":     round(info["wait_time"], 2),
                "emissions":round(info["emissions"], 4),
                "jam_risk": round(info["jam_risk"], 3),
            }
            step_log.append(step_entry)

            if step % 100 == 0:
                log.info(
                    "Deploy step %4d | phase=%d | reward=%.3f | wait=%.1fs | jam=%.2f",
                    step, info["phase"], reward, info["wait_time"], info["jam_risk"]
                )

        summary = {
            "total_reward":    round(ep_reward, 3),
            "total_steps":     step,
            "avg_wait_time":   round(np.mean([s["wait"] for s in step_log]), 2),
            "avg_emissions":   round(np.mean([s["emissions"] for s in step_log]), 4),
            "avg_jam_risk":    round(np.mean([s["jam_risk"] for s in step_log]), 3),
            "phase_0_count":   sum(1 for s in step_log if s["action"] == 0),
            "phase_2_count":   sum(1 for s in step_log if s["action"] == 2),
            "step_log":        step_log,
        }

        log.info(
            "Deployment complete:\n"
            "  Total Reward   : %.3f\n"
            "  Avg Wait Time  : %.2f s\n"
            "  Avg Emissions  : %.4f\n"
            "  Avg Jam Risk   : %.3f\n"
            "  NS Green Count : %d\n"
            "  EW Green Count : %d",
            summary["total_reward"],
            summary["avg_wait_time"],
            summary["avg_emissions"],
            summary["avg_jam_risk"],
            summary["phase_0_count"],
            summary["phase_2_count"],
        )

        return summary

    # ── Stable Baselines3 alternative path ────────────────────────────────

    def train_with_sb3(self, total_timesteps: int = 100_000) -> None:
        """
        Optional: use Stable Baselines3 PPO as an alternative to custom DQN.
        PPO is more sample-efficient when episodes are long (>1000 steps).
        Requires: pip install stable-baselines3

        Uncomment and use in place of self.train() if preferred.
        """
        try:
            from stable_baselines3 import PPO
            from stable_baselines3.common.env_checker import check_env

            log.info("Validating environment with SB3 checker…")
            check_env(self.env, warn=True)

            model = PPO(
                "MlpPolicy",
                self.env,
                verbose      = 1,
                learning_rate= self.cfg.LR,
                gamma        = self.cfg.GAMMA,
                n_steps      = 512,
                batch_size   = self.cfg.BATCH_SIZE,
                n_epochs     = 10,
                tensorboard_log= "./tb_logs/",
            )

            model.learn(total_timesteps=total_timesteps)
            model.save(os.path.join(self.cfg.CHECKPOINT_DIR, "sb3_ppo_strategist"))
            log.info("SB3 PPO training complete.")

        except ImportError:
            log.error("stable-baselines3 not installed. Run: pip install stable-baselines3")

    # ── Internal helpers ────────────────────────────────────────────────────

    def _save_training_summary(self) -> None:
        """Save reward/loss curves for plotting in the dashboard."""
        summary_path = os.path.join(self.cfg.CHECKPOINT_DIR, "training_summary.npz")
        np.savez(
            summary_path,
            rewards = np.array(self._reward_history),
            losses  = np.array(self._loss_history),
        )
        log.info("Training summary saved → %s", summary_path)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    """
    Toggle TRAIN / DEPLOY via --mode flag or ECOSYNC_MODE env variable.

    Training  : python traffic_strategist.py --mode train
    Deployment: python traffic_strategist.py --mode deploy
    """
    parser = argparse.ArgumentParser(description="EcoSync Phase 4 — The Strategist")
    parser.add_argument(
        "--mode",
        type=str,
        default=os.environ.get("ECOSYNC_MODE", "train"),
        choices=["train", "deploy"],
        help="train: learn the city | deploy: run the best model for judges",
    )
    parser.add_argument("--sumo-cfg",   type=str, default=Config.SUMO_CFG)
    parser.add_argument("--sumo-gui",   action="store_true",
                        help="Use sumo-gui for visual debugging")
    parser.add_argument("--episodes",   type=int, default=Config.TRAIN_EPISODES)
    parser.add_argument("--checkpoint", type=str, default=Config.BEST_MODEL_PATH)
    args = parser.parse_args()

    # Apply CLI overrides
    cfg                 = Config()
    cfg.SUMO_CFG        = args.sumo_cfg
    cfg.SUMO_BINARY     = "sumo-gui" if args.sumo_gui else "sumo"
    cfg.TRAIN_EPISODES  = args.episodes
    cfg.BEST_MODEL_PATH = args.checkpoint
    cfg.DEPLOY_MODEL    = args.checkpoint

    strategist = EcoSyncStrategist(config=cfg)

    if args.mode == "train":
        log.info("▶ Starting Training Mode")
        strategist.train()

    elif args.mode == "deploy":
        log.info("▶ Starting Deployment Mode (judges demo)")
        summary = strategist.deploy()

        # Pretty-print final results for the demo table
        print("\n" + "═" * 60)
        print("  EcoSync — Phase 4 Deployment Results")
        print("═" * 60)
        print(f"  Total Episode Reward : {summary['total_reward']}")
        print(f"  Avg Waiting Time     : {summary['avg_wait_time']} s/lane")
        print(f"  Avg Emission Score   : {summary['avg_emissions']}")
        print(f"  Avg Predicted Jam    : {summary['avg_jam_risk']}")
        print(f"  N-S Green Phases     : {summary['phase_0_count']}")
        print(f"  E-W Green Phases     : {summary['phase_2_count']}")
        print("═" * 60)


if __name__ == "__main__":
    main()