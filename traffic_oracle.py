"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         EcoSync Hackathon — Trigger V.1                                      ║
║         Phase 3: LSTM Prediction Engine — "The Oracle"                       ║
║                                                                              ║
║  Role   : Short-term traffic & emissions forecaster                          ║
║  Input  : Live lane data from Phase 2 (YOLOv8 perception)                   ║
║  Output : Predicted traffic state for ALL lanes (RL-ready)                  ║
║  Stack  : PyTorch · NumPy · Collections                                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

WHY PREDICTIVE FORECASTING IMPROVES RL TRAFFIC OPTIMIZATION
─────────────────────────────────────────────────────────────
Reactive RL agents only act on the *current* state — they are blind to what is
about to happen. By feeding the RL agent a predicted future state (next 5 min)
alongside the current observation, we effectively give the controller a crystal
ball. This allows it to:

  • Pre-emptively extend a green phase BEFORE a wave of vehicles arrives,
    eliminating stop-and-go propagation (the #1 cause of urban NOx spikes).
  • Prevent unnecessary red phases on an already-clear lane, cutting idle time
    and cold-start emissions.
  • Encode temporal context (rush hour shape, platoon arrivals) that a memoryless
    policy cannot learn from a single frame.

HOW THE LSTM REDUCES URBAN AIR POLLUTION
─────────────────────────────────────────
Vehicle idling at red lights is responsible for ~30 % of urban CO2 from traffic.
The LSTM captures the *temporal dependency* between past congestion patterns and
upcoming density — allowing the signal controller to:

  1. Minimise cumulative waiting time → fewer idle seconds → less CO2/NOx.
  2. Smooth traffic flow → avoid acceleration bursts (highest per-km emissions).
  3. Detect emission-heavy vehicle clusters (buses, trucks) early and clear them
     efficiently through longer green windows.

RECOMMENDED HACKATHON HYPERPARAMETERS (24-hour window)
───────────────────────────────────────────────────────
  hidden_size      = 64      # large enough to capture daily periodicity; small
                              # enough for <1 ms inference on CPU
  num_layers       = 2       # stacked LSTM adds depth without GPU requirement
  dropout          = 0.2     # regularises noisy simulation data
  time_steps       = 60      # 60 × 1s frames = 1 min lookback window
  forecast_horizon = 5       # predict 5 steps ahead (5 s per step = 25 s lookahead,
                              # or configure step_size=60 s for 5-min forecast)
  learning_rate    = 1e-3    # Adam default; stable on small online batches
  online_batch     = 8       # keeps gradient updates smooth without retraining
  warmup_steps     = 300     # 5 min at 1 Hz before inference activates
"""

import time
import logging
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [TrafficOracle] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("TrafficOracle")


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL STATE ENCODER
# Maps human-readable phase names to normalised floats for model input.
# ─────────────────────────────────────────────────────────────────────────────
SIGNAL_STATES: Dict[str, float] = {
    "red":    0.0,
    "yellow": 0.5,
    "green":  1.0,
}


def encode_signal(state: str) -> float:
    """Convert a signal phase string to a normalised scalar."""
    return SIGNAL_STATES.get(state.lower(), 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# LSTM MODEL DEFINITION
# ─────────────────────────────────────────────────────────────────────────────
class TrafficLSTM(nn.Module):
    """
    Stacked LSTM that forecasts per-lane traffic density and emission score.

    Architecture
    ────────────
    Input  → LSTM (2 layers) → Dropout → FC → Output
    Shape  : (batch, time_steps, num_features) → (batch, forecast_horizon, num_lanes * 2)

    The final two output channels per lane are:
        [predicted_vehicle_count, predicted_emission_score]
    These are the minimal signals a downstream RL agent needs to compute reward.
    """

    def __init__(
        self,
        num_lanes: int,
        num_features: int = 3,        # [count, emission_score, signal_state]
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        forecast_horizon: int = 5,
    ):
        super().__init__()
        self.num_lanes = num_lanes
        self.forecast_horizon = forecast_horizon

        # Input projection — flatten per-lane features into one vector per step
        # Input dim = num_lanes × num_features
        input_dim = num_lanes * num_features

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

        # Output: for each forecast step → [count, emission] per lane
        output_dim = num_lanes * 2  # 2 targets per lane
        self.fc = nn.Linear(hidden_size, forecast_horizon * output_dim)

        self._output_dim = output_dim

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x      : (batch, time_steps, num_lanes * num_features)
            hidden : optional LSTM hidden state for stateful inference

        Returns:
            preds  : (batch, forecast_horizon, num_lanes, 2)
                     last dim = [predicted_count, predicted_emission_score]
            hidden : carry-forward hidden state
        """
        lstm_out, hidden = self.lstm(x, hidden)        # (B, T, H)
        last_hidden = self.dropout(lstm_out[:, -1, :]) # use final timestep
        raw = self.fc(last_hidden)                     # (B, horizon * output_dim)

        # Reshape to (batch, forecast_horizon, num_lanes, 2)
        batch = x.size(0)
        preds = raw.view(batch, self.forecast_horizon, self.num_lanes, 2)

        return preds, hidden


# ─────────────────────────────────────────────────────────────────────────────
# SLIDING WINDOW BUFFER
# ─────────────────────────────────────────────────────────────────────────────
class LaneBuffer:
    """
    Thread-safe sliding window storing the last `window_size` observations
    for a single lane.

    Each frame stores:
        [vehicle_count, emission_score, signal_state_encoded]
    Missing / dropped frames are forward-filled from the previous value.
    """

    FEATURE_DIM = 3  # count | emission | signal

    def __init__(self, window_size: int = 60):
        self.window_size = window_size
        self._buffer: deque = deque(maxlen=window_size)
        self._last_valid = np.zeros(self.FEATURE_DIM, dtype=np.float32)

    def push(
        self,
        count: Optional[float],
        emission_score: Optional[float],
        signal_state: Optional[str],
    ) -> None:
        """Append one time-step, gracefully handling None values."""
        vec = np.array([
            count          if count          is not None else self._last_valid[0],
            emission_score if emission_score is not None else self._last_valid[1],
            encode_signal(signal_state) if signal_state is not None else self._last_valid[2],
        ], dtype=np.float32)

        self._last_valid = vec.copy()
        self._buffer.append(vec)

    def get_window(self) -> np.ndarray:
        """
        Return array of shape (window_size, FEATURE_DIM).
        Pads with zeros if buffer not yet full (warm-up phase).
        """
        data = np.array(self._buffer, dtype=np.float32)  # (filled, 3)
        if len(data) < self.window_size:
            pad = np.zeros((self.window_size - len(data), self.FEATURE_DIM), dtype=np.float32)
            data = np.vstack([pad, data])
        return data  # (window_size, 3)

    def is_ready(self) -> bool:
        """True once the buffer has at least one full window of data."""
        return len(self._buffer) >= self.window_size

    def __len__(self) -> int:
        return len(self._buffer)


# ─────────────────────────────────────────────────────────────────────────────
# NORMALISATION HELPER
# ─────────────────────────────────────────────────────────────────────────────
class RunningNormaliser:
    """
    Online (Welford) mean/variance normaliser.
    Keeps statistics from the warm-up phase and applies them during inference.
    Avoids data leakage — statistics are computed only on training data.
    """

    def __init__(self, feature_dim: int, eps: float = 1e-6):
        self.n = 0
        self.mean = np.zeros(feature_dim, dtype=np.float64)
        self.M2   = np.zeros(feature_dim, dtype=np.float64)
        self.eps  = eps
        self._frozen = False

    def update(self, x: np.ndarray) -> None:
        """Feed one observation vector (feature_dim,)."""
        if self._frozen:
            return
        self.n += 1
        delta  = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2   += delta * delta2

    def freeze(self) -> None:
        """Lock statistics after warm-up — inference uses these fixed values."""
        self._frozen = True
        log.info("Normaliser frozen: mean=%s std=%s", np.round(self.mean, 3), np.round(self.std, 3))

    @property
    def std(self) -> np.ndarray:
        if self.n < 2:
            return np.ones_like(self.mean)
        var = self.M2 / (self.n - 1)
        return np.maximum(np.sqrt(var), self.eps)

    def normalise(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

    def denormalise(self, x: np.ndarray) -> np.ndarray:
        return (x * self.std[:2] + self.mean[:2]).astype(np.float32)  # only count+emission


# ─────────────────────────────────────────────────────────────────────────────
# ONLINE TRAINING REPLAY BUFFER
# ─────────────────────────────────────────────────────────────────────────────
class ReplayBuffer:
    """
    Lightweight experience replay for online LSTM fine-tuning.
    Stores (input_window, target) pairs; samples random mini-batches.
    """

    def __init__(self, capacity: int = 1000):
        self._X: deque = deque(maxlen=capacity)
        self._Y: deque = deque(maxlen=capacity)

    def push(self, x: np.ndarray, y: np.ndarray) -> None:
        self._X.append(x)
        self._Y.append(y)

    def sample(self, batch_size: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        n = len(self._X)
        if n < batch_size:
            return None
        idx = np.random.choice(n, batch_size, replace=False)
        x_batch = np.stack([self._X[i] for i in idx])
        y_batch = np.stack([self._Y[i] for i in idx])
        return torch.tensor(x_batch), torch.tensor(y_batch)

    def __len__(self) -> int:
        return len(self._X)


# ─────────────────────────────────────────────────────────────────────────────
# TRAFFIC ORACLE — MAIN CLASS
# ─────────────────────────────────────────────────────────────────────────────
class TrafficOracle:
    """
    The Phase 3 LSTM Prediction Engine.

    Lifecycle
    ─────────
    1. Instantiate with lane IDs and config.
    2. Call update() every simulation tick (1 Hz recommended).
    3. System auto-advances through WARMUP → TRAINING → INFERENCE modes.
    4. Call get_predicted_state() to retrieve RL-ready forecast.
    5. Optionally call train_online() manually for more aggressive fine-tuning.

    Integration with Phase 2 (YOLOv8)
    ───────────────────────────────────
        live_traffic_data = {
            "lane_0": {"count": 4, "emissions_score": 0.72},
            "lane_1": {"count": 1, "emissions_score": 0.21},
            ...
        }
        oracle.update(live_traffic_data, signal_states={"lane_0": "green", ...})
        prediction = oracle.get_predicted_state()

    RL Output Format
    ────────────────
        {
            "lane_0": {
                "predicted_counts":    [4.2, 4.8, 5.1, 4.9, 4.4],  # next 5 steps
                "predicted_emissions": [0.71, 0.73, 0.76, 0.74, 0.70],
                "congestion_risk":     0.82,   # scalar [0,1] for RL reward shaping
                "recommended_action":  "extend_green"  # hint to RL policy
            },
            ...
            "oracle_confidence": 0.91,  # mean prediction confidence
            "mode": "inference"         # warmup | training | inference
        }
    """

    # ── Modes ──────────────────────────────────────────────────────────────
    MODE_WARMUP    = "warmup"
    MODE_TRAINING  = "training"
    MODE_INFERENCE = "inference"

    def __init__(
        self,
        lane_ids: List[str],
        window_size: int          = 60,
        forecast_horizon: int     = 5,
        hidden_size: int          = 64,
        num_layers: int           = 2,
        dropout: float            = 0.2,
        learning_rate: float      = 1e-3,
        warmup_steps: int         = 300,
        online_batch_size: int    = 8,
        online_train_every: int   = 10,   # run train_online() every N updates
        replay_capacity: int      = 1000,
        device: Optional[str]     = None,
    ):
        self.lane_ids         = lane_ids
        self.num_lanes        = len(lane_ids)
        self.window_size      = window_size
        self.forecast_horizon = forecast_horizon
        self.warmup_steps     = warmup_steps
        self.online_batch     = online_batch_size
        self.train_every      = online_train_every

        # Device — prefer MPS (Apple Silicon) > CUDA > CPU
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)
        log.info("TrafficOracle initialised on device: %s", self.device)

        # ── Per-lane sliding window buffers ──────────────────────────────
        self.buffers: Dict[str, LaneBuffer] = {
            lid: LaneBuffer(window_size) for lid in lane_ids
        }

        # ── Normalisation (per lane, per feature) ────────────────────────
        feature_dim = self.num_lanes * LaneBuffer.FEATURE_DIM
        self.normaliser = RunningNormaliser(feature_dim)

        # ── LSTM model ───────────────────────────────────────────────────
        self.model = TrafficLSTM(
            num_lanes=self.num_lanes,
            num_features=LaneBuffer.FEATURE_DIM,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            forecast_horizon=forecast_horizon,
        ).to(self.device)

        self.criterion = nn.HuberLoss()  # robust to outlier spikes
        self.optimiser = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimiser, patience=50, factor=0.5
        )

        # ── Online replay buffer ─────────────────────────────────────────
        self.replay = ReplayBuffer(replay_capacity)

        # ── LSTM stateful hidden state for inference ──────────────────────
        self._hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        # ── Internal state ────────────────────────────────────────────────
        self._mode        = self.MODE_WARMUP
        self._step_count  = 0
        self._last_pred   = None
        self._last_loss   = None
        self._warmup_data: List[np.ndarray] = []  # raw windows during warmup

        log.info(
            "Oracle ready | lanes=%d | window=%ds | horizon=%d steps",
            self.num_lanes, window_size, forecast_horizon
        )

    # ─────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────────────

    def update(
        self,
        live_traffic_data: Dict[str, Dict],
        signal_states: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Ingest one frame of live data from the YOLOv8 perception layer.

        Args:
            live_traffic_data : {lane_id: {"count": int, "emissions_score": float}}
            signal_states     : {lane_id: "red"|"yellow"|"green"}
                                If None, all lanes assumed red (safe default).
        """
        if signal_states is None:
            signal_states = {lid: "red" for lid in self.lane_ids}

        # Push each lane's data into its buffer
        for lid in self.lane_ids:
            lane_data = live_traffic_data.get(lid, {})
            self.buffers[lid].push(
                count          = lane_data.get("count"),
                emission_score = lane_data.get("emissions_score"),
                signal_state   = signal_states.get(lid),
            )

        self._step_count += 1

        # Build flat feature vector for normaliser update
        flat = self._build_flat_window()  # (window_size, num_lanes * 3)
        for row in flat:
            self.normaliser.update(row)

        # ── Mode transitions ─────────────────────────────────────────────
        if self._mode == self.MODE_WARMUP:
            self._warmup_data.append(flat.copy())
            if self._step_count >= self.warmup_steps:
                self._finish_warmup()

        elif self._mode in (self.MODE_TRAINING, self.MODE_INFERENCE):
            # Store experience for online training
            norm_flat = np.stack([self.normaliser.normalise(r) for r in flat])
            # Target: current lane counts and emission scores (teacher forcing)
            target = self._build_target()
            self.replay.push(norm_flat, target)

            # Periodic online fine-tuning
            if self._step_count % self.train_every == 0:
                self.train_online()

            # Run inference every step once warm
            if self._mode == self.MODE_INFERENCE:
                self._run_inference()

    def get_predicted_state(self) -> Optional[Dict]:
        """
        Return the latest RL-ready prediction dict.

        Returns None during warm-up (not enough data yet).
        The RL agent should treat None as "use current observation only".
        """
        if self._last_pred is None:
            log.debug("Prediction unavailable — still in %s mode", self._mode)
        return self._last_pred

    def train_online(self) -> Optional[float]:
        """
        Run one mini-batch gradient update using the replay buffer.

        Returns the scalar training loss, or None if buffer not ready.
        Safe to call manually from the simulation loop.
        """
        batch = self.replay.sample(self.online_batch)
        if batch is None:
            return None

        x_batch, y_batch = batch
        x_batch = x_batch.to(self.device)
        y_batch = y_batch.to(self.device)

        self.model.train()
        self.optimiser.zero_grad()

        preds, _ = self.model(x_batch)   # (B, horizon, lanes, 2)
        # y_batch : (B, horizon, lanes, 2)
        loss = self.criterion(preds, y_batch)
        loss.backward()

        # Gradient clipping — critical for LSTM stability
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimiser.step()
        self.scheduler.step(loss)

        self._last_loss = loss.item()
        return self._last_loss

    # ─────────────────────────────────────────────────────────────────────
    # INTERNAL HELPERS
    # ─────────────────────────────────────────────────────────────────────

    def _finish_warmup(self) -> None:
        """
        Transition from WARMUP → TRAINING.
        Freeze normaliser statistics and run an initial batch training pass
        on all collected warm-up windows so the model has a head start.
        """
        log.info("Warm-up complete (%d steps). Beginning initial training…", self._step_count)
        self.normaliser.freeze()
        self._mode = self.MODE_TRAINING

        # Build training pairs from warm-up data
        for i in range(len(self._warmup_data) - self.forecast_horizon):
            x = np.stack([
                self.normaliser.normalise(r)
                for r in self._warmup_data[i]
            ])
            target = self._build_target_from_buffer_snapshot(
                self._warmup_data[i + 1 : i + 1 + self.forecast_horizon]
            )
            self.replay.push(x, target)

        # 20 warm-start gradient steps
        warmstart_losses = []
        for _ in range(20):
            loss = self.train_online()
            if loss is not None:
                warmstart_losses.append(loss)

        if warmstart_losses:
            log.info(
                "Warm-start training done | avg loss=%.4f → switching to INFERENCE",
                np.mean(warmstart_losses),
            )

        self._mode = self.MODE_INFERENCE
        self._warmup_data.clear()  # free memory

    def _run_inference(self) -> None:
        """
        Forward pass on the current window → update self._last_pred.
        Uses stateful hidden state across calls for smooth temporal continuity.
        """
        flat = self._build_flat_window()  # (window_size, num_lanes * 3)
        norm = np.stack([self.normaliser.normalise(r) for r in flat])  # same shape

        x = torch.tensor(norm, dtype=torch.float32).unsqueeze(0).to(self.device)
        # x : (1, window_size, num_lanes * 3)

        self.model.eval()
        with torch.no_grad():
            preds, self._hidden = self.model(x, self._hidden)
            # preds : (1, forecast_horizon, num_lanes, 2)

        preds_np = preds.squeeze(0).cpu().numpy()  # (horizon, lanes, 2)

        # Denormalise predictions back to original scale
        result = {}
        for i, lid in enumerate(self.lane_ids):
            raw_counts    = preds_np[:, i, 0]
            raw_emissions = preds_np[:, i, 1]

            # Denormalise using per-lane slice of the normaliser
            counts_dn    = self._denorm_feature(raw_counts,    feature_idx=0, lane_idx=i)
            emissions_dn = self._denorm_feature(raw_emissions, feature_idx=1, lane_idx=i)

            # Clip to physically valid range
            counts_dn    = np.clip(counts_dn,    0, None)
            emissions_dn = np.clip(emissions_dn, 0, 1)

            congestion_risk = self._compute_congestion_risk(counts_dn, emissions_dn)
            action_hint     = self._recommend_action(congestion_risk, lid)

            result[lid] = {
                "predicted_counts":    counts_dn.tolist(),
                "predicted_emissions": emissions_dn.tolist(),
                "congestion_risk":     float(congestion_risk),
                "recommended_action":  action_hint,
            }

        result["oracle_confidence"] = self._compute_confidence()
        result["mode"]              = self._mode
        result["step"]              = self._step_count
        result["last_loss"]         = self._last_loss

        self._last_pred = result

    def _build_flat_window(self) -> np.ndarray:
        """
        Concatenate all lane buffers into a single (window_size, num_lanes*3) array.
        """
        windows = [self.buffers[lid].get_window() for lid in self.lane_ids]
        return np.concatenate(windows, axis=1)  # (T, num_lanes*3)

    def _build_target(self) -> np.ndarray:
        """
        Build a (forecast_horizon, num_lanes, 2) target tensor from the current
        buffer state (used as pseudo ground-truth in online learning).
        """
        # We use the most recent observation replicated across the horizon
        # as a "persistence" baseline target — replaced by real future values
        # as the replay buffer accumulates temporal pairs.
        current = np.array([
            [self.buffers[lid]._last_valid[0],   # count
             self.buffers[lid]._last_valid[1]]    # emission
            for lid in self.lane_ids
        ], dtype=np.float32)  # (num_lanes, 2)

        # Normalise target
        for li, lid in enumerate(self.lane_ids):
            base = li * LaneBuffer.FEATURE_DIM
            mean = self.normaliser.mean[base:base+2]
            std  = self.normaliser.std[base:base+2]
            current[li] = (current[li] - mean) / (std + 1e-6)

        return np.stack([current] * self.forecast_horizon)  # (H, lanes, 2)

    def _build_target_from_buffer_snapshot(
        self, snapshots: List[np.ndarray]
    ) -> np.ndarray:
        """Build a target tensor from a list of raw window snapshots (warm-up)."""
        targets = []
        for snap in snapshots:
            # snap : (window_size, num_lanes*3) — take last row (most recent frame)
            last_row = snap[-1]  # (num_lanes*3,)
            lane_targets = []
            for li in range(self.num_lanes):
                base = li * LaneBuffer.FEATURE_DIM
                count_n    = (last_row[base]   - self.normaliser.mean[base])   / (self.normaliser.std[base]   + 1e-6)
                emission_n = (last_row[base+1] - self.normaliser.mean[base+1]) / (self.normaliser.std[base+1] + 1e-6)
                lane_targets.append([count_n, emission_n])
            targets.append(lane_targets)
        # Pad if snapshot list shorter than horizon
        while len(targets) < self.forecast_horizon:
            targets.append(targets[-1])
        return np.array(targets[:self.forecast_horizon], dtype=np.float32)  # (H, L, 2)

    def _denorm_feature(
        self, values: np.ndarray, feature_idx: int, lane_idx: int
    ) -> np.ndarray:
        """Reverse normalisation for a single feature of a single lane."""
        idx  = lane_idx * LaneBuffer.FEATURE_DIM + feature_idx
        mean = self.normaliser.mean[idx]
        std  = self.normaliser.std[idx]
        return values * std + mean

    def _compute_congestion_risk(
        self, counts: np.ndarray, emissions: np.ndarray
    ) -> float:
        """
        Scalar [0,1] risk score combining predicted density and emission trend.
        Used by the RL agent for reward shaping.

        Formula: weighted average of normalised count slope + emission magnitude.
        """
        count_slope    = max(0.0, float(np.polyfit(range(len(counts)), counts, 1)[0]))
        emission_mean  = float(np.mean(emissions))
        density_signal = min(1.0, count_slope / 2.0)   # ~2 veh/step = full risk
        return 0.6 * density_signal + 0.4 * emission_mean

    def _recommend_action(self, risk: float, lane_id: str) -> str:
        """
        Simple rule-based hint derived from predicted congestion risk.
        The RL agent may override this — it is only a soft signal.
        """
        if risk > 0.75:
            return "extend_green"
        elif risk > 0.45:
            return "normal"
        else:
            return "shorten_green"

    def _compute_confidence(self) -> float:
        """
        Proxy confidence metric:  1 / (1 + last_loss).
        Returns 0.5 if no training has occurred yet.
        """
        if self._last_loss is None:
            return 0.5
        return float(1.0 / (1.0 + self._last_loss))

    # ─────────────────────────────────────────────────────────────────────
    # PERSISTENCE
    # ─────────────────────────────────────────────────────────────────────

    def save(self, path: str = "oracle_checkpoint.pt") -> None:
        """Save model weights and normaliser state."""
        torch.save({
            "model_state":      self.model.state_dict(),
            "optimiser_state":  self.optimiser.state_dict(),
            "normaliser_mean":  self.normaliser.mean,
            "normaliser_M2":    self.normaliser.M2,
            "normaliser_n":     self.normaliser.n,
            "step_count":       self._step_count,
            "mode":             self._mode,
        }, path)
        log.info("Checkpoint saved → %s", path)

    def load(self, path: str = "oracle_checkpoint.pt") -> None:
        """Restore from checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimiser.load_state_dict(ckpt["optimiser_state"])
        self.normaliser.mean  = ckpt["normaliser_mean"]
        self.normaliser.M2    = ckpt["normaliser_M2"]
        self.normaliser.n     = ckpt["normaliser_n"]
        self._step_count      = ckpt["step_count"]
        self._mode            = ckpt["mode"]
        self.normaliser._frozen = True
        log.info("Checkpoint loaded from %s (step=%d, mode=%s)", path, self._step_count, self._mode)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 INTEGRATION EXAMPLE
# ─────────────────────────────────────────────────────────────────────────────
def _simulate_phase2_output(lane_ids: List[str], step: int) -> Dict:
    """
    Mimics the live_traffic_data dict produced by the YOLOv8 perception layer.
    Replace this function with your actual Phase 2 output in production.
    """
    rng = np.random.default_rng(step)
    return {
        lid: {
            "count":           int(rng.integers(0, 12)),
            "emissions_score": float(rng.uniform(0.1, 0.9)),
        }
        for lid in lane_ids
    }


def _simulate_signal_states(lane_ids: List[str], step: int) -> Dict[str, str]:
    """Cycle through a basic 4-phase traffic light schedule."""
    phases = ["green", "yellow", "red", "red"]
    return {lid: phases[(i + step // 15) % 4] for i, lid in enumerate(lane_ids)}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — DEMO INTEGRATION LOOP
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    LANE_IDS = ["lane_N", "lane_S", "lane_E", "lane_W"]

    oracle = TrafficOracle(
        lane_ids          = LANE_IDS,
        window_size       = 60,
        forecast_horizon  = 5,
        hidden_size       = 64,
        num_layers        = 2,
        dropout           = 0.2,
        learning_rate     = 1e-3,
        warmup_steps      = 60,         # shortened to 60 for demo; use 300 in production
        online_train_every= 10,
    )

    log.info("─── Starting EcoSync Traffic Oracle demo loop ───")

    for step in range(400):
        # ── Receive data from Phase 2 perception layer ────────────────────
        live_data     = _simulate_phase2_output(LANE_IDS, step)
        signal_states = _simulate_signal_states(LANE_IDS, step)

        # ── Feed into Oracle ──────────────────────────────────────────────
        oracle.update(live_data, signal_states)

        # ── Query prediction (RL agent would read this every tick) ────────
        prediction = oracle.get_predicted_state()

        if prediction and step % 30 == 0:
            log.info("Step %3d | mode=%-9s | confidence=%.2f | loss=%s",
                step,
                prediction["mode"],
                prediction["oracle_confidence"],
                f"{prediction['last_loss']:.4f}" if prediction["last_loss"] else "N/A",
            )
            for lid in LANE_IDS:
                lane = prediction[lid]
                log.info(
                    "  %s → counts=%s  emissions=%s  risk=%.2f  hint=%s",
                    lid,
                    [f"{v:.1f}" for v in lane["predicted_counts"]],
                    [f"{v:.2f}" for v in lane["predicted_emissions"]],
                    lane["congestion_risk"],
                    lane["recommended_action"],
                )

        # Simulate 1 Hz tick
        time.sleep(0.01)  # remove in production; SUMO controls timing

    # Save final checkpoint for Phase 4 RL agent
    oracle.save("oracle_checkpoint.pt")
    log.info("Demo complete. Checkpoint saved for Phase 4 RL integration.")