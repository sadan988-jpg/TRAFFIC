"""
EcoSync — Flask API Backend
Serves live simulation data to the HTML/JS dashboard frontend.
"""

import math
import time
import threading
from collections import deque
from typing import Dict, List
import numpy as np

from flask import Flask, jsonify, send_from_directory, Response
from flask_cors import CORS

# ── Demo Data Engine (ported from dashboard.py) ─────────────────────────────

INCOMING_LANES: List[str] = [
    "N_in_0", "N_in_1", "N_in_2",
    "S_in_0", "S_in_1", "S_in_2",
    "E_in_0", "E_in_1", "E_in_2",
    "W_in_0", "W_in_1", "W_in_2",
]


class DemoDataEngine:
    """Generates realistic fake data for demo/presentation when SUMO isn't running."""

    def __init__(self):
        self.step = 0
        self.ai_wait = 0.0
        self.baseline_wait = 0.0
        self.ai_emissions = 0.0
        self.baseline_emissions = 0.0
        self.ai_jam = 0.0
        self.baseline_jam = 0.0
        self.actual_density: deque = deque(maxlen=200)
        self.predicted_density: deque = deque(maxlen=200)
        self.log_entries: List[dict] = []
        self._rng = np.random.default_rng(42)
        self.running = True
        self._lock = threading.Lock()

    def tick(self):
        with self._lock:
            self.step += 1
            t = self.step
            base = 5 + 4 * math.sin(t * 0.05) + 2 * math.sin(t * 0.13)
            noise = float(self._rng.normal(0, 0.5))
            actual = max(0.0, base + noise)
            predicted = max(0.0, base + 0.8 * math.sin((t + 5) * 0.05) * 4 + 5 + float(self._rng.normal(0, 0.3)))
            self.actual_density.append(round(actual, 3))
            self.predicted_density.append(round(predicted, 3))

            bw = abs(float(self._rng.normal(12, 4)))
            aw = abs(float(self._rng.normal(7, 3)))
            self.baseline_wait += bw
            self.ai_wait += aw
            be = abs(float(self._rng.normal(5.5, 1.5)))
            ae = abs(float(self._rng.normal(3.2, 1.2)))
            self.baseline_emissions += be
            self.ai_emissions += ae
            bj = abs(float(self._rng.normal(0.6, 0.2)))
            aj = abs(float(self._rng.normal(0.3, 0.15)))
            self.baseline_jam += bj
            self.ai_jam += aj

            if t % 3 == 0:
                actions = [
                    ("Phase 0 (NS Green)", "Oracle predicted high-density bus cluster on N_in_1"),
                    ("Phase 2 (EW Green)", "Clearing E_in queue — predicted jam risk 0.78"),
                    ("Phase 0 (NS Green)", "Emergency vehicle detected on S_in_0 — priority override"),
                    ("Phase 2 (EW Green)", "Low density on NS arms — switching to serve EW backlog"),
                    ("Phase 0 (NS Green)", "Predicted platoon arrival from North in 3 steps"),
                ]
                act, reason = actions[t % len(actions)]
                self.log_entries.append({
                    "step": t,
                    "action": act,
                    "reason": reason,
                })
                if len(self.log_entries) > 50:
                    self.log_entries = self.log_entries[-50:]

    def get_impact_saved(self) -> float:
        dw = max(0.0, self.baseline_wait - self.ai_wait)
        de = max(0.0, self.baseline_emissions - self.ai_emissions)
        dj = max(0.0, self.baseline_jam - self.ai_jam)
        return 0.50 * dw + 0.30 * de + 0.20 * dj

    def get_lane_data(self) -> Dict:
        result = {}
        for lid in INCOMING_LANES:
            count = max(0, int(self._rng.normal(4, 2)))
            em = round(abs(float(self._rng.normal(2.5, 1.5))), 2)
            result[lid] = {"count": count, "emissions_score": em}
        return result

    def snapshot(self) -> dict:
        """Thread-safe snapshot of all current state."""
        with self._lock:
            bw, aw = self.baseline_wait, self.ai_wait
            be, ae = self.baseline_emissions, self.ai_emissions
            bj, aj = self.baseline_jam, self.ai_jam

            pct_w = ((bw - aw) / max(bw, 1.0)) * 100
            pct_e = ((be - ae) / max(be, 1.0)) * 100
            pct_j = ((bj - aj) / max(bj, 1.0)) * 100

            return {
                "step": self.step,
                "running": self.running,
                "impact_saved": round(self.get_impact_saved(), 1),
                "oracle_mode": "INFERENCE" if self.step > 60 else "WARMUP",
                "agent_epsilon": round(max(0.05, 1.0 * 0.995 ** self.step), 3),
                "metrics": {
                    "wait_pct": round(pct_w, 1),
                    "wait_abs": round(bw - aw, 1),
                    "emissions_pct": round(pct_e, 1),
                    "emissions_abs": round(be - ae, 1),
                    "jam_pct": round(pct_j, 1),
                    "jam_abs": round(bj - aj, 2),
                },
                "density": {
                    "actual": list(self.actual_density),
                    "predicted": list(self.predicted_density),
                },
                "log_entries": list(self.log_entries[-20:]),
                "lane_data": self.get_lane_data(),
            }


# ── Background tick thread ───────────────────────────────────────────────────

engine = DemoDataEngine()

def _tick_loop(fps: float = 10.0):
    interval = 1.0 / fps
    while True:
        if engine.running:
            engine.tick()
        time.sleep(interval)

tick_thread = threading.Thread(target=_tick_loop, daemon=True)
tick_thread.start()


# ── Flask App ────────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder=".")
CORS(app)


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/api/state")
def api_state():
    return jsonify(engine.snapshot())


@app.route("/api/control", methods=["POST"])
def api_control():
    from flask import request
    data = request.get_json(force=True, silent=True) or {}
    if "running" in data:
        engine.running = bool(data["running"])
    return jsonify({"running": engine.running})


if __name__ == "__main__":
    print("=" * 60)
    print("  EcoSync HTML Dashboard Backend")
    print("  Open: http://localhost:5000")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
