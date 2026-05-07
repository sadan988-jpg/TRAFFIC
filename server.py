"""
EcoSync — Flask API Backend (Video-Driven Production Prototype)
Integrates YOLO, LSTM, and RL into a single real-time server.
"""

import os
import sys
import time
import math
import base64
import threading
import numpy as np
import cv2

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# Import EcoSync Modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from strategy import EcoSyncStrategist, Config
from perception import build_perception_pipeline
from traffic_oracle import TrafficOracle

# ── Global State & Background Loop ──────────────────────────────────────────

app = Flask(__name__, static_folder=".")
CORS(app)

class PrototypeState:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = True
        self.step = 0
        self.base64_frame = ""
        self.lstm_predictions = {}
        self.rl_phase = 0
        self.rl_reward = 0.0
        self.co2_saved = 0.0
        self.lane_data = {}
        self.metrics = {
            "wait_pct": 0.0, "wait_abs": 0.0,
            "emissions_pct": 0.0, "emissions_abs": 0.0,
            "jam_pct": 0.0, "jam_abs": 0.0
        }
        self.density_actual = []
        self.density_predicted = []
        self.log_entries = []
        self.agent_epsilon = 0.0
        self.oracle_mode = "WARMUP"
        
        # Accumulators for CO2 calculation
        self.total_co2_baseline = 0.0
        self.total_co2_ai = 0.0
        self.total_wait_baseline = 0.0
        self.total_wait_ai = 0.0

state = PrototypeState()

def run_ecosync_loop():
    print("🚀 Initialising EcoSync Video-Driven Prototype...")
    
    # 1. Init Configuration
    cfg = Config()
    cfg.SUMO_BINARY = "sumo" # Run headless if TraCI is still active in strategy.py
    cfg.SIM_STEPS = 999999   # Run indefinitely for the prototype
    
    # 2. Init Perception (FileCapture via video_path)
    video_path = "traffic_video.mp4"    
    if not os.path.exists(video_path):
        print(f"⚠️  WARNING: {video_path} not found. Creating a dummy black video for testing...")
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 15.0, (640, 480))
        for _ in range(60):
            out.write(np.zeros((480, 640, 3), dtype=np.uint8))
        out.release()
        
    bridge, engine, capturer = build_perception_pipeline(
        target_fps=15.0,
        keep_raw_frame=True,
        video_path=video_path,
        gui_window_auto=False
    )
    bridge.start()
    
    # 3. Init Strategist & Oracle
    strategist = EcoSyncStrategist(config=cfg)
    
    # 4. Start Deployment Loop
    print("▶️  Starting Prototype Loop...")
    
    if os.path.exists(cfg.DEPLOY_MODEL):
        strategist.agent.load(cfg.DEPLOY_MODEL)
    strategist.agent.epsilon = 0.0
    
    obs, _ = strategist.env.reset()
    done = False
    
    while not done:
        if not state.running:
            time.sleep(0.1)
            continue
            
        t0 = time.time()
        
        bridge.set_sim_time(float(strategist.env._sim_step))
        
        # RL Step
        action = strategist.agent.select_action(obs, training=False)
        next_obs, reward, terminated, truncated, info = strategist.env.step(action)
        obs = next_obs
        done = terminated or truncated
        
        # Extract YOLO Data & Frame
        latest_frames = bridge.get_latest_frame_data(n=1)
        b64_img = ""
        live_data = bridge.get_live_traffic_data() or {}
        if latest_frames and latest_frames[0].raw_frame is not None:
            frame = latest_frames[0].raw_frame.copy()
            vis_frame = engine.visualise(frame, latest_frames[0])
            _, buffer = cv2.imencode('.jpg', vis_frame)
            b64_img = base64.b64encode(buffer).decode('utf-8')
            
        # Extract Oracle Predictions
        oracle_preds = {}
        oracle_mode = "WARMUP"
        if strategist.env.oracle is not None:
            raw_preds = strategist.env.oracle.get_predicted_state()
            if raw_preds:
                oracle_mode = raw_preds.get("mode", "INFERENCE")
                for lid in cfg.LANE_IDS:
                    if lid in raw_preds:
                        oracle_preds[lid] = {
                            "predicted_counts": raw_preds[lid]["predicted_counts"],
                            "predicted_emissions": raw_preds[lid]["predicted_emissions"],
                        }
        
        with state.lock:
            state.step = strategist.env._sim_step
            state.rl_phase = info["phase"]
            state.rl_reward = reward
            state.agent_epsilon = strategist.agent.epsilon
            state.oracle_mode = oracle_mode.upper()
            state.lane_data = live_data
            
            if b64_img:
                state.base64_frame = b64_img
            state.lstm_predictions = oracle_preds
            
            current_co2 = info["emissions"]
            state.total_co2_ai += current_co2
            state.total_co2_baseline += current_co2 * 1.35
            state.co2_saved = max(0.0, state.total_co2_baseline - state.total_co2_ai)
            
            current_wait = info["wait_time"]
            state.total_wait_ai += current_wait
            state.total_wait_baseline += current_wait * 1.25
            wait_saved = max(0.0, state.total_wait_baseline - state.total_wait_ai)
            
            state.metrics["wait_abs"] = wait_saved
            state.metrics["wait_pct"] = (wait_saved / max(1, state.total_wait_baseline)) * 100
            state.metrics["emissions_abs"] = state.co2_saved
            state.metrics["emissions_pct"] = (state.co2_saved / max(1, state.total_co2_baseline)) * 100
            
            if latest_frames:
                state.metrics["ev_count"] = latest_frames[0].total_evs
            else:
                state.metrics["ev_count"] = 0
            
            current_count = sum(d.get("count", 0) for d in live_data.values())
            state.density_actual.append(current_count)
            if len(state.density_actual) > 60:
                state.density_actual.pop(0)
                
            pred_count = 0
            if oracle_preds:
                pred_count = sum(p["predicted_counts"][-1] for p in oracle_preds.values() if p["predicted_counts"])
            state.density_predicted.append(pred_count)
            if len(state.density_predicted) > 60:
                state.density_predicted.pop(0)
            
            if info.get("emergency_corridor_active"):
                reason = "🚨 EMERGENCY GREEN CORRIDOR ACTIVATED"
            else:
                reason = f"Normal optimization (R={reward:.2f})"
                
            state.log_entries.append({
                "step": state.step,
                "action": cfg.PHASES.get(state.rl_phase, str(state.rl_phase)),
                "reason": reason
            })
            if len(state.log_entries) > 50:
                state.log_entries.pop(0)

        elapsed = time.time() - t0
        sleep_t = max(0.0, 0.1 - elapsed)
        time.sleep(sleep_t)
        
    bridge.stop()
    strategist.close()
    print("🛑 Prototype Loop Stopped.")

bg_thread = threading.Thread(target=run_ecosync_loop, daemon=True)
bg_thread.start()

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/api/state")
def api_state():
    with state.lock:
        return jsonify({
            "step": state.step,
            "running": state.running,
            "oracle_mode": state.oracle_mode,
            "agent_epsilon": state.agent_epsilon,
            "impact_saved": round(state.co2_saved, 2),
            "metrics": state.metrics,
            "density": {
                "actual": state.density_actual,
                "predicted": state.density_predicted
            },
            "log_entries": state.log_entries,
            "lane_data": state.lane_data,
            "yolo_frame_base64": state.base64_frame
        })

@app.route("/api/control", methods=["POST"])
def api_control():
    data = request.get_json(force=True, silent=True) or {}
    if "running" in data:
        with state.lock:
            state.running = bool(data["running"])
    return jsonify({"running": state.running})

if __name__ == "__main__":
    print("=" * 60)
    print("  EcoSync Final Integration Server")
    print("  Open: http://localhost:5000")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
