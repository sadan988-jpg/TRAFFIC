"""
EcoSync — FastAPI Backend (Video-Driven Production Prototype)
Refactored for high-performance asynchronous operations.
"""

import os
import sys
import time
import base64
import threading
import asyncio
import numpy as np
import cv2

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles

# Import EcoSync Modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from strategy import EcoSyncStrategist, Config
from perception import build_perception_pipeline
from traffic_oracle import TrafficOracle

# ── Global State ──────────────────────────────────────────────────────────

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
            "jam_pct": 0.0, "jam_abs": 0.0,
            "ev_count": 0
        }
        self.density_actual = []
        self.density_predicted = []
        self.log_entries = []
        self.agent_epsilon = 0.0
        self.oracle_mode = "WARMUP"
        
        self.night_active = False
        self.fog_active = False
        self.rain_active = False

        self.total_co2_baseline = 0.0
        self.total_co2_ai = 0.0
        self.total_wait_baseline = 0.0
        self.total_wait_ai = 0.0

        # ── Feature 1: Vehicle Throughput Tracking ────────────────────────
        self.vehicle_throughput = {"N": 0, "S": 0, "E": 0, "W": 0}  # cumulative per direction
        self.total_vehicles_passed = 0
        self.congestion_risk = {}  # per lane_id: "LOW" | "MEDIUM" | "HIGH"

        # ── Feature 2: EV vs Fuel CO2 Tracking ────────────────────────────
        self.ev_count_live = 0
        self.fuel_count_live = 0
        self.co2_rate_live = 0.0   # g/km for current frame vehicles
        self.cumulative_co2_fuel = 0.0   # g/km accumulated from fuel vehicles
        self.cumulative_co2_saved_ev = 0.0  # g/km that EVs did NOT emit

        # ── Feature 3: Ambulance Emergency State ──────────────────────────
        self.ambulance_active = False
        self.ambulance_alert_msg = ""

state = PrototypeState()
app = FastAPI(title="EcoSync API")

# Global bridge reference so the /api/control endpoint can pause it
_bridge = None

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount React Static Files
if os.path.exists("frontend/dist"):
    app.mount("/assets", StaticFiles(directory="frontend/dist/assets"), name="assets")

# ── Background Logic ──────────────────────────────────────────────────────

def run_ecosync_loop():
    print("🚀 Initializing EcoSync FastAPI Prototype...")
    
    cfg = Config()
    cfg.SUMO_BINARY = "sumo" 
    cfg.SIM_STEPS = 999999   
    
    video_path = "traffi.mp4"    
    if not os.path.exists(video_path):
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

    # Store bridge globally so the control endpoint can pause it
    global _bridge
    _bridge = bridge
    
    strategist = EcoSyncStrategist(config=cfg)
    
    if os.path.exists(cfg.DEPLOY_MODEL):
        strategist.agent.load(cfg.DEPLOY_MODEL)
    strategist.agent.epsilon = 0.0
    
    obs, _ = strategist.env.reset()
    done = False
    
    # Track which frame IDs we've already counted to avoid double-counting
    last_counted_frame_id = -1

    while not done:
        if not state.running:
            time.sleep(0.1)
            continue
            
        t0 = time.time()
        bridge.set_sim_time(float(strategist.env._sim_step))
        with state.lock:
            night, fog, rain = state.night_active, state.fog_active, state.rain_active
        bridge.set_atmospheric_modes(night, fog, rain)
        strategist.env.set_atmospheric_modes(night, fog, rain)
        
        action = strategist.agent.select_action(obs, training=False)
        next_obs, reward, terminated, truncated, info = strategist.env.step(action)
        obs = next_obs
        done = terminated or truncated
        
        latest_frames = bridge.get_latest_frame_data(n=1)
        b64_img = ""
        live_data = bridge.get_live_traffic_data() or {}
        
        if latest_frames and latest_frames[0].raw_frame is not None:
            frame = latest_frames[0].raw_frame.copy()
            vis_frame = engine.visualise(frame, latest_frames[0])
            _, buffer = cv2.imencode('.jpg', vis_frame)
            b64_img = base64.b64encode(buffer).decode('utf-8')
            
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

            # ── Feature 3: Ambulance detection ────────────────────────────
            frame_ambu = latest_frames[0].ambulance_detected if latest_frames else False
            state.ambulance_active = frame_ambu or info.get("emergency_corridor_active", False)
            if state.ambulance_active:
                state.ambulance_alert_msg = "🚨 AMBULANCE DETECTED — VEHICLES AHEAD CLEAR THE LANE"
            else:
                state.ambulance_alert_msg = ""

            # ── Feature 1: Vehicle throughput per direction ────────────────
            #    Only count NEW frames (avoid re-counting same vehicles)
            if latest_frames:
                lf = latest_frames[0]
                current_frame_id = lf.frame_id

                # Only count this frame if we haven't already
                if current_frame_id != last_counted_frame_id:
                    last_counted_frame_id = current_frame_id

                    # Show current visible vehicles (snapshot, not cumulative)
                    state.total_vehicles_passed = lf.total_vehicles

                    # Per-direction: show current snapshot count (not cumulative)
                    for arm in ["N", "S", "E", "W"]:
                        arm_count = sum(
                            v.get("count", 0)
                            for lid, v in live_data.items()
                            if lid.startswith(arm)
                        )
                        state.vehicle_throughput[arm] = arm_count

                # Congestion risk (always update from latest data)
                state.congestion_risk = {}
                for lid, v in live_data.items():
                    cnt = v.get("count", 0)
                    state.congestion_risk[lid] = "HIGH" if cnt > 6 else ("MEDIUM" if cnt > 3 else "LOW")

            # ── Feature 2: EV vs Fuel CO2 tracking ────────────────────────
            if latest_frames:
                lf = latest_frames[0]
                state.ev_count_live = lf.total_evs
                state.fuel_count_live = lf.total_fuel_vehicles
                state.co2_rate_live = round(lf.co2_rate_g_per_frame, 1)
                state.cumulative_co2_fuel += lf.co2_rate_g_per_frame
                # CO2 saved = what EVs would have emitted if they were fuel cars
                avg_car_co2 = 140.0  # HBEFA 4.1 g/km for car
                state.cumulative_co2_saved_ev += lf.total_evs * avg_car_co2

            # Impact Calculations
            current_co2 = info["emissions"]
            state.total_co2_ai += current_co2
            state.total_co2_baseline += current_co2 * 1.35
            state.co2_saved = max(0.0, state.total_co2_baseline - state.total_co2_ai)
            
            current_wait = info["wait_time"]
            state.total_wait_ai += current_wait
            state.total_wait_baseline += current_wait * 1.25
            wait_saved = max(0.0, state.total_wait_baseline - state.total_wait_ai)
            
            state.metrics.update({
                "wait_abs": wait_saved,
                "wait_pct": (wait_saved / max(1, state.total_wait_baseline)) * 100,
                "emissions_abs": state.co2_saved,
                "emissions_pct": (state.co2_saved / max(1, state.total_co2_baseline)) * 100,
                "ev_count": latest_frames[0].total_evs if latest_frames else 0
            })
            
            current_count = sum(d.get("count", 0) for d in live_data.values())
            state.density_actual.append(current_count)
            
            pred_count = sum(p["predicted_counts"][-1] for p in oracle_preds.values() if p["predicted_counts"]) if oracle_preds else 0
            state.density_predicted.append(pred_count)
            
            # Log Management
            reason = info.get("atmospheric_log", f"Optimization (R={reward:.2f})")
            if info.get("emergency_corridor_active"):
                reason = "🚨 EMERGENCY GREEN CORRIDOR | " + reason
            state.log_entries.append({"step": state.step, "action": cfg.PHASES.get(state.rl_phase, str(state.rl_phase)), "reason": reason})
            
            if len(state.density_actual) > 60: state.density_actual.pop(0)
            if len(state.density_predicted) > 60: state.density_predicted.pop(0)
            if len(state.log_entries) > 50: state.log_entries.pop(0)

        elapsed = time.time() - t0
        time.sleep(max(0.0, 0.1 - elapsed))
        
    bridge.stop()
    strategist.close()

# Start background thread
threading.Thread(target=run_ecosync_loop, daemon=True).start()

# ── API Endpoints ────────────────────────────────────────────────────────

class ControlRequest(BaseModel):
    running: bool
    night_mode: bool = False
    fog_mode: bool = False
    rain_mode: bool = False

@app.get("/", response_class=HTMLResponse)
async def index():
    if os.path.exists("frontend/dist/index.html"):
        return FileResponse("frontend/dist/index.html")
    return FileResponse("index.html")

@app.get("/api/state")
async def get_state():
    with state.lock:
        return {
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
            "yolo_frame_base64": state.base64_frame,
            "night_active": state.night_active,
            "fog_active": state.fog_active,
            "rain_active": state.rain_active,
            # Feature 1: Vehicle throughput & congestion
            "vehicle_throughput": state.vehicle_throughput,
            "total_vehicles_passed": state.total_vehicles_passed,
            "congestion_risk": state.congestion_risk,
            # Feature 2: EV vs Fuel CO2
            "ev_count_live": state.ev_count_live,
            "fuel_count_live": state.fuel_count_live,
            "co2_rate_live": state.co2_rate_live,
            "cumulative_co2_fuel": round(state.cumulative_co2_fuel, 1),
            "cumulative_co2_saved_ev": round(state.cumulative_co2_saved_ev, 1),
            # Feature 3: Ambulance
            "ambulance_active": state.ambulance_active,
            "ambulance_alert_msg": state.ambulance_alert_msg,
        }

@app.post("/api/control")
async def control_system(data: ControlRequest):
    with state.lock:
        state.running = data.running
        state.night_active = data.night_mode
        state.fog_active = data.fog_mode
        state.rain_active = data.rain_mode
    # Also pause/resume the perception bridge so video stops advancing
    if _bridge is not None:
        _bridge.set_paused(not data.running)
        _bridge.set_atmospheric_modes(data.night_mode, data.fog_mode, data.rain_mode)
    return {"running": state.running}

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("  EcoSync Final Integration Server (FastAPI)")
    print("  Open: http://localhost:5000")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=5000)