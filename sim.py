"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          EcoSync — Phase 1: Environment & Simulation Bootstrap               ║
║          Trigger V.1 Hackathon  |  Hybrid YOLO-LSTM-RL Traffic System        ║
║          Senior Research Engineer: Traffic Management & Urban Mobility        ║
╚══════════════════════════════════════════════════════════════════════════════╝

Architecture:
  ┌────────────────────────────────────────────────────────────────────────┐
  │  generate_net_xml()   → Writes 4-way, 3-lane urban intersection        │
  │  generate_rou_xml()   → Randomised high-density demand (car/bus/EV)    │
  │  generate_sumocfg()   → 0.1 s step-length for AI-grade granularity     │
  │  EcoSyncSim           → TraCI bridge:  get_state() / step() / reset()  │
  └────────────────────────────────────────────────────────────────────────┘

Phase 4 hook:
  The RL agent receives a flat numpy array from get_state() and calls
  apply_phase(phase_id) to mutate the traffic-light programme.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Standard library
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys
import time
import random
import subprocess
import logging
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

# ─────────────────────────────────────────────────────────────────────────────
# Third-party  (traci is bundled with SUMO; numpy for RL-ready state vectors)
# ─────────────────────────────────────────────────────────────────────────────
try:
    import numpy as np
except ImportError:
    raise SystemExit("❌  numpy not found — run:  pip install numpy")

try:
    import traci
    import traci.constants as tc
except ImportError:
    raise SystemExit(
        "❌  traci not found.\n"
        "    Set SUMO_HOME and add $SUMO_HOME/tools to PYTHONPATH, or:\n"
        "    pip install eclipse-sumo  (Linux / macOS via Conda preferred)"
    )

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)-8s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("EcoSync")


# ══════════════════════════════════════════════════════════════════════════════
# 1.  INTERSECTION GEOMETRY   →   .net.xml
# ══════════════════════════════════════════════════════════════════════════════

def generate_net_xml(out_dir: Path) -> Path:
    """
    Construct a high-density 4-way urban intersection using netconvert.
    """
    net_path = out_dir / "ecosync.net.xml"
    nod_path = out_dir / "ecosync.nod.xml"
    edg_path = out_dir / "ecosync.edg.xml"
    con_path = out_dir / "ecosync.con.xml"

    # 1. Write Nodes
    nod_xml = textwrap.dedent("""\
    <nodes>
      <node id="N_end" x="0"    y="200"  type="dead_end"/>
      <node id="S_end" x="0"    y="-200" type="dead_end"/>
      <node id="E_end" x="200"  y="0"    type="dead_end"/>
      <node id="W_end" x="-200" y="0"    type="dead_end"/>
      <node id="J0" x="0" y="0" type="traffic_light"/>
    </nodes>
    """)
    nod_path.write_text(nod_xml, encoding='utf-8')

    # 2. Write Edges
    edg_xml = textwrap.dedent("""\
    <edges>
      <edge id="N_in"  from="N_end" to="J0"   numLanes="3" speed="13.89" priority="10"/>
      <edge id="N_out" from="J0"   to="N_end" numLanes="3" speed="13.89" priority="10"/>
      <edge id="S_in"  from="S_end" to="J0"   numLanes="3" speed="13.89" priority="10"/>
      <edge id="S_out" from="J0"   to="S_end" numLanes="3" speed="13.89" priority="10"/>
      <edge id="E_in"  from="E_end" to="J0"   numLanes="3" speed="13.89" priority="10"/>
      <edge id="E_out" from="J0"   to="E_end" numLanes="3" speed="13.89" priority="10"/>
      <edge id="W_in"  from="W_end" to="J0"   numLanes="3" speed="13.89" priority="10"/>
      <edge id="W_out" from="J0"   to="W_end" numLanes="3" speed="13.89" priority="10"/>
    </edges>
    """)
    edg_path.write_text(edg_xml, encoding='utf-8')

    # 3. Write Connections
    con_xml = textwrap.dedent("""\
    <connections>
      <connection from="N_in" to="E_out" fromLane="0" toLane="0" dir="l"/>
      <connection from="N_in" to="S_out" fromLane="1" toLane="1" dir="s"/>
      <connection from="N_in" to="S_out" fromLane="2" toLane="2" dir="s"/>
      <connection from="N_in" to="W_out" fromLane="2" toLane="2" dir="r"/>

      <connection from="S_in" to="W_out" fromLane="0" toLane="0" dir="l"/>
      <connection from="S_in" to="N_out" fromLane="1" toLane="1" dir="s"/>
      <connection from="S_in" to="N_out" fromLane="2" toLane="2" dir="s"/>
      <connection from="S_in" to="E_out" fromLane="2" toLane="2" dir="r"/>

      <connection from="E_in" to="S_out" fromLane="0" toLane="0" dir="l"/>
      <connection from="E_in" to="W_out" fromLane="1" toLane="1" dir="s"/>
      <connection from="E_in" to="W_out" fromLane="2" toLane="2" dir="s"/>
      <connection from="E_in" to="N_out" fromLane="2" toLane="2" dir="r"/>

      <connection from="W_in" to="N_out" fromLane="0" toLane="0" dir="l"/>
      <connection from="W_in" to="E_out" fromLane="1" toLane="1" dir="s"/>
      <connection from="W_in" to="E_out" fromLane="2" toLane="2" dir="s"/>
      <connection from="W_in" to="S_out" fromLane="2" toLane="2" dir="r"/>
    </connections>
    """)
    con_path.write_text(con_xml, encoding='utf-8')

    # 4. Call netconvert
    cmd = [
        "netconvert",
        "--node-files", str(nod_path),
        "--edge-files", str(edg_path),
        "--connection-files", str(con_path),
        "--output-file", str(net_path),
        "--no-warnings", "true",
        "--tls.join", "true",
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        log.info("✅  net.xml generated via netconvert \u2192 %s", net_path)
    except subprocess.CalledProcessError as e:
        log.error("❌  netconvert failed: %s", e.stderr.decode())
        raise

    return net_path


# ══════════════════════════════════════════════════════════════════════════════
# 2.  TRAFFIC DEMAND   →   .rou.xml
# ══════════════════════════════════════════════════════════════════════════════

# All 12 legal O-D pairs for the 4-arm intersection
OD_ROUTES: List[Tuple[str, str, str]] = [
    ("rNS", "N_in S_out", "N→S straight"),
    ("rNE", "N_in E_out", "N→E left"),
    ("rNW", "N_in W_out", "N→W right"),
    ("rSN", "S_in N_out", "S→N straight"),
    ("rSW", "S_in W_out", "S→W left"),
    ("rSE", "S_in E_out", "S→E right"),
    ("rEW", "E_in W_out", "E→W straight"),
    ("rEN", "E_in N_out", "E→N left"),
    ("rES", "E_in S_out", "E→S right"),
    ("rWE", "W_in E_out", "W→E straight"),
    ("rWS", "W_in S_out", "W→S left"),
    ("rWN", "W_in N_out", "W→N right"),
]


def _vehicle_block(
    veh_id: str,
    vtype: str,
    route_id: str,
    depart: float,
    depart_lane: str = "best",
    depart_speed: str = "max",
    color: str = "1,1,0",
) -> str:
    return (
        f'  <vehicle id="{veh_id}" type="{vtype}" route="{route_id}" '
        f'depart="{depart:.1f}" departLane="{depart_lane}" '
        f'departSpeed="{depart_speed}" color="{color}"/>\n'
    )


def generate_rou_xml(
    out_dir: Path,
    sim_duration: float = 3600.0,
    seed: int = 42,
) -> Path:
    """
    Generate randomised, high-density Smart City traffic demand.

    Vehicle mix:
        • passenger cars  — 70 %   (HBEFA4 Euro-6 emission class)
        • buses           — 15 %   (heavy diesel)
        • emergency EVs   — 10 %   (electric, priority override)
        • motorcycles     —  5 %   (low emission)

    Peak-hour shaping: demand ramps up 0→20 min, flat 20→40 min,
                       ramps down 40→60 min (repeats each hour).
    """
    rng = random.Random(seed)
    path = out_dir / "ecosync.rou.xml"

    # ── Vehicle type definitions ─────────────────────────────────────────────
    vtypes = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <!--
        EcoSync  |  High-Density Smart City Demand
        Auto-generated by ecosync_sim.py
    -->
    <routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
            xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">

      <!-- ══ Vehicle Types ══════════════════════════════════════════════════ -->

      <!-- Passenger Car — Euro-6 HBEFA4 petrol -->
      <vType id="car" accel="2.6" decel="4.5" sigma="0.5"
             length="4.5" minGap="2.5" maxSpeed="13.89"
             guiShape="passenger" color="0.7,0.7,1.0"
             personCapacity="5"/>

      <!-- City Bus — heavy diesel Euro-VI -->
      <vType id="bus" accel="1.2" decel="3.0" sigma="0.3"
             length="12.0" minGap="3.0" maxSpeed="11.11"
             guiShape="bus" color="0.2,0.6,0.2"
             personCapacity="60"/>

      <!-- Emergency Vehicle — BEV (zero tailpipe emissions) -->
      <vType id="emergency" accel="3.5" decel="6.0" sigma="0.1"
             length="5.5" minGap="2.0" maxSpeed="22.22"
             guiShape="emergency" color="1.0,0.0,0.0"
             speedFactor="1.5" speedDev="0.1"
             lcStrategic="1.0" lcCooperative="0.0"
             jmIgnoreKeepClearTime="-1"/>

      <!-- Motorcycle — petrol Euro-4 low-cc -->
      <vType id="moto" accel="3.2" decel="5.0" sigma="0.6"
             length="2.2" minGap="1.5" maxSpeed="16.67"
             guiShape="moped" color="1.0,0.8,0.0"
             personCapacity="2"/>

    """)

    # ── Route definitions ────────────────────────────────────────────────────
    route_defs = "  <!-- ══ Routes ══════════════════════════════════════════════════════ -->\n"
    for rid, edges, comment in OD_ROUTES:
        route_defs += f'  <route id="{rid}" edges="{edges}"/>  <!-- {comment} -->\n'
    route_defs += "\n"

    # ── Demand generation ────────────────────────────────────────────────────
    vehicles: List[str] = []
    veh_count = 0

    # Base departures-per-second (will be scaled by peak factor)
    BASE_RATE = 0.55          # vehicles / second across all approaches

    VTYPE_WEIGHTS = [
        ("car",       0.70),
        ("bus",       0.15),
        ("emergency", 0.10),
        ("moto",      0.05),
    ]
    cumulative = [sum(w for _, w in VTYPE_WEIGHTS[:i+1]) for i in range(len(VTYPE_WEIGHTS))]

    def pick_vtype() -> str:
        r = rng.random()
        for (vt, _), cum in zip(VTYPE_WEIGHTS, cumulative):
            if r < cum:
                return vt
        return "car"

    def peak_factor(t: float, period: float = 3600.0) -> float:
        """Sinusoidal peak shaping — peak at ~20 min, trough at ~50 min."""
        phase = (t % period) / period          # 0..1
        return 0.4 + 0.6 * (0.5 + 0.5 * np.sin(2 * np.pi * phase - np.pi / 2))

    t = 0.0
    while t < sim_duration:
        rate = BASE_RATE * peak_factor(t)      # vehicles / second
        gap  = rng.expovariate(rate)           # Poisson inter-arrival
        t   += gap
        if t >= sim_duration:
            break

        veh_id  = f"veh_{veh_count:05d}"
        vtype   = pick_vtype()
        route   = rng.choice(OD_ROUTES)[0]

        color_map = {
            "car":       "0.7,0.7,1.0",
            "bus":       "0.2,0.6,0.2",
            "emergency": "1.0,0.0,0.0",
            "moto":      "1.0,0.8,0.0",
        }
        vehicles.append(_vehicle_block(
            veh_id, vtype, route, t,
            depart_lane="best" if vtype != "emergency" else "0",
            depart_speed="max" if vtype == "emergency" else "random",
            color=color_map[vtype],
        ))
        veh_count += 1

    log.info("📦  Generated %d vehicles over %.0f s (seed=%d)", veh_count, sim_duration, seed)

    # ── Assemble and write ────────────────────────────────────────────────────
    body = vtypes + route_defs
    body += "  <!-- ══ Vehicles ══════════════════════════════════════════════════════ -->\n"
    body += "".join(vehicles)
    body += "\n</routes>\n"

    path.write_text(body, encoding='utf-8')
    log.info("✅  rou.xml written → %s  (%d vehicles)", path, veh_count)
    return path


# ══════════════════════════════════════════════════════════════════════════════
# 3.  SIMULATION CONFIGURATION   →   .sumocfg
# ══════════════════════════════════════════════════════════════════════════════

def generate_sumocfg(
    out_dir: Path,
    sim_duration: float = 3600.0,
    step_length: float = 0.1,
) -> Path:
    """
    0.1 s step-length enables AI-grade temporal resolution.
    Emissions output is activated for both XML log and per-step TraCI queries.
    """
    path = out_dir / "ecosync.sumocfg"

    xml = textwrap.dedent(f"""\
    <?xml version="1.0" encoding="UTF-8"?>
    <!--
        EcoSync  |  SUMO Configuration
        Step length : {step_length} s  (AI-grade granularity)
        Auto-generated by ecosync_sim.py
    -->
    <configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                   xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">

      <input>
        <net-file       value="ecosync.net.xml"/>
        <route-files    value="ecosync.rou.xml"/>
        <additional-files value=""/>          <!-- hook for detectors, polygons, etc. -->
      </input>

      <time>
        <begin       value="0"/>
        <end         value="{sim_duration:.0f}"/>
        <step-length value="{step_length}"/>
      </time>

      <processing>
        <!-- Collision avoidance -->
        <collision.action        value="warn"/>
        <collision.mingap-factor value="0"/>

        <!-- Emergency vehicle override -->
        <device.bluelight.reactiondist value="25.0"/>

        <!-- Lateral resolution for 3-lane fidelity -->
        <lateral-resolution value="0.8"/>

        <!-- Time-to-teleport: long to avoid artificial throughput -->
        <time-to-teleport value="300"/>

        <!-- Routing threads -->
        <threads value="4"/>
      </processing>

      <routing>
        <routing-algorithm value="dijkstra"/>
        <device.rerouting.period value="60"/>
      </routing>

      <!-- ── Emission / Air Quality outputs ─────────────────────────────── -->
      <output>
        <emission-output          value="outputs/ecosync_emissions.xml"/>
        <summary-output           value="outputs/ecosync_summary.xml"/>
        <queue-output             value="outputs/ecosync_queues.xml"/>
        <lanechange-output        value="outputs/ecosync_lanechanges.xml"/>
        <tripinfo-output          value="outputs/ecosync_tripinfo.xml"/>
        <emission-output.precision value="6"/>
      </output>

      <!-- ── TraCI / Remote-control ─────────────────────────────────────── -->
      <traci_server>
        <remote-port value="0"/>         <!-- 0 = OS assigns port dynamically -->
      </traci_server>

      <!-- ── GUI (used only when gui=True) ──────────────────────────────── -->
      <gui_only>
        <start      value="true"/>
        <quit-on-end value="true"/>
        <breakpoints.file value=""/>
      </gui_only>

      <!-- ── Logging ────────────────────────────────────────────────────── -->
      <report>
        <verbose value="true"/>
        <print-options value="true"/>
        <log value="outputs/ecosync_sumo.log"/>
        <message-log value="outputs/ecosync_messages.log"/>
        <error-log   value="outputs/ecosync_errors.log"/>
        <duration-log.statistics value="true"/>
        <duration-log.disable    value="false"/>
      </report>

    </configuration>
    """)

    path.write_text(xml, encoding='utf-8')
    log.info("✅  sumocfg written → %s", path)
    return path


# ══════════════════════════════════════════════════════════════════════════════
# 4.  ECOSYNC SIM — TraCI BRIDGE
# ══════════════════════════════════════════════════════════════════════════════

# ── Emission subscription constants ─────────────────────────────────────────
# tc.VAR_CO2EMISSION, tc.VAR_NOXEMISSION are mg/s per vehicle
# Accumulated to mg/step then normalised per lane per second.

SUBSCRIBED_VARS: List[int] = [
    tc.VAR_SPEED,           # m/s
    tc.VAR_WAITING_TIME,    # s
    tc.VAR_CO2EMISSION,     # mg/s
    tc.VAR_NOXEMISSION,     # mg/s
    tc.VAR_PMXEMISSION,     # mg/s  (bonus: particulate matter)
    tc.VAR_LANE_ID,         # lane membership string
]

# Ordered list of all 12 incoming lanes
INCOMING_LANES: List[str] = [
    "N_in_0", "N_in_1", "N_in_2",
    "S_in_0", "S_in_1", "S_in_2",
    "E_in_0", "E_in_1", "E_in_2",
    "W_in_0", "W_in_1", "W_in_2",
]

# Arms for naming / grouping
ARMS: List[str] = ["N", "S", "E", "W"]


@dataclass
class LaneState:
    """
    Snapshot of a single lane at time t.
    Designed to be serialised to a numpy feature vector for the RL agent.
    """
    lane_id:          str
    vehicle_count:    int   = 0
    density:          float = 0.0   # vehicles / km
    avg_wait:         float = 0.0   # seconds
    avg_speed:        float = 0.0   # m/s
    co2_per_second:   float = 0.0   # mg / s
    nox_per_second:   float = 0.0   # mg / s
    pmx_per_second:   float = 0.0   # mg / s
    queue_length:     float = 0.0   # metres
    occupancy:        float = 0.0   # 0–1


@dataclass
class IntersectionState:
    """
    Aggregated state for the entire intersection — fed directly to the RL agent.
    """
    sim_time:          float
    tls_phase:         int
    tls_phase_duration: float                            # s since last change
    lane_states:       Dict[str, LaneState] = field(default_factory=dict)
    total_co2:         float = 0.0                       # mg/s, all lanes
    total_nox:         float = 0.0                       # mg/s, all lanes
    total_pmx:         float = 0.0                       # mg/s, all lanes
    avg_wait_global:   float = 0.0                       # seconds
    throughput_step:   int   = 0                         # vehicles departed this step

    def to_numpy(self) -> np.ndarray:
        """
        Flat feature vector for the RL agent.
        Shape: (12 lanes × 8 features) + 5 global = 101 floats.
        """
        lane_features = []
        for lid in INCOMING_LANES:
            ls = self.lane_states.get(lid, LaneState(lid))
            lane_features.extend([
                ls.vehicle_count,
                ls.density,
                ls.avg_wait,
                ls.avg_speed,
                ls.co2_per_second,
                ls.nox_per_second,
                ls.pmx_per_second,
                ls.queue_length,
            ])
        global_features = [
            self.tls_phase,
            self.tls_phase_duration,
            self.total_co2,
            self.total_nox,
            self.avg_wait_global,
        ]
        return np.array(lane_features + global_features, dtype=np.float32)


class EcoSyncSim:
    """
    TraCI bridge for the EcoSync SUMO environment.

    Usage (headless RL training):
    ─────────────────────────────
        sim = EcoSyncSim(gui=False)
        state = sim.reset()                      # numpy array (101,)
        while not sim.done:
            action   = agent.act(state)
            state, reward, done, info = sim.step(action)
        sim.close()

    Action space (Phase 4 RL):
        Discrete — index into TLS programme phases.
        sim.apply_phase(phase_id: int)

    Observation space:
        Box(0, ∞, shape=(101,), dtype=float32)
    """

    TLS_ID   = "J0_tls"
    LANE_LEN = 200.0      # metres — arm length from net.xml

    # Total TLS phases in the programme (8 = 4 green + 4 yellow)
    N_PHASES = 8

    def __init__(
        self,
        cfg_path:    Path,
        gui:         bool  = False,
        step_length: float = 0.1,
        max_steps:   int   = 36_000,     # 3600 s / 0.1 s
        port:        int   = 8813,
        seed:        int   = 42,
    ) -> None:
        self.cfg_path    = Path(cfg_path)
        self.gui         = gui
        self.step_length = step_length
        self.max_steps   = max_steps
        self.port        = port
        self.seed        = seed

        self._step_count: int = 0
        self._sumo_proc:  Optional[subprocess.Popen] = None
        self._connected:  bool = False
        self._last_tls_change: float = 0.0
        self._prev_teleported: int = 0
        self.done: bool = False

        # ── SUMO binary detection ─────────────────────────────────────────
        sumo_home = os.environ.get("SUMO_HOME")
        if sumo_home is None:
            raise EnvironmentError(
                "SUMO_HOME is not set.\n"
                "  Linux:   export SUMO_HOME=/usr/share/sumo\n"
                "  Windows: set SUMO_HOME=C:\\Program Files (x86)\\Eclipse\\Sumo"
            )
        binary = "sumo-gui" if gui else "sumo"
        self._sumo_bin = Path(sumo_home) / "bin" / binary
        if not self._sumo_bin.exists():
            # Fallback: search PATH
            self._sumo_bin = binary
            log.warning("SUMO binary not found under SUMO_HOME — relying on PATH")

    # ── Private helpers ───────────────────────────────────────────────────────

    def _launch_sumo(self) -> None:
        """Spawn SUMO as a subprocess and open a TraCI connection."""
        cmd = [
            str(self._sumo_bin),
            "-c", str(self.cfg_path),
            "--remote-port", str(self.port),
            "--seed", str(self.seed),
            "--no-step-log",                  # suppress per-step console spam
            "--collision.check-junctions",
            "--step-length", str(self.step_length),
        ]
        if not self.gui:
            cmd += ["--no-warnings"]

        log.info("🚦  Launching SUMO: %s", " ".join(cmd))
        self._sumo_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # Allow SUMO to initialise before connecting
        time.sleep(1.5)
        traci.init(self.port)
        self._connected = True
        log.info("🔗  TraCI connected on port %d", self.port)

    def _subscribe_all_vehicles(self) -> None:
        """Subscribe every vehicle currently loaded for multi-var polling."""
        for vid in traci.vehicle.getIDList():
            traci.vehicle.subscribe(vid, SUBSCRIBED_VARS)

    def _terminate(self) -> None:
        if self._connected:
            try:
                traci.close()
            except Exception:
                pass
            self._connected = False
        if self._sumo_proc and self._sumo_proc.poll() is None:
            self._sumo_proc.terminate()
            try:
                self._sumo_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._sumo_proc.kill()
            self._sumo_proc = None

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        """
        (Re)start the simulation from t=0.
        Returns the initial observation as a numpy array.
        """
        self._terminate()
        self._step_count     = 0
        self._last_tls_change = 0.0
        self._prev_teleported = 0
        self.done            = False

        self._launch_sumo()

        # Warm-up: advance 10 steps to allow vehicles to load
        for _ in range(10):
            traci.simulationStep()
        self._subscribe_all_vehicles()

        state = self.get_state()
        return state.to_numpy()

    def get_state(self) -> IntersectionState:
        """
        Query TraCI for the full intersection state.

        Emission metrics (Urban Air Quality focus):
            CO2  — primary greenhouse gas indicator
            NOx  — key urban smog precursor
            PMx  — fine particulate matter (health impact)

        Returns an IntersectionState dataclass with both per-lane granularity
        and intersection-level aggregates.
        """
        sim_time = traci.simulation.getTime()

        # ── TLS state ─────────────────────────────────────────────────────
        tls_phase    = traci.trafficlight.getPhase(self.TLS_ID)
        tls_phase_t  = sim_time - self._last_tls_change

        # ── Subscribe newly-arrived vehicles ──────────────────────────────
        for vid in traci.simulation.getDepartedIDList():
            traci.vehicle.subscribe(vid, SUBSCRIBED_VARS)

        # ── Build per-lane accumulators ───────────────────────────────────
        lane_accum: Dict[str, Dict] = {
            lid: {
                "count":  0,
                "wait":   0.0,
                "speed":  0.0,
                "co2":    0.0,
                "nox":    0.0,
                "pmx":    0.0,
            }
            for lid in INCOMING_LANES
        }

        # Pull subscribed results for every active vehicle
        sub_results = traci.vehicle.getAllSubscriptionResults()

        for vid, vals in sub_results.items():
            lane_id = vals.get(tc.VAR_LANE_ID, "")
            if lane_id not in lane_accum:
                continue

            acc = lane_accum[lane_id]
            acc["count"] += 1
            acc["wait"]  += vals.get(tc.VAR_WAITING_TIME, 0.0)
            acc["speed"] += vals.get(tc.VAR_SPEED,        0.0)
            acc["co2"]   += vals.get(tc.VAR_CO2EMISSION,  0.0)
            acc["nox"]   += vals.get(tc.VAR_NOXEMISSION,  0.0)
            acc["pmx"]   += vals.get(tc.VAR_PMXEMISSION,  0.0)

        # ── Build LaneState objects ───────────────────────────────────────
        lane_states: Dict[str, LaneState] = {}
        total_co2 = total_nox = total_pmx = 0.0
        total_wait = 0.0
        total_count = 0

        for lid in INCOMING_LANES:
            acc   = lane_accum[lid]
            n     = acc["count"]

            # TraCI direct lane queries for queue and occupancy
            try:
                queue_m   = traci.lane.getLastStepHaltingNumber(lid) * 7.5  # ~7.5 m/veh
                occupancy = traci.lane.getLastStepOccupancy(lid)             # 0–1
            except traci.exceptions.TraCIException:
                queue_m   = 0.0
                occupancy = 0.0

            ls = LaneState(
                lane_id        = lid,
                vehicle_count  = n,
                density        = (n / (self.LANE_LEN / 1000)) if n > 0 else 0.0,
                avg_wait       = acc["wait"]  / n if n > 0 else 0.0,
                avg_speed      = acc["speed"] / n if n > 0 else 0.0,
                co2_per_second = acc["co2"],          # already mg/s (sum over lane)
                nox_per_second = acc["nox"],
                pmx_per_second = acc["pmx"],
                queue_length   = queue_m,
                occupancy      = occupancy,
            )
            lane_states[lid] = ls

            total_co2  += acc["co2"]
            total_nox  += acc["nox"]
            total_pmx  += acc["pmx"]
            total_wait += acc["wait"]
            total_count += n

        # ── Throughput: vehicles that left the intersection this step ──────
        teleported_now = traci.simulation.getStartingTeleportNumber()
        throughput     = len(traci.simulation.getArrivedIDList())

        state = IntersectionState(
            sim_time           = sim_time,
            tls_phase          = tls_phase,
            tls_phase_duration = tls_phase_t,
            lane_states        = lane_states,
            total_co2          = total_co2,
            total_nox          = total_nox,
            total_pmx          = total_pmx,
            avg_wait_global    = (total_wait / total_count) if total_count > 0 else 0.0,
            throughput_step    = throughput,
        )
        return state

    def apply_phase(self, phase_id: int) -> None:
        """
        Phase 4 RL hook — set the TLS to a specific phase index.
        The RL agent calls this before each step().

        Args:
            phase_id: 0 .. N_PHASES-1
        """
        if not (0 <= phase_id < self.N_PHASES):
            log.warning("apply_phase: invalid phase %d (max %d)", phase_id, self.N_PHASES - 1)
            return
        current = traci.trafficlight.getPhase(self.TLS_ID)
        if current != phase_id:
            traci.trafficlight.setPhase(self.TLS_ID, phase_id)
            self._last_tls_change = traci.simulation.getTime()

    def step(
        self,
        action: Optional[int] = None,
    ) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Advance the simulation by one step (step_length seconds).

        Args:
            action: TLS phase index for RL agent.  None = baseline (timed).

        Returns:
            obs     — numpy array (101,)
            reward  — scalar (negative weighted emission + wait penalty)
            done    — True when simulation ends
            info    — diagnostic dict for logging / debugging
        """
        if not self._connected:
            raise RuntimeError("Call reset() before step().")

        # ── Apply RL action ───────────────────────────────────────────────
        if action is not None:
            self.apply_phase(action)

        # ── Advance simulation ────────────────────────────────────────────
        try:
            traci.simulationStep()
        except traci.exceptions.FatalTraCIError as exc:
            log.error("TraCI fatal error at step %d: %s", self._step_count, exc)
            self.done = True
            dummy_obs = np.zeros(101, dtype=np.float32)
            return dummy_obs, 0.0, True, {"error": str(exc)}

        self._step_count += 1

        # ── Termination check ─────────────────────────────────────────────
        remaining = traci.simulation.getMinExpectedNumber()
        if remaining == 0 or self._step_count >= self.max_steps:
            self.done = True

        # ── Get state ─────────────────────────────────────────────────────
        state = self.get_state()
        obs   = state.to_numpy()

        # ── Reward (Phase 4 RL signal — Urban Air Quality focus) ──────────
        # R = − ( α·CO2 + β·NOx + γ·PMx + δ·avg_wait )
        # Weights tuned to normalise approximately equal magnitudes.
        ALPHA, BETA, GAMMA, DELTA = 0.001, 0.01, 0.05, 0.5
        reward = -(
            ALPHA * state.total_co2
            + BETA  * state.total_nox
            + GAMMA * state.total_pmx
            + DELTA * state.avg_wait_global
        )

        info = {
            "sim_time":       state.sim_time,
            "step":           self._step_count,
            "tls_phase":      state.tls_phase,
            "total_co2_mg_s": state.total_co2,
            "total_nox_mg_s": state.total_nox,
            "total_pmx_mg_s": state.total_pmx,
            "avg_wait_s":     state.avg_wait_global,
            "throughput":     state.throughput_step,
            "vehicles_active": sum(ls.vehicle_count for ls in state.lane_states.values()),
        }
        return obs, reward, self.done, info

    def close(self) -> None:
        """Gracefully tear down TraCI and SUMO subprocess."""
        log.info("🔌  Closing EcoSyncSim (step=%d)", self._step_count)
        self._terminate()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ══════════════════════════════════════════════════════════════════════════════
# 5.  ENVIRONMENT FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def build_environment(
    workspace:    str  = "ecosync_env",
    gui:          bool = False,
    sim_duration: float = 3600.0,
    seed:         int  = 42,
) -> EcoSyncSim:
    """
    One-call factory: generates all XML assets and returns a ready-to-run
    EcoSyncSim instance.

    Args:
        workspace:    directory where all SUMO files are written
        gui:          True to open the SUMO-GUI window
        sim_duration: seconds to simulate
        seed:         random seed for reproducible demand generation

    Returns:
        EcoSyncSim — call .reset() to start
    """
    outputs = Path(workspace)
    outputs.mkdir(parents=True, exist_ok=True)
    (outputs / "outputs").mkdir(exist_ok=True)

    log.info("═" * 60)
    log.info("  EcoSync Environment Builder  —  Phase 1")
    log.info("═" * 60)
    log.info("  Workspace : %s", outputs.resolve())

    generate_net_xml(outputs)
    generate_rou_xml(outputs, sim_duration=sim_duration, seed=seed)
    generate_sumocfg(outputs, sim_duration=sim_duration)

    cfg_path = outputs / "ecosync.sumocfg"
    sim = EcoSyncSim(
        cfg_path    = cfg_path,
        gui         = gui,
        step_length = 0.1,
        max_steps   = int(sim_duration / 0.1),
        seed        = seed,
    )

    log.info("═" * 60)
    log.info("  ✅  Environment ready — call sim.reset() to begin")
    log.info("═" * 60)
    return sim


# ══════════════════════════════════════════════════════════════════════════════
# 6.  SMOKE-TEST  (run with:  python ecosync_sim.py)
# ══════════════════════════════════════════════════════════════════════════════

def _smoke_test() -> None:
    """
    Quick sanity check — runs 100 steps headless, prints state summary.
    No RL agent; action=None means baseline timed TLS control.
    """
    log.info("🧪  Running smoke test (100 steps, headless) …")

    sim = build_environment(gui=False, sim_duration=600.0, seed=7)
    obs = sim.reset()

    log.info("  Initial obs shape : %s", obs.shape)
    log.info("  Initial obs[:5]   : %s", obs[:5])

    rewards = []
    for i in range(100):
        obs, reward, done, info = sim.step(action=None)
        rewards.append(reward)
        if i % 20 == 0:
            log.info(
                "  Step %3d | t=%.1f s | TLS=%d | CO2=%.1f mg/s | "
                "NOx=%.2f mg/s | wait=%.2f s | reward=%.4f",
                i,
                info["sim_time"],
                info["tls_phase"],
                info["total_co2_mg_s"],
                info["total_nox_mg_s"],
                info["avg_wait_s"],
                reward,
            )
        if done:
            log.info("  Simulation ended early at step %d", i)
            break

    log.info("  Mean reward (100 steps): %.5f", float(np.mean(rewards)))
    sim.close()
    log.info("✅  Smoke test passed.")


if __name__ == "__main__":
    # ── CLI flags ─────────────────────────────────────────────────────────
    import argparse

    parser = argparse.ArgumentParser(
        description="EcoSync — Phase 1 SUMO Environment Builder & TraCI Bridge",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace",    default="ecosync_env",  help="Output directory")
    parser.add_argument("--gui",          action="store_true",     help="Open SUMO-GUI")
    parser.add_argument("--duration",     type=float, default=3600.0, help="Simulation duration (s)")
    parser.add_argument("--seed",         type=int,   default=42,     help="Random seed")
    parser.add_argument("--smoke-test",   action="store_true",     help="Run 100-step smoke test")
    parser.add_argument("--generate-only",action="store_true",     help="Only write XML files, no TraCI")
    args = parser.parse_args()

    if args.smoke_test:
        _smoke_test()
    elif args.generate_only:
        d = Path(args.workspace)
        d.mkdir(parents=True, exist_ok=True)
        (d / "outputs").mkdir(exist_ok=True)
        generate_net_xml(d)
        generate_rou_xml(d, sim_duration=args.duration, seed=args.seed)
        generate_sumocfg(d, sim_duration=args.duration)
        log.info("🗂   XML files written to %s — launch SUMO manually.", d.resolve())
    else:
        sim = build_environment(
            workspace    = args.workspace,
            gui          = args.gui,
            sim_duration = args.duration,
            seed         = args.seed,
        )
        log.info("📋  EcoSyncSim object created.  Call sim.reset() to start.")
        log.info("    Example integration:")
        log.info("      obs = sim.reset()")
        log.info("      obs, reward, done, info = sim.step(action=0)")
        log.info("      sim.close()")