"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        EcoSync — Phase 2: YOLOv8 Perception Engine                          ║
║        Trigger V.1 Hackathon  |  Hybrid YOLO-LSTM-RL Traffic System         ║
║        Senior Computer Vision Engineer                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

Architecture:
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  ScreenCapture      → mss-based zero-copy SUMO-GUI frame grabber        │
  │  ROIManager         → Polygonal lane masks with Shapely / cv2            │
  │  EmissionsScorer    → Per-class CO2 / NOx proxy weights                  │
  │  YOLOPerceptionEngine → Full pipeline: capture → detect → classify       │
  │  PerceptionBridge   → Thread-safe adapter for LSTM (Phase 3) / RL        │
  └─────────────────────────────────────────────────────────────────────────┘

Phase 3 / 4 hooks:
  • PerceptionBridge.get_latest_frame_data() → list of FrameData (ring buffer)
  • YOLOPerceptionEngine.process_frame()     → single-frame dict (sync call)
  • All outputs are typed dataclasses — drop-in for LSTM sequence builder
"""

# ─────────────────────────────────────────────────────────────────────────────
# Standard library
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys
import time
import logging
import threading
import warnings
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Third-party — hard requirements
# ─────────────────────────────────────────────────────────────────────────────
try:
    import numpy as np
except ImportError:
    raise SystemExit("❌  numpy not found — pip install numpy")

try:
    import cv2
except ImportError:
    raise SystemExit("❌  opencv-python not found — pip install opencv-python")

try:
    import mss
    import mss.tools
except ImportError:
    raise SystemExit("❌  mss not found — pip install mss")

try:
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER as _yolo_logger
    _yolo_logger.setLevel(logging.WARNING)          # suppress YOLO chatter
except ImportError:
    raise SystemExit("❌  ultralytics not found — pip install ultralytics")

# Shapely is optional — we fall back to cv2 point-in-contour if absent
try:
    from shapely.geometry import Point, Polygon as ShapelyPolygon
    _SHAPELY_AVAILABLE = True
except ImportError:
    _SHAPELY_AVAILABLE = False
    warnings.warn("shapely not installed — falling back to cv2 for ROI tests. "
                  "pip install shapely  for ~2× faster polygon hit-tests.")

# TraCI — soft dependency (engine works without it for offline video testing)
try:
    import traci
    _TRACI_AVAILABLE = True
except ImportError:
    _TRACI_AVAILABLE = False
    warnings.warn("traci not available — TraCI fusion disabled.")

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)-8s]  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("EcoSync.Perception")


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS & EMISSION PROXY WEIGHTS
# ══════════════════════════════════════════════════════════════════════════════

# COCO class names that map to road vehicles
VEHICLE_COCO_CLASSES: Dict[int, str] = {
    2:  "car",
    5:  "bus",
    7:  "truck",
    3:  "motorcycle",
}

# ── Emissions proxy weights (Urban Air Quality focus) ─────────────────────────
# Derived from HBEFA4 average emission factors (g CO2-eq / km, urban arterial):
#   Car        ~  140 g/km  →  weight 1.0   (baseline)
#   Motorcycle ~   80 g/km  →  weight 0.6
#   Truck      ~  650 g/km  →  weight 4.6
#   Bus        ~  900 g/km  →  weight 6.4
#
# NOx proxy (relative mg/km, Euro-average):
#   Car 40 / Motorcycle 60 / Truck 450 / Bus 500
#   → NOx multiplier added as secondary index
#
# Combined score = CO2_weight + NOx_weight  (then normalised 0–10 per lane)
EMISSION_WEIGHTS: Dict[str, Dict[str, float]] = {
    "car":        {"co2": 1.0,  "nox": 0.08, "pmx": 0.05, "combined": 1.13},
    "motorcycle": {"co2": 0.6,  "nox": 0.12, "pmx": 0.03, "combined": 0.75},
    "truck":      {"co2": 4.6,  "nox": 0.90, "pmx": 0.40, "combined": 5.90},
    "bus":        {"co2": 6.4,  "nox": 1.00, "pmx": 0.60, "combined": 8.00},
}

# Default (unknown vehicle class)
_DEFAULT_WEIGHT = EMISSION_WEIGHTS["car"]

# Lane IDs must align with Phase 1 SUMO net (12 incoming lanes)
DEFAULT_LANE_IDS: List[str] = [
    "N_in_0", "N_in_1", "N_in_2",
    "S_in_0", "S_in_1", "S_in_2",
    "E_in_0", "E_in_1", "E_in_2",
    "W_in_0", "W_in_1", "W_in_2",
]

# YOLOv8 inference settings (tuned for real-time on CPU/GPU)
YOLO_MODEL_NAME: str  = "yolov8n.pt"   # Nano — fastest; swap to yolov8s for +mAP
YOLO_CONF_THRESH: float = 0.35          # lower = more detections, more FP
YOLO_IOU_THRESH:  float = 0.45          # NMS overlap threshold
YOLO_IMG_SIZE:    int   = 640           # inference resolution


# ══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BoundingBox:
    """Normalised [0,1] bounding box from YOLO + pixel centre."""
    x1:       float
    y1:       float
    x2:       float
    y2:       float
    cx:       float          # pixel centroid x (absolute)
    cy:       float          # pixel centroid y (absolute)
    conf:     float
    cls_id:   int
    cls_name: str

    @property
    def area_px(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


@dataclass
class LaneDetection:
    """Per-lane detection result for one video frame."""
    lane_id:         str
    vehicle_count:   int               = 0
    class_counts:    Dict[str, int]    = field(default_factory=dict)
    emissions_score: float             = 0.0   # combined CO2+NOx proxy
    co2_score:       float             = 0.0
    nox_score:       float             = 0.0
    pmx_score:       float             = 0.0
    detections:      List[BoundingBox] = field(default_factory=list)


@dataclass
class FrameData:
    """
    Complete perception output for a single captured frame.
    Fed into the LSTM sequence buffer (Phase 3).
    """
    timestamp:        float                        # wall-clock time
    sim_time:         float                        # SUMO simulation time (s)
    frame_id:         int
    fps:              float
    raw_frame:        Optional[np.ndarray]         # BGR image (kept for debug)
    lane_data:        Dict[str, LaneDetection]     # keyed by lane_id
    total_vehicles:   int                = 0
    total_emissions:  float              = 0.0
    inference_ms:     float              = 0.0     # YOLO latency

    def to_live_traffic_data(self) -> Dict[str, Dict]:
        """
        Canonical output contract for Phase 3 LSTM and Phase 4 RL.

        Returns:
            {
              "N_in_0": {"count": 3, "emissions_score": 4.26,
                         "co2": 3.0, "nox": 0.24, "pmx": 0.15,
                         "class_counts": {"car": 2, "bus": 1}},
              ...
            }
        """
        return {
            lid: {
                "count":           ld.vehicle_count,
                "emissions_score": round(ld.emissions_score, 4),
                "co2":             round(ld.co2_score,       4),
                "nox":             round(ld.nox_score,       4),
                "pmx":             round(ld.pmx_score,       4),
                "class_counts":    dict(ld.class_counts),
            }
            for lid, ld in self.lane_data.items()
        }


# ══════════════════════════════════════════════════════════════════════════════
# 1.  SCREEN CAPTURE  —  zero-copy mss grabber
# ══════════════════════════════════════════════════════════════════════════════

class ScreenCapture:
    """
    High-performance screen region capture using mss (direct DMA on X11/Win/Mac).

    The capture region is specified as (left, top, width, height) in screen
    pixels.  On a 1080p display the SUMO-GUI viewport is typically ~1200×800.

    Performance:
        • mss grabs raw BGRA from the GPU framebuffer — no encoding overhead.
        • BGRA → BGR conversion is a single numpy slice (no copy).
        • Typical throughput: 60–120 FPS on a modest machine.
    """

    def __init__(
        self,
        region: Optional[Dict[str, int]] = None,
        monitor_index: int = 1,
    ) -> None:
        """
        Args:
            region:        {"left": x, "top": y, "width": w, "height": h}
                           If None, captures the primary monitor.
            monitor_index: mss monitor number (1 = primary).
        """
        self._sct           = mss.mss()
        self._monitor_index = monitor_index
        self._region        = region or self._sct.monitors[monitor_index]
        self._frame_id      = 0
        self._last_t        = time.perf_counter()
        self._fps           = 0.0
        self._fps_alpha     = 0.1    # EMA smoothing

        log.info("📷  ScreenCapture initialised | region=%s", self._region)

    def set_region(self, region: Dict[str, int]) -> None:
        """Update capture region at runtime (e.g. after window resize)."""
        self._region = region
        log.info("📷  Capture region updated → %s", region)

    def auto_detect_sumo_window(self) -> bool:
        """
        Attempt to find the SUMO-GUI window via wmctrl (Linux) or win32gui (Win).
        Populates self._region if successful.  Returns True on success.
        """
        # ── Linux / X11 ──────────────────────────────────────────────────
        try:
            import subprocess
            out = subprocess.check_output(
                ["wmctrl", "-lG"],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            for line in out.splitlines():
                if "sumo" in line.lower() or "SUMO" in line:
                    parts = line.split()
                    # wmctrl -lG format: wid desktop x y w h machine title
                    x, y, w, h = int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
                    self._region = {"left": x, "top": y, "width": w, "height": h}
                    log.info("🔍  SUMO window detected (Linux): %s", self._region)
                    return True
        except Exception:
            pass

        # ── Windows ──────────────────────────────────────────────────────
        try:
            import win32gui  # type: ignore

            def _cb(hwnd, ctx):
                if "sumo" in win32gui.GetWindowText(hwnd).lower():
                    rect = win32gui.GetWindowRect(hwnd)
                    ctx.append({
                        "left":   rect[0],
                        "top":    rect[1],
                        "width":  rect[2] - rect[0],
                        "height": rect[3] - rect[1],
                    })

            results: List[Dict] = []
            win32gui.EnumWindows(_cb, results)
            if results:
                self._region = results[0]
                log.info("🔍  SUMO window detected (Windows): %s", self._region)
                return True
        except Exception:
            pass

        log.warning("⚠️   Could not auto-detect SUMO window — using full monitor.")
        return False

    def grab(self) -> Optional[np.ndarray]:
        """
        Capture one frame.

        Returns:
            BGR numpy array  (H, W, 3)  or  None if capture fails.
        """
        try:
            raw = self._sct.grab(self._region)
            # BGRA → BGR  (no-copy slice on the last axis)
            frame = np.frombuffer(raw.bgra, dtype=np.uint8).reshape(
                raw.height, raw.width, 4
            )[:, :, :3]

            # FPS tracking (EMA)
            now         = time.perf_counter()
            inst_fps    = 1.0 / max(now - self._last_t, 1e-6)
            self._fps   = self._fps_alpha * inst_fps + (1 - self._fps_alpha) * self._fps
            self._last_t = now
            self._frame_id += 1
            return frame
        except Exception as exc:
            log.error("Screen capture error: %s", exc)
            return None

    @property
    def fps(self) -> float:
        return round(self._fps, 1)

    @property
    def frame_id(self) -> int:
        return self._frame_id

    @property
    def region(self) -> Dict[str, int]:
        return self._region


# ══════════════════════════════════════════════════════════════════════════════
# 2.  ROI MANAGER  —  polygonal lane regions
# ══════════════════════════════════════════════════════════════════════════════

class ROIManager:
    """
    Manages polygonal Regions of Interest (one per lane approach arm).

    Each ROI is a convex (or concave) polygon defined in pixel coordinates
    relative to the captured frame.  Polygons are automatically scaled when
    the frame resolution changes.

    Default geometry assumes a 1280×720 SUMO-GUI viewport centred on a
    4-way intersection — call set_rois() or load_from_file() to customise.

    Hit-testing priority:
        1.  Shapely (vectorised, ~2× faster for large detection counts)
        2.  cv2.pointPolygonTest (fallback)
    """

    def __init__(
        self,
        frame_width:  int = 1280,
        frame_height: int = 720,
    ) -> None:
        self._fw = frame_width
        self._fh = frame_height
        # {lane_id: np.ndarray shape (N,2) dtype=int32}
        self._polygons:   Dict[str, np.ndarray] = {}
        # Shapely cache  {lane_id: ShapelyPolygon}
        self._shapely_cache: Dict[str, Any]    = {}
        # Mask cache  {lane_id: np.ndarray bool (H,W)}
        self._mask_cache: Dict[str, np.ndarray] = {}

        self._build_default_rois()

    # ── Default ROI geometry ─────────────────────────────────────────────────

    def _build_default_rois(self) -> None:
        """
        Hardcoded default polygons for a 1280×720 SUMO top-down view.
        The intersection core sits at roughly (640, 360).

        Each arm has 3 lanes (L / S / SR) stretching from the stop-line
        outward ~200 px.  Adapt with set_rois() for any other resolution.

                 N_in   (top centre, downward traffic)
            ┌────[L][S][SR]────┐
         W_in [L][S][SR] ┼ [L][S][SR] E_in
            └────[L][S][SR]────┘
                 S_in   (bottom centre, upward traffic)
        """
        W, H = self._fw, self._fh
        cx, cy = W // 2, H // 2
        margin = 5    # px gap between polygon edges

        # Lane widths (px) for a typical SUMO-GUI zoom level
        lw = W // 18  # ≈ 71 px per lane at 1280

        # ── North arm — vehicles moving downward ─────────────────────────
        # Lane 0 (left-turn):  leftmost strip
        self._polygons["N_in_0"] = np.array([
            [cx - 3*lw,     cy - 20],
            [cx - 2*lw - margin, cy - 20],
            [cx - 2*lw - margin, cy - H//3],
            [cx - 3*lw,     cy - H//3],
        ], dtype=np.int32)

        # Lane 1 (straight):   centre strip
        self._polygons["N_in_1"] = np.array([
            [cx - 2*lw + margin, cy - 20],
            [cx - lw  - margin, cy - 20],
            [cx - lw  - margin, cy - H//3],
            [cx - 2*lw + margin, cy - H//3],
        ], dtype=np.int32)

        # Lane 2 (straight + right):  rightmost strip
        self._polygons["N_in_2"] = np.array([
            [cx - lw  + margin, cy - 20],
            [cx,                cy - 20],
            [cx,                cy - H//3],
            [cx - lw  + margin, cy - H//3],
        ], dtype=np.int32)

        # ── South arm — vehicles moving upward ───────────────────────────
        self._polygons["S_in_0"] = np.array([
            [cx,                cy + 20],
            [cx + lw  - margin, cy + 20],
            [cx + lw  - margin, cy + H//3],
            [cx,                cy + H//3],
        ], dtype=np.int32)

        self._polygons["S_in_1"] = np.array([
            [cx + lw  + margin, cy + 20],
            [cx + 2*lw - margin, cy + 20],
            [cx + 2*lw - margin, cy + H//3],
            [cx + lw  + margin, cy + H//3],
        ], dtype=np.int32)

        self._polygons["S_in_2"] = np.array([
            [cx + 2*lw + margin, cy + 20],
            [cx + 3*lw,         cy + 20],
            [cx + 3*lw,         cy + H//3],
            [cx + 2*lw + margin, cy + H//3],
        ], dtype=np.int32)

        # ── East arm — vehicles moving leftward ──────────────────────────
        self._polygons["E_in_0"] = np.array([
            [cx + 20,     cy - 3*lw],
            [cx + W//3,   cy - 3*lw],
            [cx + W//3,   cy - 2*lw - margin],
            [cx + 20,     cy - 2*lw - margin],
        ], dtype=np.int32)

        self._polygons["E_in_1"] = np.array([
            [cx + 20,   cy - 2*lw + margin],
            [cx + W//3, cy - 2*lw + margin],
            [cx + W//3, cy - lw   - margin],
            [cx + 20,   cy - lw   - margin],
        ], dtype=np.int32)

        self._polygons["E_in_2"] = np.array([
            [cx + 20,   cy - lw + margin],
            [cx + W//3, cy - lw + margin],
            [cx + W//3, cy],
            [cx + 20,   cy],
        ], dtype=np.int32)

        # ── West arm — vehicles moving rightward ─────────────────────────
        self._polygons["W_in_0"] = np.array([
            [cx - 20,   cy],
            [cx - W//3, cy],
            [cx - W//3, cy + lw   - margin],
            [cx - 20,   cy + lw   - margin],
        ], dtype=np.int32)

        self._polygons["W_in_1"] = np.array([
            [cx - 20,   cy + lw + margin],
            [cx - W//3, cy + lw + margin],
            [cx - W//3, cy + 2*lw - margin],
            [cx - 20,   cy + 2*lw - margin],
        ], dtype=np.int32)

        self._polygons["W_in_2"] = np.array([
            [cx - 20,   cy + 2*lw + margin],
            [cx - W//3, cy + 2*lw + margin],
            [cx - W//3, cy + 3*lw],
            [cx - 20,   cy + 3*lw],
        ], dtype=np.int32)

        self._rebuild_cache()

    # ── Cache management ─────────────────────────────────────────────────────

    def _rebuild_cache(self) -> None:
        self._shapely_cache.clear()
        self._mask_cache.clear()
        if _SHAPELY_AVAILABLE:
            for lid, poly in self._polygons.items():
                self._shapely_cache[lid] = ShapelyPolygon(poly.tolist())

    def _get_mask(self, lane_id: str) -> np.ndarray:
        """Lazily compute and cache a boolean pixel mask for a lane polygon."""
        if lane_id not in self._mask_cache:
            mask = np.zeros((self._fh, self._fw), dtype=np.uint8)
            cv2.fillPoly(mask, [self._polygons[lane_id]], 255)
            self._mask_cache[lane_id] = mask.astype(bool)
        return self._mask_cache[lane_id]

    # ── Public API ────────────────────────────────────────────────────────────

    def set_rois(self, rois: Dict[str, List[Tuple[int, int]]]) -> None:
        """
        Override ROI polygons at runtime.

        Args:
            rois: { "N_in_0": [(x1,y1),(x2,y2),(x3,y3),(x4,y4)], ... }
        """
        self._polygons = {
            lid: np.array(pts, dtype=np.int32)
            for lid, pts in rois.items()
        }
        self._rebuild_cache()
        log.info("🗺   ROIs updated — %d lanes configured.", len(rois))

    def rescale(self, new_w: int, new_h: int) -> None:
        """Proportionally rescale all polygons to a new frame size."""
        sx = new_w / self._fw
        sy = new_h / self._fh
        self._polygons = {
            lid: (poly * np.array([sx, sy])).astype(np.int32)
            for lid, poly in self._polygons.items()
        }
        self._fw, self._fh = new_w, new_h
        self._rebuild_cache()
        log.info("📐  ROIs rescaled to %dx%d", new_w, new_h)

    def point_in_lane(self, cx: float, cy: float, lane_id: str) -> bool:
        """Fast point-in-polygon test for a vehicle centroid."""
        pt_x, pt_y = int(cx), int(cy)
        if _SHAPELY_AVAILABLE:
            return self._shapely_cache[lane_id].contains(Point(pt_x, pt_y))
        poly = self._polygons[lane_id]
        return cv2.pointPolygonTest(poly, (float(pt_x), float(pt_y)), False) >= 0

    def assign_detections_to_lanes(
        self, boxes: List[BoundingBox]
    ) -> Dict[str, List[BoundingBox]]:
        """
        Assign each detected bounding box to zero or one lane ROI.

        Uses vehicle centroid for assignment.  If a centroid falls in
        multiple overlapping ROIs (shouldn't happen with non-overlapping
        lane polygons), the first match wins.

        Returns:
            { lane_id: [BoundingBox, ...] }
        """
        result: Dict[str, List[BoundingBox]] = {lid: [] for lid in self._polygons}

        for box in boxes:
            for lid in self._polygons:
                if self.point_in_lane(box.cx, box.cy, lid):
                    result[lid].append(box)
                    break    # each vehicle assigned to at most one lane

        return result

    def draw_rois(
        self,
        frame:  np.ndarray,
        counts: Optional[Dict[str, int]] = None,
    ) -> np.ndarray:
        """
        Overlay ROI polygons on a frame (BGR).  Used for debug visualisation.

        Args:
            frame:  BGR image
            counts: optional { lane_id: vehicle_count } for labels
        """
        overlay = frame.copy()
        palette = [
            (255, 80,  80),   # North — blue-ish
            (80, 255,  80),   # South — green
            (80,  80, 255),   # East  — red-ish
            (255, 255, 80),   # West  — yellow
        ]
        arm_map  = {"N": 0, "S": 1, "E": 2, "W": 3}
        alpha    = 0.25

        for lid, poly in self._polygons.items():
            arm    = lid[0]
            colour = palette[arm_map.get(arm, 0)]
            cv2.fillPoly(overlay, [poly], colour)
            cv2.polylines(frame, [poly], True, colour, 2)

            # centroid label
            M   = cv2.moments(poly)
            if M["m00"] != 0:
                lx = int(M["m10"] / M["m00"])
                ly = int(M["m01"] / M["m00"])
                label = lid
                if counts:
                    label += f" [{counts.get(lid, 0)}]"
                cv2.putText(frame, label, (lx - 30, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1, cv2.LINE_AA)

        return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    @property
    def lane_ids(self) -> List[str]:
        return list(self._polygons.keys())


# ══════════════════════════════════════════════════════════════════════════════
# 3.  EMISSIONS SCORER
# ══════════════════════════════════════════════════════════════════════════════

class EmissionsScorer:
    """
    Compute per-lane emissions proxy scores from vehicle class counts.

    All scores are RELATIVE (not g/km absolutes) and designed to provide
    a meaningful ordinal signal for the RL reward function.

    Urban Air Quality focus:
        • CO2  — climate impact (primary optimisation target)
        • NOx  — smog / respiratory harm
        • PMx  — fine particulate / carcinogenic risk

    Normalisation strategy:
        Raw scores are normalised by lane capacity (max_vehicles) so a
        fully-queued bus lane scores higher than a half-queued truck lane,
        reflecting real urban congestion cost.
    """

    MAX_VEHICLES_PER_LANE: int = 20   # approx jam density for 200 m lane

    def score_lane(
        self,
        detections: List[BoundingBox],
        normalise:  bool = False,
    ) -> Tuple[float, float, float, float]:
        """
        Calculate emissions scores for a single lane.

        Args:
            detections: list of BoundingBox detections assigned to this lane
            normalise:  if True, divide by MAX_VEHICLES_PER_LANE

        Returns:
            (combined_score, co2_score, nox_score, pmx_score)
        """
        co2 = nox = pmx = combined = 0.0

        for box in detections:
            w = EMISSION_WEIGHTS.get(box.cls_name, _DEFAULT_WEIGHT)
            co2      += w["co2"]
            nox      += w["nox"]
            pmx      += w["pmx"]
            combined += w["combined"]

        if normalise and self.MAX_VEHICLES_PER_LANE > 0:
            factor = self.MAX_VEHICLES_PER_LANE
            return combined/factor, co2/factor, nox/factor, pmx/factor

        return combined, co2, nox, pmx

    def score_intersection(
        self,
        lane_detections: Dict[str, List[BoundingBox]],
    ) -> Dict[str, Dict[str, float]]:
        """
        Score all lanes in one call.

        Returns:
            { lane_id: {"combined": f, "co2": f, "nox": f, "pmx": f} }
        """
        return {
            lid: dict(zip(
                ["combined", "co2", "nox", "pmx"],
                self.score_lane(dets)
            ))
            for lid, dets in lane_detections.items()
        }


# ══════════════════════════════════════════════════════════════════════════════
# 4.  YOLO PERCEPTION ENGINE  —  main pipeline
# ══════════════════════════════════════════════════════════════════════════════

class YOLOPerceptionEngine:
    """
    End-to-end perception pipeline:

        SUMO-GUI frame  →  YOLOv8n detection  →  ROI assignment
                        →  Emissions scoring   →  FrameData

    Performance optimisations:
        • Frame skipping:  only run YOLO every `inference_every` frames;
                           interpolate counts on skipped frames.
        • Half precision:  auto-enabled on CUDA GPUs.
        • Letterbox resize: YOLO receives 640×640 — no aspect-ratio distortion.
        • Result caching:  last detection reused on skipped frames.
        • NMS threshold:   tuned to reduce false positives in SUMO's clean
                           top-down render.

    Thread safety:
        process_frame() / get_live_traffic_data() are NOT thread-safe.
        Use PerceptionBridge for concurrent access.
    """

    def __init__(
        self,
        model_path:       str   = YOLO_MODEL_NAME,
        roi_manager:      Optional[ROIManager]    = None,
        emissions_scorer: Optional[EmissionsScorer] = None,
        conf:             float = YOLO_CONF_THRESH,
        iou:              float = YOLO_IOU_THRESH,
        img_size:         int   = YOLO_IMG_SIZE,
        inference_every:  int   = 1,     # run YOLO every N frames (1 = all)
        device:           str   = "",    # "" = auto (CUDA if available, else CPU)
        keep_raw_frame:   bool  = False,
    ) -> None:
        # ── Model ──────────────────────────────────────────────────────────
        log.info("🤖  Loading YOLOv8 model: %s", model_path)
        self._model = YOLO(model_path)
        self._model.fuse()               # fuse Conv+BN for faster inference

        self._conf           = conf
        self._iou            = iou
        self._img_size       = img_size
        self._device         = device
        self._inference_every = max(1, inference_every)
        self._keep_raw       = keep_raw_frame

        # ── Sub-systems ────────────────────────────────────────────────────
        self._roi     = roi_manager      or ROIManager()
        self._scorer  = emissions_scorer or EmissionsScorer()

        # ── State ──────────────────────────────────────────────────────────
        self._frame_counter:   int  = 0
        self._last_frame_data: Optional[FrameData] = None
        self._cached_boxes:    List[BoundingBox]   = []
        self._sim_time:        float = 0.0

        # Vehicle class filter — COCO IDs we care about
        self._vehicle_cls_ids: List[int] = list(VEHICLE_COCO_CLASSES.keys())

        log.info(
            "✅  YOLOPerceptionEngine ready | conf=%.2f | iou=%.2f | "
            "size=%d | every=%d frames",
            conf, iou, img_size, self._inference_every,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _run_yolo(self, frame: np.ndarray) -> List[BoundingBox]:
        """
        Run YOLOv8 inference on one BGR frame.

        Returns a list of BoundingBox objects filtered to vehicle classes.
        """
        h, w = frame.shape[:2]
        t0   = time.perf_counter()

        results = self._model.predict(
            source   = frame,
            conf     = self._conf,
            iou      = self._iou,
            imgsz    = self._img_size,
            classes  = self._vehicle_cls_ids,
            device   = self._device,
            verbose  = False,
            half     = (self._device != "cpu"),   # FP16 on GPU only
            augment  = False,
            agnostic_nms = False,
        )

        t1       = time.perf_counter()
        self._last_inference_ms = (t1 - t0) * 1000

        boxes: List[BoundingBox] = []
        for r in results:
            if r.boxes is None:
                continue
            xyxyn = r.boxes.xyxyn.cpu().numpy()    # normalised [0,1]
            confs = r.boxes.conf.cpu().numpy()
            clses = r.boxes.cls.cpu().numpy().astype(int)

            for (x1n, y1n, x2n, y2n), conf, cls_id in zip(xyxyn, confs, clses):
                # Convert normalised → pixel absolute
                x1 = x1n * w;  y1 = y1n * h
                x2 = x2n * w;  y2 = y2n * h
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                cls_name = VEHICLE_COCO_CLASSES.get(cls_id, "car")
                boxes.append(BoundingBox(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    cx=cx, cy=cy,
                    conf=float(conf),
                    cls_id=cls_id,
                    cls_name=cls_name,
                ))
        return boxes

    def _build_lane_detections(
        self,
        assigned: Dict[str, List[BoundingBox]],
        scores:   Dict[str, Dict[str, float]],
    ) -> Dict[str, LaneDetection]:
        """Merge assignment and emission scores into LaneDetection objects."""
        result: Dict[str, LaneDetection] = {}
        for lid in self._roi.lane_ids:
            dets  = assigned.get(lid, [])
            sc    = scores.get(lid, {"combined": 0.0, "co2": 0.0, "nox": 0.0, "pmx": 0.0})

            # Class histogram
            cls_counts: Dict[str, int] = {}
            for box in dets:
                cls_counts[box.cls_name] = cls_counts.get(box.cls_name, 0) + 1

            result[lid] = LaneDetection(
                lane_id         = lid,
                vehicle_count   = len(dets),
                class_counts    = cls_counts,
                emissions_score = sc["combined"],
                co2_score       = sc["co2"],
                nox_score       = sc["nox"],
                pmx_score       = sc["pmx"],
                detections      = dets,
            )
        return result

    # ── Public API ────────────────────────────────────────────────────────────

    def set_sim_time(self, t: float) -> None:
        """Inject current SUMO simulation time (called by TraCI loop)."""
        self._sim_time = t

    def process_frame(self, frame: np.ndarray) -> FrameData:
        """
        Full pipeline: detect → assign → score → package.

        Args:
            frame: BGR numpy array from ScreenCapture

        Returns:
            FrameData — complete perception snapshot
        """
        t_wall = time.time()
        h, w   = frame.shape[:2]

        # ── Auto-rescale ROIs if frame size changed ────────────────────────
        if (w, h) != (self._roi._fw, self._roi._fh):
            self._roi.rescale(w, h)

        # ── Frame skipping logic ───────────────────────────────────────────
        run_yolo = (self._frame_counter % self._inference_every == 0)
        self._frame_counter += 1
        self._last_inference_ms = 0.0

        if run_yolo:
            self._cached_boxes = self._run_yolo(frame)

        boxes = self._cached_boxes

        # ── ROI assignment ─────────────────────────────────────────────────
        assigned = self._roi.assign_detections_to_lanes(boxes)

        # ── Emissions scoring ──────────────────────────────────────────────
        scores = self._scorer.score_intersection(assigned)

        # ── Package into LaneDetection objects ────────────────────────────
        lane_data = self._build_lane_detections(assigned, scores)

        total_vehicles  = sum(ld.vehicle_count   for ld in lane_data.values())
        total_emissions = sum(ld.emissions_score for ld in lane_data.values())

        fd = FrameData(
            timestamp       = t_wall,
            sim_time        = self._sim_time,
            frame_id        = self._frame_counter,
            fps             = 0.0,       # filled by caller
            raw_frame       = frame.copy() if self._keep_raw else None,
            lane_data       = lane_data,
            total_vehicles  = total_vehicles,
            total_emissions = total_emissions,
            inference_ms    = self._last_inference_ms,
        )
        self._last_frame_data = fd
        return fd

    def get_live_traffic_data(self, frame: np.ndarray) -> Dict[str, Dict]:
        """
        Convenience wrapper — returns the canonical live_traffic_data dict
        directly (Phase 3 / Phase 4 contract).

        Args:
            frame: BGR frame from ScreenCapture.grab()

        Returns:
            {
              "N_in_0": {"count": 3, "emissions_score": 4.26, ...},
              ...
            }
        """
        fd = self.process_frame(frame)
        return fd.to_live_traffic_data()

    def visualise(
        self,
        frame:     np.ndarray,
        frame_data: FrameData,
    ) -> np.ndarray:
        """
        Render YOLO bounding boxes + ROI overlays + per-lane HUD on a frame.
        Returns annotated BGR image (does NOT show window — caller decides).
        """
        vis = frame.copy()

        # ── ROI overlay ────────────────────────────────────────────────────
        counts = {lid: ld.vehicle_count for lid, ld in frame_data.lane_data.items()}
        vis    = self._roi.draw_rois(vis, counts)

        # ── Bounding boxes ─────────────────────────────────────────────────
        cls_colours = {
            "car":        (200, 200, 255),
            "bus":        (80,  220,  80),
            "truck":      (80,  80,  220),
            "motorcycle": (220, 220,  80),
        }
        for box in self._cached_boxes:
            colour = cls_colours.get(box.cls_name, (200, 200, 200))
            x1, y1, x2, y2 = int(box.x1), int(box.y1), int(box.x2), int(box.y2)
            cv2.rectangle(vis, (x1, y1), (x2, y2), colour, 2)
            label = f"{box.cls_name} {box.conf:.2f}"
            cv2.putText(vis, label, (x1, max(y1 - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1, cv2.LINE_AA)

        # ── HUD ────────────────────────────────────────────────────────────
        hud_lines = [
            f"EcoSync Perception  |  Frame {frame_data.frame_id}",
            f"SimTime: {frame_data.sim_time:.1f}s  |  YOLO: {frame_data.inference_ms:.1f}ms",
            f"Vehicles: {frame_data.total_vehicles}  |  Emissions Score: {frame_data.total_emissions:.2f}",
        ]
        for i, line in enumerate(hud_lines):
            cv2.putText(vis, line, (10, 20 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        return vis

    @property
    def roi_manager(self) -> ROIManager:
        return self._roi


# ══════════════════════════════════════════════════════════════════════════════
# 5.  PERCEPTION BRIDGE  —  thread-safe adapter for LSTM / RL
# ══════════════════════════════════════════════════════════════════════════════

class PerceptionBridge:
    """
    Thread-safe bridge between the SUMO-TraCI loop and the LSTM / RL agent.

    The bridge runs a background capture→detect→score loop and maintains
    a ring buffer of FrameData objects.  The LSTM can consume the buffer
    at its own pace without blocking the simulation.

    Usage:
        bridge = PerceptionBridge(engine, capturer)
        bridge.start()

        # In LSTM / RL thread:
        frames = bridge.get_latest_frame_data(n=10)   # last 10 frames
        data   = bridge.get_live_traffic_data()        # most recent snapshot

        bridge.stop()

    Performance:
        target_fps controls the capture rate.  The background thread
        dynamically sleeps to hit the target without busy-waiting.
    """

    def __init__(
        self,
        engine:       YOLOPerceptionEngine,
        capturer:     ScreenCapture,
        buffer_size:  int   = 50,        # ring buffer depth (LSTM window + headroom)
        target_fps:   float = 15.0,      # max capture rate
    ) -> None:
        self._engine     = engine
        self._capturer   = capturer
        self._buffer:    Deque[FrameData] = deque(maxlen=buffer_size)
        self._target_fps = target_fps
        self._interval   = 1.0 / target_fps

        self._lock       = threading.Lock()
        self._stop_event = threading.Event()
        self._thread:    Optional[threading.Thread] = None
        self._running    = False

    # ── Background capture loop ───────────────────────────────────────────────

    def _loop(self) -> None:
        log.info("🔄  PerceptionBridge capture loop started (%.0f FPS cap)", self._target_fps)
        while not self._stop_event.is_set():
            t0 = time.perf_counter()

            frame = self._capturer.grab()
            if frame is None:
                time.sleep(0.05)
                continue

            if _TRACI_AVAILABLE and traci.isLoaded():
                self._engine.set_sim_time(traci.simulation.getTime())

            fd      = self._engine.process_frame(frame)
            fd.fps  = self._capturer.fps

            with self._lock:
                self._buffer.append(fd)

            elapsed = time.perf_counter() - t0
            sleep_t = max(0.0, self._interval - elapsed)
            time.sleep(sleep_t)

        log.info("🛑  PerceptionBridge capture loop stopped.")

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background capture thread."""
        if self._running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target   = self._loop,
            name     = "EcoSync-Perception",
            daemon   = True,
        )
        self._thread.start()
        self._running = True
        log.info("▶️   PerceptionBridge started.")

    def stop(self) -> None:
        """Signal the background thread to exit and wait for it."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        self._running = False
        log.info("⏹️   PerceptionBridge stopped.")

    def get_latest_frame_data(self, n: int = 1) -> List[FrameData]:
        """
        Return the most recent n FrameData objects from the ring buffer.
        Used by the LSTM to build temporal sequences.

        Args:
            n: number of frames (1 = just the latest)

        Returns:
            List[FrameData] — oldest first, length ≤ n
        """
        with self._lock:
            frames = list(self._buffer)
        return frames[-n:]

    def get_live_traffic_data(self) -> Optional[Dict[str, Dict]]:
        """
        Return the canonical live_traffic_data dict from the most recent frame.
        Thread-safe — safe to call from any thread.
        """
        with self._lock:
            if not self._buffer:
                return None
            return self._buffer[-1].to_live_traffic_data()

    def get_lstm_sequence(
        self,
        n:         int = 10,
        lane_ids:  Optional[List[str]] = None,
    ) -> Optional[np.ndarray]:
        """
        Build a (T, N_lanes, N_features) tensor for the LSTM.

        Features per lane per timestep:
            [count, emissions_score, co2, nox, pmx]  → 5 features

        Args:
            n:         sequence length (timesteps)
            lane_ids:  lanes to include; None = all 12

        Returns:
            np.ndarray shape (T, L, 5) or None if buffer too short
        """
        frames = self.get_latest_frame_data(n)
        if len(frames) < 2:
            return None

        if lane_ids is None:
            lane_ids = DEFAULT_LANE_IDS

        seq = []
        for fd in frames:
            ts = []
            for lid in lane_ids:
                ld = fd.lane_data.get(lid)
                if ld:
                    ts.append([
                        ld.vehicle_count,
                        ld.emissions_score,
                        ld.co2_score,
                        ld.nox_score,
                        ld.pmx_score,
                    ])
                else:
                    ts.append([0.0, 0.0, 0.0, 0.0, 0.0])
            seq.append(ts)

        return np.array(seq, dtype=np.float32)   # (T, L, 5)

    @property
    def buffer_len(self) -> int:
        with self._lock:
            return len(self._buffer)

    @property
    def is_running(self) -> bool:
        return self._running


# ══════════════════════════════════════════════════════════════════════════════
# 6.  FACTORY FUNCTION  —  one-call integration with EcoSyncSim
# ══════════════════════════════════════════════════════════════════════════════

def build_perception_pipeline(
    capture_region:   Optional[Dict[str, int]] = None,
    roi_override:     Optional[Dict[str, List[Tuple[int, int]]]] = None,
    model_path:       str   = YOLO_MODEL_NAME,
    inference_every:  int   = 2,      # skip every other frame for speed
    target_fps:       float = 15.0,
    buffer_size:      int   = 50,
    keep_raw_frame:   bool  = False,
    gui_window_auto:  bool  = True,
) -> Tuple[PerceptionBridge, YOLOPerceptionEngine, ScreenCapture]:
    """
    Construct the full Phase 2 perception stack in one call.

    Args:
        capture_region:    screen pixel region {"left","top","width","height"}.
                           None = auto-detect SUMO-GUI window, else full monitor.
        roi_override:      custom ROI polygons; None = use built-in defaults.
        model_path:        path to YOLOv8 weights (.pt).
        inference_every:   run YOLO every N frames (1 = every frame).
        target_fps:        capture loop target FPS.
        buffer_size:       ring buffer depth for LSTM.
        keep_raw_frame:    store BGR frames in FrameData (memory cost).
        gui_window_auto:   attempt to auto-detect the SUMO-GUI window on screen.

    Returns:
        (PerceptionBridge, YOLOPerceptionEngine, ScreenCapture)

    Example:
        bridge, engine, capturer = build_perception_pipeline()
        bridge.start()

        # In main loop (with traci active):
        live_data = bridge.get_live_traffic_data()
        # → {"N_in_0": {"count": 2, "emissions_score": 2.26, ...}, ...}

        # For LSTM input:
        sequence = bridge.get_lstm_sequence(n=10)
        # → np.ndarray (10, 12, 5)
    """
    capturer = ScreenCapture(region=capture_region)
    if gui_window_auto and capture_region is None:
        capturer.auto_detect_sumo_window()

    roi_mgr = ROIManager(
        frame_width  = capturer.region.get("width",  1280),
        frame_height = capturer.region.get("height",  720),
    )
    if roi_override:
        roi_mgr.set_rois(roi_override)

    scorer = EmissionsScorer()
    engine = YOLOPerceptionEngine(
        model_path       = model_path,
        roi_manager      = roi_mgr,
        emissions_scorer = scorer,
        inference_every  = inference_every,
        keep_raw_frame   = keep_raw_frame,
    )
    bridge = PerceptionBridge(
        engine      = engine,
        capturer    = capturer,
        buffer_size = buffer_size,
        target_fps  = target_fps,
    )

    log.info("═" * 60)
    log.info("  EcoSync Perception Pipeline  —  Phase 2")
    log.info("  Model        : %s", model_path)
    log.info("  Infer every  : %d frame(s)", inference_every)
    log.info("  Target FPS   : %.0f", target_fps)
    log.info("  Buffer depth : %d", buffer_size)
    log.info("  ROI lanes    : %d", len(roi_mgr.lane_ids))
    log.info("═" * 60)

    return bridge, engine, capturer


# ══════════════════════════════════════════════════════════════════════════════
# 7.  STANDALONE DEMO  (python ecosync_perception.py --demo)
# ══════════════════════════════════════════════════════════════════════════════

def _demo_with_webcam(duration_s: float = 30.0) -> None:
    """
    Offline demo using webcam (no SUMO required).
    Validates the full pipeline end-to-end.
    """
    import cv2 as _cv2

    log.info("🎥  Webcam demo — press Q to quit")
    cap = _cv2.VideoCapture(0)
    if not cap.isOpened():
        log.error("No webcam found.")
        return

    ret, frame0 = cap.read()
    if not ret:
        log.error("Cannot read from webcam.")
        return

    h, w = frame0.shape[:2]
    roi_mgr = ROIManager(frame_width=w, frame_height=h)
    engine  = YOLOPerceptionEngine(inference_every=1, keep_raw_frame=False)
    engine._roi = roi_mgr

    t0 = time.time()
    while time.time() - t0 < duration_s:
        ret, frame = cap.read()
        if not ret:
            break

        fd  = engine.process_frame(frame)
        vis = engine.visualise(frame, fd)
        ltd = fd.to_live_traffic_data()

        # Print summary every 30 frames
        if fd.frame_id % 30 == 0:
            log.info(
                "Frame %d | %.1f FPS | %d vehicles | Emissions: %.2f | YOLO: %.1f ms",
                fd.frame_id, cap.get(_cv2.CAP_PROP_FPS),
                fd.total_vehicles, fd.total_emissions, fd.inference_ms,
            )
            for lid, v in ltd.items():
                if v["count"] > 0:
                    log.info("  %s → count=%d  score=%.2f", lid, v["count"], v["emissions_score"])

        _cv2.imshow("EcoSync Perception Demo", vis)
        if _cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    _cv2.destroyAllWindows()
    log.info("✅  Webcam demo complete.")


def _demo_with_sumo(gui_region: Optional[Dict[str, int]] = None) -> None:
    """
    Integration demo — requires a running SUMO-GUI (from Phase 1).
    Call AFTER sim.reset() in ecosync_sim.py.
    """
    bridge, engine, capturer = build_perception_pipeline(
        capture_region  = gui_region,
        inference_every = 2,
        target_fps      = 15.0,
        keep_raw_frame  = True,
    )
    bridge.start()
    time.sleep(1.0)   # let buffer fill

    log.info("▶️   Running perception demo (10 s)…")
    t0 = time.time()
    while time.time() - t0 < 10.0:
        ltd = bridge.get_live_traffic_data()
        seq = bridge.get_lstm_sequence(n=5)

        if ltd:
            total_em = sum(v["emissions_score"] for v in ltd.values())
            log.info("Live emissions total: %.2f  |  buffer=%d", total_em, bridge.buffer_len)

        if seq is not None:
            log.info("LSTM tensor shape: %s", seq.shape)

        time.sleep(0.5)

    bridge.stop()


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="EcoSync — Phase 2 YOLOv8 Perception Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--demo",         action="store_true",  help="Run webcam offline demo")
    parser.add_argument("--demo-sumo",    action="store_true",  help="Run SUMO-GUI integration demo")
    parser.add_argument("--duration",     type=float, default=30.0, help="Demo duration (s)")
    parser.add_argument("--region",       type=str,   default=None,
                        help='Screen capture region as JSON: \'{"left":0,"top":0,"width":1280,"height":720}\'')
    parser.add_argument("--model",        type=str,   default=YOLO_MODEL_NAME)
    parser.add_argument("--infer-every",  type=int,   default=2)
    parser.add_argument("--fps",          type=float, default=15.0)
    args = parser.parse_args()

    region = None
    if args.region:
        import json
        region = json.loads(args.region)

    if args.demo:
        _demo_with_webcam(duration_s=args.duration)
    elif args.demo_sumo:
        _demo_with_sumo(gui_region=region)
    else:
        # Just build the pipeline and print status
        bridge, engine, capturer = build_perception_pipeline(
            capture_region = region,
            model_path     = args.model,
            inference_every = args.infer_every,
            target_fps     = args.fps,
        )
        log.info("📋  Perception pipeline built.  Call bridge.start() to begin.")
        log.info("    live_data = bridge.get_live_traffic_data()")
        log.info("    sequence  = bridge.get_lstm_sequence(n=10)  # → (10,12,5)")