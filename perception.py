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
    3:  "motorcycle",
    5:  "bus",
    7:  "truck",
    9:  "emergency",
}

# ── Emissions proxy weights (Urban Air Quality focus) ─────────────────────────
#   Motorcycle ~   80 g/km  →  weight 0.6
#   Truck      ~  650 g/km  →  weight 4.6
#   Bus        ~  900 g/km  →  weight 6.4
#
# NOx proxy (relative mg/km, Euro-average):
#   Car 40 / Motorcycle 60 / Truck 450 / Bus 500
#   → NOx multiplier added as secondary index
#
# Combined score = CO2_weight + NOx_weight  (then normalised 0–10 per lane)
# HBEFA 4.1 standards (Grams of CO2/km)
EMISSION_WEIGHTS: Dict[str, Dict[str, float]] = {
    "car":        {"co2": 140.0,  "nox": 0.08, "pmx": 0.05, "combined": 140.0},
    "motorcycle": {"co2": 80.0,   "nox": 0.12, "pmx": 0.03, "combined": 80.0},
    "truck":      {"co2": 650.0,  "nox": 0.90, "pmx": 0.40, "combined": 650.0},
    "bus":        {"co2": 900.0,  "nox": 1.00, "pmx": 0.60, "combined": 900.0},
    "emergency":  {"co2": 140.0,  "nox": 0.08, "pmx": 0.05, "combined": 140.0},
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
YOLO_MODEL_NAME: str  = "yolov8s.pt"   # Nano — fastest; swap to yolov8s for +mAP
YOLO_CONF_THRESH: float = 0.25          # lower = more detections, more FP
YOLO_IOU_THRESH:  float = 0.45          # NMS overlap threshold
YOLO_IMG_SIZE:    int   = 640           # inference resolution


# ══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BoundingBox:
    """Normalised [0,1] bounding box from YOLO + pixel centre."""
    x1:           float
    y1:           float
    x2:           float
    y2:           float
    cx:           float          # pixel centroid x (absolute)
    cy:           float          # pixel centroid y (absolute)
    conf:         float
    cls_id:       int
    cls_name:     str
    is_ev:        bool = False
    is_ambulance: bool = False   # True when COCO class 9 (emergency) detected

    @property
    def area_px(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


@dataclass
class LaneDetection:
    """Per-lane detection result for one video frame."""
    lane_id:          str
    vehicle_count:    int               = 0
    class_counts:     Dict[str, int]    = field(default_factory=dict)
    emissions_score:  float             = 0.0   # combined CO2+NOx proxy
    co2_score:        float             = 0.0
    nox_score:        float             = 0.0
    pmx_score:        float             = 0.0
    detections:       List[BoundingBox] = field(default_factory=list)
    ev_count:         int               = 0
    fuel_count:       int               = 0     # non-EV vehicles in lane
    ambulance_count:  int               = 0     # emergency vehicles in lane


@dataclass
class FrameData:
    """
    Complete perception output for a single captured frame.
    Fed into the LSTM sequence buffer (Phase 3).
    """
    timestamp:           float                        # wall-clock time
    sim_time:            float                        # SUMO simulation time (s)
    frame_id:            int
    fps:                 float
    raw_frame:           Optional[np.ndarray]         # BGR image (kept for debug)
    lane_data:           Dict[str, LaneDetection]     # keyed by lane_id
    total_vehicles:      int                = 0
    total_evs:           int                = 0
    total_fuel_vehicles: int                = 0       # non-EV vehicles detected
    total_ambulances:    int                = 0       # emergency vehicles detected
    ambulance_detected:  bool               = False   # True if any ambulance in frame
    total_emissions:     float              = 0.0
    co2_rate_g_per_frame: float             = 0.0    # estimated CO2 g/km for this frame
    inference_ms:        float              = 0.0     # YOLO latency

    def to_live_traffic_data(self) -> Dict[str, Dict]:
        """
        Canonical output contract for Phase 3 LSTM and Phase 4 RL.

        Returns:
            {
              "N_in_0": {"count": 3, "emissions_score": 4.26,
                         "co2": 3.0, "nox": 0.24, "pmx": 0.15,
                         "class_counts": {"car": 2, "bus": 1},
                         "ambulance_count": 0},
              ...
            }
        """
        return {
            lid: {
                "count":           ld.vehicle_count,
                "ev_count":        ld.ev_count,
                "fuel_count":      ld.fuel_count,
                "ambulance_count": ld.ambulance_count,
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


class FileCapture:
    """
    Video file capture for production prototype using cv2.VideoCapture.
    Loops automatically when reaching the end of the file.
    Paces frame delivery to the video's native FPS so playback looks correct.
    """
    def __init__(self, video_path: str = "traffi.mp4") -> None:
        self._video_path = video_path
        self._cap = cv2.VideoCapture(video_path)
        if not self._cap.isOpened():
            log.error("FileCapture error: Cannot open %s", video_path)

        # Read native FPS for rate-controlled frame delivery
        self._native_fps = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        self._native_interval = 1.0 / max(self._native_fps, 1.0)
        self._last_grab_t = 0.0  # time of last frame delivery

        self._frame_id  = 0
        self._last_t    = time.perf_counter()
        self._fps       = 0.0
        self._fps_alpha = 0.1
        log.info("📷  FileCapture initialised | video=%s | native_fps=%.1f",
                 video_path, self._native_fps)

    def _reopen(self) -> bool:
        """Re-open video from the beginning for looping."""
        self._cap.release()
        self._cap = cv2.VideoCapture(self._video_path)
        opened = self._cap.isOpened()
        if opened:
            log.info("🔄  FileCapture looped: %s", self._video_path)
        else:
            log.error("FileCapture error: Failed to reopen %s", self._video_path)
        return opened

    def grab(self) -> Optional[np.ndarray]:
        """
        Read the next frame from the video file.
        Paces delivery to the video's native FPS — if called too fast,
        returns None until the next frame is due. This ensures a 10s
        video takes 10s to play through, not 3s.
        """
        # --- Frame-rate pacing: don't deliver faster than native FPS ---
        now = time.perf_counter()
        if (now - self._last_grab_t) < self._native_interval:
            return None  # too early, caller should sleep and retry

        if not self._cap.isOpened():
            if not self._reopen():
                return None

        ret, frame = self._cap.read()
        if not ret:
            if not self._reopen():
                return None
            ret, frame = self._cap.read()
            if not ret:
                return None

        self._last_grab_t = now
        inst_fps = 1.0 / max(now - self._last_t, 1e-6)
        self._fps   = self._fps_alpha * inst_fps + (1 - self._fps_alpha) * self._fps
        self._last_t = now
        self._frame_id += 1
        return frame

    @property
    def fps(self) -> float:
        return round(self._fps, 1)

    @property
    def frame_id(self) -> int:
        return self._frame_id

    @property
    def region(self) -> Dict[str, int]:
        """Return frame dimensions as a region dict for ROIManager."""
        if self._cap.isOpened():
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if w > 0 and h > 0:
                return {"left": 0, "top": 0, "width": w, "height": h}
        return {"left": 0, "top": 0, "width": 1280, "height": 720}



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
            if box.is_ev:
                continue
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
        
        self._night_mode:      bool  = False
        self._fog_mode:        bool  = False
        self._rain_mode:       bool  = False
        self._rain_buffer:     List[np.ndarray] = []

        # Vehicle class filter — COCO IDs we care about
        self._vehicle_cls_ids: List[int] = list(VEHICLE_COCO_CLASSES.keys())

        log.info(
            "✅  YOLOPerceptionEngine ready | conf=%.2f | iou=%.2f | "
            "size=%d | every=%d frames",
            conf, iou, img_size, self._inference_every,
        )

    def set_atmospheric_modes(self, night: bool, fog: bool, rain: bool) -> None:
        self._night_mode = night
        self._fog_mode = fog
        self._rain_mode = rain

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _detect_ev_by_plate_color(frame: np.ndarray, x1: float, y1: float,
                                   x2: float, y2: float) -> bool:
        """
        Detect whether a vehicle has a GREEN number plate (Indian EV standard).

        In India, Electric Vehicles carry a GREEN number plate with white text.
        Petrol / Diesel vehicles use white (private) or yellow (commercial) plates.

        Algorithm:
          1. Extract the bottom 25% of the vehicle bounding box
             (number plates are mounted at the front/rear, near the bottom).
          2. Convert the crop to HSV colour space.
          3. Threshold for Indian EV green plate colour:
               Hue   : 35 – 90  (green band in OpenCV 0-179 scale)
               Sat   : ≥ 60     (vivid, not washed-out)
               Val   : ≥ 60     (not too dark)
          4. Return True if ≥ 8 % of the crop pixels satisfy the mask.

        Returns:
            True  → vehicle has a green plate → treat as EV (zero emissions).
            False → petrol / diesel / unknown vehicle.
        """
        h_frame, w_frame = frame.shape[:2]

        # --- clamp bounding box to frame bounds ---
        ix1 = max(0,       int(x1))
        ix2 = min(w_frame, int(x2))
        iy1 = max(0,       int(y1))
        iy2 = min(h_frame, int(y2))

        box_h = iy2 - iy1
        box_w = ix2 - ix1
        if box_h < 10 or box_w < 10:
            return False  # bounding box too small to analyse

        # --- crop the bottom 25 % of the bounding box (plate region) ---
        plate_y1 = iy2 - max(6, int(box_h * 0.25))
        plate_crop = frame[plate_y1:iy2, ix1:ix2]

        if plate_crop.size == 0:
            return False

        # --- HSV threshold for Indian EV green plate ---
        hsv = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2HSV)

        # Green hue range in OpenCV (0-179): 35-90
        lower_green = np.array([35,  60,  60], dtype=np.uint8)
        upper_green = np.array([90, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_green, upper_green)

        green_ratio = mask.sum() / 255.0 / max(mask.size, 1)
        return green_ratio >= 0.08   # at least 8 % of plate crop is green

    def _run_yolo(self, frame: np.ndarray) -> List[BoundingBox]:
        """
        Run YOLOv8 inference on one BGR frame.

        Returns a list of BoundingBox objects filtered to vehicle classes.
        Ambulances (COCO class 9) are explicitly flagged with is_ambulance=True.
        EVs are detected by NUMBER PLATE COLOUR (green plate = Indian EV standard).
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
            half     = (self._device != "cpu"),
            augment  = False,
            agnostic_nms = False,
        )

        t1 = time.perf_counter()
        self._last_inference_ms = (t1 - t0) * 1000

        boxes: List[BoundingBox] = []
        for r in results:
            if r.boxes is None:
                continue
            xyxyn = r.boxes.xyxyn.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clses = r.boxes.cls.cpu().numpy().astype(int)

            for (x1n, y1n, x2n, y2n), conf, cls_id in zip(xyxyn, confs, clses):
                x1 = x1n * w;  y1 = y1n * h
                x2 = x2n * w;  y2 = y2n * h
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                cls_name = VEHICLE_COCO_CLASSES.get(cls_id, "car")

                # ── Feature 3: Ambulance / Emergency Detection ──────────────
                is_ambulance = (cls_id == 9)  # COCO class 9 = emergency vehicle

                # ── Feature 2: EV Detection by NUMBER PLATE COLOUR ──────────
                # Indian standard: Green number plate = Electric Vehicle
                # Petrol/Diesel: White (private) or Yellow (commercial) plate
                is_ev = False
                if not is_ambulance and cls_name in ("car", "bus", "truck", "motorcycle"):
                    is_ev = self._detect_ev_by_plate_color(frame, x1, y1, x2, y2)

                # Night-Vision: Boost confidence if bright spots detected
                if self._night_mode:
                    crop = frame[max(0,int(y1)):int(y2), max(0,int(x1)):int(x2)]
                    if crop.size > 0:
                        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                        if np.any(gray_crop > 200):
                            conf = min(float(conf) + 0.3, 1.0)

                boxes.append(BoundingBox(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    cx=cx, cy=cy,
                    conf=float(conf),
                    cls_id=cls_id,
                    cls_name=cls_name,
                    is_ev=is_ev,
                    is_ambulance=is_ambulance,
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

            # Class histogram + EV / fuel / ambulance counts
            cls_counts:     Dict[str, int] = {}
            ev_counts     = 0
            fuel_counts   = 0
            ambul_counts  = 0
            for box in dets:
                cls_counts[box.cls_name] = cls_counts.get(box.cls_name, 0) + 1
                if box.is_ambulance:
                    ambul_counts += 1
                elif box.is_ev:
                    ev_counts += 1
                else:
                    fuel_counts += 1

            result[lid] = LaneDetection(
                lane_id         = lid,
                vehicle_count   = len(dets),
                class_counts    = cls_counts,
                emissions_score = sc["combined"],
                co2_score       = sc["co2"],
                nox_score       = sc["nox"],
                pmx_score       = sc["pmx"],
                detections      = dets,
                ev_count        = ev_counts,
                fuel_count      = fuel_counts,
                ambulance_count = ambul_counts,
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

        # Atmospheric modules
        if self._fog_mode:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl,a,b))
            frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            
        if self._rain_mode:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if len(self._rain_buffer) < 3:
                self._rain_buffer = [gray, gray, gray]
            self._rain_buffer.pop(0)
            self._rain_buffer.append(gray)
            d1 = cv2.absdiff(self._rain_buffer[0], self._rain_buffer[1])
            d2 = cv2.absdiff(self._rain_buffer[1], self._rain_buffer[2])
            mask = cv2.bitwise_and(d1, d2)
            _, thresh = cv2.threshold(mask, 25, 255, cv2.THRESH_BINARY)
            blurred = cv2.medianBlur(frame, 5)
            mask_3ch = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            frame = np.where(mask_3ch == 255, blurred, frame)
            
        if self._night_mode:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            bright_3ch = cv2.cvtColor(bright_mask, cv2.COLOR_GRAY2BGR)
            frame = cv2.addWeighted(frame, 0.8, bright_3ch, 0.2, 0)

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
        total_evs       = sum(box.is_ev        for box in boxes)
        total_fuels     = sum(box.is_ambulance is False and box.is_ev is False for box in boxes)
        total_ambulances= sum(box.is_ambulance for box in boxes)
        total_emissions = sum(ld.emissions_score for ld in lane_data.values())

        # ── Feature 2: CO2 rate based on current vehicle mix ───────────────
        # Weighted sum of HBEFA 4.1 CO2 g/km values for all fuel vehicles
        co2_rate = 0.0
        for box in boxes:
            if box.is_ambulance or box.is_ev:
                continue  # EVs emit 0, ambulances excluded from CO2 metric
            w = EMISSION_WEIGHTS.get(box.cls_name, _DEFAULT_WEIGHT)
            co2_rate += w["co2"]

        fd = FrameData(
            timestamp            = t_wall,
            sim_time             = self._sim_time,
            frame_id             = self._frame_counter,
            fps                  = 0.0,       # filled by caller
            raw_frame            = frame.copy() if self._keep_raw else None,
            lane_data            = lane_data,
            total_vehicles       = total_vehicles,
            total_evs            = int(total_evs),
            total_fuel_vehicles  = int(total_fuels),
            total_ambulances     = int(total_ambulances),
            ambulance_detected   = (int(total_ambulances) > 0),
            total_emissions      = total_emissions,
            co2_rate_g_per_frame = co2_rate,
            inference_ms         = self._last_inference_ms,
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
        Highlights ambulances with red flashing border.
        Returns annotated BGR image (does NOT show window — caller decides).
        """
        vis = frame.copy()

        # ── Feature 3: Ambulance Alert Overlay ────────────────────────────
        if frame_data.ambulance_detected:
            # Draw thick red border around entire frame
            h_f, w_f = vis.shape[:2]
            cv2.rectangle(vis, (0, 0), (w_f - 1, h_f - 1), (0, 0, 255), 8)
            cv2.putText(vis, "🚨 AMBULANCE DETECTED — CLEAR CORRIDOR",
                        (10, h_f - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0, 0, 255), 2, cv2.LINE_AA)

        # ── ROI overlay ────────────────────────────────────────────────────
        counts = {lid: ld.vehicle_count for lid, ld in frame_data.lane_data.items()}
        vis    = self._roi.draw_rois(vis, counts)

        # ── Bounding boxes ─────────────────────────────────────────────────
        cls_colours = {
            "car":        (200, 200, 255),
            "bus":        ( 80, 220,  80),
            "truck":      ( 80,  80, 220),
            "motorcycle": (220, 220,  80),
        }
        for box in self._cached_boxes:
            x1, y1, x2, y2 = int(box.x1), int(box.y1), int(box.x2), int(box.y2)

            if box.is_ambulance:
                # Feature 3: Red box + AMBULANCE label
                colour    = (0, 0, 255)
                label     = f"AMBULANCE {box.conf:.2f}"
                thickness = 3
            elif box.is_ev:
                # Feature 2: Bright green box = EV (green number plate)
                colour    = (0, 255, 80)   # vivid green in BGR
                label     = f"EV {box.cls_name} {box.conf:.2f}"
                thickness = 2
            else:
                # Feature 1: Standard class colours (fuel vehicles)
                colour    = cls_colours.get(box.cls_name, (200, 200, 200))
                label     = f"{box.cls_name} {box.conf:.2f}"
                thickness = 2

            cv2.rectangle(vis, (x1, y1), (x2, y2), colour, thickness)
            cv2.putText(vis, label, (x1, max(y1 - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1, cv2.LINE_AA)

            # ── Draw simulated number plate at bottom of bounding box ───────
            plate_h    = max(5, (y2 - y1) // 6)
            plate_y1_d = y2 - plate_h
            plate_w    = (x2 - x1)

            if box.is_ambulance:
                plate_col = (0, 0, 255)       # red plate
                plate_txt = "AMBU"
            elif box.is_ev:
                plate_col = (0, 200, 60)      # green plate (Indian EV standard)
                plate_txt = "EV"
            else:
                # White plate for private fuel vehicles
                plate_col = (220, 220, 220)
                plate_txt = "FUEL"

            cv2.rectangle(vis, (x1, plate_y1_d), (x2, y2), plate_col, -1)
            cv2.putText(vis, plate_txt,
                        (x1 + plate_w // 4, y2 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 0, 0), 1, cv2.LINE_AA)


        # ── HUD (Feature 1 + 2 stats) ──────────────────────────────────────
        ev_pct = (frame_data.total_evs / max(1, frame_data.total_vehicles)) * 100
        hud_lines = [
            f"EcoSync Perception  |  Frame {frame_data.frame_id}",
            f"Vehicles: {frame_data.total_vehicles}  |  Fuel: {frame_data.total_fuel_vehicles}  |  EV: {frame_data.total_evs} ({ev_pct:.0f}%)",
            f"CO2 Rate: {frame_data.co2_rate_g_per_frame:.0f} g/km  |  YOLO: {frame_data.inference_ms:.1f}ms",
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
        buffer_size:  int   = 50,
        target_fps:   float = 15.0,
    ) -> None:
        self._engine     = engine
        self._capturer   = capturer
        self._buffer:    Deque[FrameData] = deque(maxlen=buffer_size)
        self._target_fps = target_fps
        self._interval   = 1.0 / target_fps

        self._lock        = threading.Lock()
        self._stop_event  = threading.Event()
        self._pause_event = threading.Event()   # pause support
        self._thread:     Optional[threading.Thread] = None
        self._running     = False
        self._current_sim_time = 0.0

    def set_sim_time(self, t: float) -> None:
        """Thread-safe update of simulation time."""
        self._current_sim_time = t

    def set_paused(self, paused: bool) -> None:
        """Pause or resume the capture loop. When paused, no frames are read."""
        if paused:
            self._pause_event.set()
        else:
            self._pause_event.clear()

    def set_atmospheric_modes(self, night: bool, fog: bool, rain: bool) -> None:
        self._engine.set_atmospheric_modes(night, fog, rain)

    # ── Background capture loop ───────────────────────────────────────────────

    def _loop(self) -> None:
        log.info("🔄  PerceptionBridge capture loop started (%.0f FPS cap)", self._target_fps)
        while not self._stop_event.is_set():
            # ── Pause: sleep while paused, don't read frames ──────────
            if self._pause_event.is_set():
                time.sleep(0.1)
                continue

            t0 = time.perf_counter()

            frame = self._capturer.grab()
            if frame is None:
                time.sleep(0.02)  # frame-rate pacing returned None
                continue

            self._engine.set_sim_time(self._current_sim_time)

            fd      = self._engine.process_frame(frame)
            fd.fps  = self._capturer.fps

            with self._lock:
                self._buffer.append(fd)

            if self._engine._frame_counter % 50 == 0:
                log.info("📊  PerceptionBridge: processed %d frames (current FPS: %.1f)",
                         self._engine._frame_counter, self._capturer.fps)

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
    video_path:       Optional[str] = None,
) -> Tuple[PerceptionBridge, YOLOPerceptionEngine, Any]:
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
    if video_path:
        capturer = FileCapture(video_path)
    else:
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