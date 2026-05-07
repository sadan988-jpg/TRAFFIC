import React, { useState, useEffect, useRef } from 'react';
import {
  Activity, Target, Zap, Cpu, Play, Pause,
  BarChart3, Layers, Wind, Car, Siren, TrendingUp,
  AlertOctagon, CheckCircle, Gauge
} from 'lucide-react';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer
} from 'recharts';

// ── Custom Hooks ─────────────────────────────────────────────────────────────
const useInterval = (callback, delay) => {
  const savedCallback = useRef();
  useEffect(() => { savedCallback.current = callback; }, [callback]);
  useEffect(() => {
    if (delay !== null) {
      const id = setInterval(() => savedCallback.current(), delay);
      return () => clearInterval(id);
    }
  }, [delay]);
};

// ── Sub-components ───────────────────────────────────────────────────────────

const MetricCard = ({ icon: Icon, label, value, color = 'var(--neon)', sub }) => (
  <div className="card metric-card">
    <div className="metric-label">{label}</div>
    <div className="metric-value" style={{ color }}>
      <Icon size={15} style={{ marginRight: '6px', verticalAlign: 'middle' }} />
      {value}
    </div>
    {sub && <div style={{ fontSize: '0.6rem', color: 'var(--dim)', marginTop: '4px' }}>{sub}</div>}
  </div>
);

const CongestionBadge = ({ risk }) => {
  const colors = { HIGH: '#ff4444', MEDIUM: '#ffaa00', LOW: 'var(--neon)' };
  return (
    <span style={{
      fontSize: '0.55rem', padding: '1px 5px', borderRadius: '3px',
      border: `1px solid ${colors[risk] || '#888'}`,
      color: colors[risk] || '#888', fontFamily: 'Orbitron',
    }}>
      {risk || 'N/A'}
    </span>
  );
};

// ── Main App ─────────────────────────────────────────────────────────────────
const App = () => {
  const [data, setData] = useState(null);
  const [running, setRunning] = useState(true);
  const [fps, setFps] = useState(10);
  const [weights, setWeights] = useState({ w1: 0.5, w2: 0.3, w3: 0.2 });
  const [prevAmbulance, setPrevAmbulance] = useState(false);
  const canvasRef = useRef(null);

  const fetchData = async () => {
    try {
      const res = await fetch('/api/state');
      setData(await res.json());
    } catch (err) { console.error('Fetch error:', err); }
  };

  useInterval(fetchData, 1000 / fps);

  // Draw live video feed
  useEffect(() => {
    if (data?.yolo_frame_base64 && canvasRef.current) {
      const img = new Image();
      img.onload = () => {
        const ctx = canvasRef.current.getContext('2d');
        ctx.drawImage(img, 0, 0, canvasRef.current.width, canvasRef.current.height);
        ctx.fillStyle = 'rgba(57,255,20,0.015)';
        ctx.fillRect(0, ((data.step * 4) % canvasRef.current.height), canvasRef.current.width, 2);
      };
      img.src = 'data:image/jpeg;base64,' + data.yolo_frame_base64;
    }
  }, [data?.yolo_frame_base64]);

  // Track ambulance state change for alert
  useEffect(() => {
    if (data) setPrevAmbulance(data.ambulance_active);
  }, [data?.ambulance_active]);

  const toggleRunning = async () => {
    const next = !running;
    setRunning(next);
    await fetch('/api/control', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ running: next })
    });
  };

  const getHeatmapColor = (score) => {
    if (score < 1) return `rgba(57,255,20,${0.2 + score * 0.6})`;
    if (score < 3) return `rgba(255,170,0,${0.4 + (score - 1) * 0.3})`;
    return 'rgba(255,68,68,0.85)';
  };

  const getCongestionColor = (risk) =>
    ({ HIGH: '#ff4444', MEDIUM: '#ffaa00', LOW: 'var(--neon)' })[risk] || '#888';

  if (!data) return (
    <div className="app-container" style={{ justifyContent: 'center', alignItems: 'center' }}>
      <div style={{ fontFamily: 'Orbitron', fontSize: '1.3rem', color: 'var(--neon)' }}>
        <Zap /> INITIALIZING ECOSYNC...
      </div>
    </div>
  );

  const chartData = data.density.actual.map((val, i) => ({
    name: i, actual: val, predicted: data.density.predicted[i] || null
  }));

  // Feature 1: aggregate per-arm congestion
  const armRisk = ['N', 'S', 'E', 'W'].map(arm => {
    const laneRisks = [0, 1, 2].map(i => data.congestion_risk?.[`${arm}_in_${i}`] || 'LOW');
    const worst = laneRisks.includes('HIGH') ? 'HIGH' : laneRisks.includes('MEDIUM') ? 'MEDIUM' : 'LOW';
    return { arm, risk: worst };
  });

  // Feature 2: EV percentage
  const totalLive = (data.ev_count_live || 0) + (data.fuel_count_live || 0);
  const evPct = totalLive > 0 ? Math.round((data.ev_count_live / totalLive) * 100) : 0;

  return (
    <div className="app-container">

      {/* ── FEATURE 3: Ambulance Emergency Banner ─────────────────────── */}
      {data.ambulance_active && (
        <div className="emergency-banner">
          <Siren size={22} className="siren-icon" />
          <span>{data.ambulance_alert_msg || '🚨 AMBULANCE DETECTED — CLEAR THE CORRIDOR'}</span>
          <Siren size={22} className="siren-icon" />
        </div>
      )}

      {/* ── Sidebar ──────────────────────────────────────────────────────── */}
      <aside className="sidebar">
        <div style={{ textAlign: 'center', color: 'var(--neon)', fontSize: '0.65rem', marginBottom: '16px', letterSpacing: '2px' }}>
          🎯 BELLA CIAO PROTOCOL
        </div>
        <h2 style={{ fontSize: '1.1rem', color: 'var(--neon)', marginBottom: '2px' }}>⚡ ECOSYNC</h2>
        <p style={{ fontSize: '0.6rem', color: 'var(--dim)', marginBottom: '20px' }}>AI TRAFFIC COMMAND</p>

        <div style={{ borderTop: '1px solid var(--border)', margin: '12px 0' }} />

        <div className="input-group">
          <div className="label-row"><span>TARGET FPS</span><span className="text-neon">{fps}</span></div>
          <input type="range" min="1" max="30" value={fps} onChange={e => setFps(+e.target.value)} />
        </div>

        <button className={`btn ${running ? 'active' : ''}`} onClick={toggleRunning} style={{ width: '100%', marginBottom: '16px' }}>
          {running
            ? <><Pause size={13} style={{ marginRight: '6px' }} /> PAUSE FEED</>
            : <><Play size={13} style={{ marginRight: '6px' }} /> RESUME FEED</>}
        </button>

        <div style={{ borderTop: '1px solid var(--border)', margin: '12px 0' }} />
        <h3 style={{ fontSize: '0.65rem', color: 'var(--neon)', marginBottom: '12px' }}>REWARD WEIGHTS</h3>
        {['WaitTime', 'Emissions', 'JamRisk'].map((label, i) => (
          <div className="input-group" key={label}>
            <div className="label-row">
              <span>{label}</span>
              <span className="text-neon">{weights[`w${i + 1}`].toFixed(2)}</span>
            </div>
            <input type="range" min="0" max="1" step="0.05" value={weights[`w${i + 1}`]}
              onChange={e => setWeights({ ...weights, [`w${i + 1}`]: +e.target.value })} />
          </div>
        ))}

        {/* ── Feature 1: Direction Congestion ─────────────── */}
        <div style={{ borderTop: '1px solid var(--border)', margin: '12px 0' }} />
        <h3 style={{ fontSize: '0.65rem', color: 'var(--neon)', marginBottom: '10px' }}>SIGNAL CONGESTION</h3>
        {armRisk.map(({ arm, risk }) => (
          <div key={arm} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px', alignItems: 'center' }}>
            <span style={{ fontSize: '0.7rem' }}>
              {{ N: '↓ North', S: '↑ South', E: '← East', W: '→ West' }[arm]}
            </span>
            <CongestionBadge risk={risk} />
          </div>
        ))}
      </aside>

      {/* ── Main Content ─────────────────────────────────────────────────── */}
      <main className="main-content">
        <div style={{ textAlign: 'center', paddingBottom: '8px', borderBottom: '1px solid rgba(57,255,20,0.1)' }}>
          <h1 style={{ fontSize: '0.9rem', color: 'var(--neon)', letterSpacing: '3px' }}>
            ◆ ECOSYNC COMMAND CENTER — AI TRAFFIC ANALYSIS ◆
          </h1>
        </div>

        {/* Metrics Grid */}
        <div className="metrics-grid">
          <MetricCard icon={Activity} label="Sim Step" value={data.step.toLocaleString()} />
          <MetricCard icon={TrendingUp} label="Vehicles Passed" value={data.total_vehicles_passed?.toLocaleString() || '0'} sub="cumulative total" />
          <MetricCard icon={Cpu} label="Oracle Mode" value={data.oracle_mode}
            color={data.oracle_mode === 'WARMUP' ? '#ffaa00' : 'var(--neon)'} />
          <MetricCard icon={Wind} label="EV Live" value={data.ev_count_live || 0} color="#00ffff"
            sub={`${evPct}% of traffic`} />
          <MetricCard icon={Car} label="Fuel Vehicles" value={data.fuel_count_live || 0}
            color="#ff9944" sub={`CO₂: ${data.co2_rate_live || 0} g/km`} />
        </div>

        {/* ── Feature 2 + 3: CO2 + Ambulance Status Row ─────────────────── */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '16px' }}>

          {/* EV vs Fuel CO2 Panel */}
          <div className="card">
            <div className="card-title"><Gauge size={15} /> CO₂ EMISSIONS ANALYSIS</div>
            <div style={{ fontSize: '0.65rem', color: 'var(--dim)', marginBottom: '12px' }}>
              Based on HBEFA 4.1 Standards
            </div>
            <div style={{ marginBottom: '8px' }}>
              <div className="label-row" style={{ marginBottom: '4px' }}>
                <span style={{ fontSize: '0.7rem' }}>EV Traffic</span>
                <span style={{ color: '#00ffff', fontFamily: 'Orbitron', fontSize: '0.85rem' }}>{data.ev_count_live || 0}</span>
              </div>
              <div style={{ height: '6px', background: 'var(--border)', borderRadius: '3px', overflow: 'hidden' }}>
                <div style={{ width: `${evPct}%`, height: '100%', background: '#00ffff', borderRadius: '3px', transition: 'width 0.5s' }} />
              </div>
            </div>
            <div style={{ marginBottom: '8px' }}>
              <div className="label-row" style={{ marginBottom: '4px' }}>
                <span style={{ fontSize: '0.7rem' }}>Fuel Traffic</span>
                <span style={{ color: '#ff9944', fontFamily: 'Orbitron', fontSize: '0.85rem' }}>{data.fuel_count_live || 0}</span>
              </div>
              <div style={{ height: '6px', background: 'var(--border)', borderRadius: '3px', overflow: 'hidden' }}>
                <div style={{ width: `${100 - evPct}%`, height: '100%', background: '#ff9944', borderRadius: '3px', transition: 'width 0.5s' }} />
              </div>
            </div>
            <div style={{ borderTop: '1px solid var(--border)', paddingTop: '10px', marginTop: '8px' }}>
              <div className="label-row">
                <span style={{ fontSize: '0.65rem' }}>Live CO₂ Rate</span>
                <span style={{ color: '#ff4444', fontFamily: 'Orbitron', fontSize: '0.8rem' }}>{data.co2_rate_live || 0} g/km</span>
              </div>
              <div className="label-row" style={{ marginTop: '6px' }}>
                <span style={{ fontSize: '0.65rem' }}>CO₂ Saved by EVs</span>
                <span style={{ color: 'var(--neon)', fontFamily: 'Orbitron', fontSize: '0.8rem' }}>{(data.cumulative_co2_saved_ev || 0).toFixed(0)} g</span>
              </div>
            </div>
          </div>

          {/* ── Feature 1: Vehicle Throughput per Direction ─────────────────── */}
          <div className="card">
            <div className="card-title"><TrendingUp size={15} /> VEHICLE THROUGHPUT</div>
            <div style={{ fontSize: '0.65rem', color: 'var(--dim)', marginBottom: '12px' }}>Cumulative count by approach direction</div>
            {Object.entries(data.vehicle_throughput || {}).map(([arm, count]) => {
              const maxVal = Math.max(...Object.values(data.vehicle_throughput || { N: 1 }), 1);
              const pct = (count / maxVal) * 100;
              return (
                <div key={arm} style={{ marginBottom: '10px' }}>
                  <div className="label-row" style={{ marginBottom: '3px' }}>
                    <span style={{ fontSize: '0.7rem' }}>{{ N: '↓ North', S: '↑ South', E: '← East', W: '→ West' }[arm]}</span>
                    <span style={{ color: 'var(--neon)', fontFamily: 'Orbitron', fontSize: '0.8rem' }}>{count.toLocaleString()}</span>
                  </div>
                  <div style={{ height: '5px', background: 'var(--border)', borderRadius: '2px', overflow: 'hidden' }}>
                    <div style={{ width: `${pct}%`, height: '100%', background: 'var(--neon)', borderRadius: '2px', transition: 'width 0.6s', boxShadow: '0 0 6px var(--neon)' }} />
                  </div>
                </div>
              );
            })}
          </div>

          {/* ── Feature 3: Ambulance Status Panel ──────────────────────── */}
          <div className="card" style={{ border: data.ambulance_active ? '1px solid #ff4444' : '1px solid var(--border)', boxShadow: data.ambulance_active ? '0 0 20px rgba(255,68,68,0.4)' : undefined }}>
            <div className="card-title" style={{ color: data.ambulance_active ? '#ff4444' : 'var(--neon)' }}>
              <Siren size={15} /> EMERGENCY CORRIDOR
            </div>
            <div style={{ textAlign: 'center', padding: '16px 0' }}>
              {data.ambulance_active ? (
                <>
                  <div style={{ fontSize: '2.5rem', animation: 'pulse 0.8s infinite' }}>🚨</div>
                  <div style={{ color: '#ff4444', fontFamily: 'Orbitron', fontSize: '0.8rem', marginTop: '8px' }}>ACTIVE</div>
                  <div style={{ color: '#ff4444', fontSize: '0.65rem', marginTop: '8px', lineHeight: '1.4' }}>
                    Ambulance detected!<br />Notifying upstream vehicles<br />to clear the lane.
                  </div>
                </>
              ) : (
                <>
                  <div style={{ fontSize: '2.5rem', opacity: 0.4 }}>✅</div>
                  <div style={{ color: 'var(--neon)', fontFamily: 'Orbitron', fontSize: '0.8rem', marginTop: '8px' }}>CLEAR</div>
                  <div style={{ color: 'var(--dim)', fontSize: '0.65rem', marginTop: '8px' }}>
                    No emergency vehicles<br />detected in frame.
                  </div>
                </>
              )}
            </div>
          </div>
        </div>

        {/* ── Top Row: Live Feed + Impact ─────────────────────────────────── */}
        <div style={{ display: 'grid', gridTemplateColumns: '3fr 2fr', gap: '20px' }}>
          <div className="card">
            <div className="card-title">
              <span className="dot" /> THE LIVE PERCEPTION FEED
              {data.ambulance_active && (
                <span style={{ marginLeft: '12px', color: '#ff4444', fontSize: '0.65rem', fontFamily: 'Orbitron', animation: 'pulse 0.6s infinite' }}>
                  🚨 AMBULANCE
                </span>
              )}
            </div>
            <canvas ref={canvasRef} width={640} height={320} className="live-feed-canvas"
              style={{ border: data.ambulance_active ? '2px solid #ff4444' : undefined }} />
            <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--dim)', fontSize: '0.6rem', marginTop: '8px' }}>
              <span>Frame {data.step} | {data.oracle_mode}</span>
              <span>EV: {data.ev_count_live || 0} | Fuel: {data.fuel_count_live || 0} | CO₂: {data.co2_rate_live || 0} g/km</span>
            </div>
          </div>

          <div className="card">
            <div className="card-title"><BarChart3 size={15} /> IMPACT METER</div>
            <div style={{ textAlign: 'center', color: 'var(--dim)', fontSize: '0.6rem', letterSpacing: '1px' }}>
              TOTAL ENVIRONMENTAL IMPACT SAVED
            </div>
            <div className="impact-value">{data.impact_saved.toLocaleString()}</div>
            <div className="impact-stats">
              <div className="stat-box">
                <div className="stat-pct">{data.metrics.wait_pct.toFixed(1)}%</div>
                <div className="stat-abs">-{data.metrics.wait_abs.toFixed(0)}s</div>
                <div className="metric-label" style={{ marginTop: '2px' }}>Wait ↓</div>
              </div>
              <div className="stat-box">
                <div className="stat-pct">{data.metrics.emissions_pct.toFixed(1)}%</div>
                <div className="stat-abs">-{data.metrics.emissions_abs.toFixed(0)}</div>
                <div className="metric-label" style={{ marginTop: '2px' }}>CO₂ ↓</div>
              </div>
              <div className="stat-box">
                <div className="stat-pct">{data.metrics.jam_pct.toFixed(1)}%</div>
                <div className="stat-abs">-{data.metrics.jam_abs.toFixed(1)}</div>
                <div className="metric-label" style={{ marginTop: '2px' }}>Jam ↓</div>
              </div>
            </div>
          </div>
        </div>

        {/* ── Oracle Chart + Log ──────────────────────────────────────────── */}
        <div style={{ display: 'grid', gridTemplateColumns: '3fr 2fr', gap: '20px' }}>
          <div className="card">
            <div className="card-title"><Activity size={15} /> ORACLE'S EYE — LSTM Density Forecast</div>
            <div style={{ height: '260px' }}>
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={chartData}>
                  <defs>
                    <linearGradient id="gActual" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="var(--neon)" stopOpacity={0.12} />
                      <stop offset="95%" stopColor="var(--neon)" stopOpacity={0} />
                    </linearGradient>
                    <linearGradient id="gPred" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#ff4444" stopOpacity={0.1} />
                      <stop offset="95%" stopColor="#ff4444" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                  <XAxis dataKey="name" hide />
                  <YAxis stroke="#444" fontSize={9} />
                  <Tooltip contentStyle={{ background: '#12121a', border: '1px solid #1e1e2e', color: '#fff', fontSize: '10px' }} />
                  <Area type="monotone" dataKey="actual" stroke="var(--neon)" fill="url(#gActual)" strokeWidth={2} isAnimationActive={false} name="Actual Density" />
                  <Area type="monotone" dataKey="predicted" stroke="#ff4444" fill="url(#gPred)" strokeWidth={2} strokeDasharray="5 5" isAnimationActive={false} name="Predicted" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="card">
            <div className="card-title"><Layers size={15} /> STRATEGIST LOG</div>
            <div className="log-container">
              {data.log_entries.slice().reverse().map((e, i) => (
                <div key={i} className="log-entry">
                  <span className="log-time">[T={String(e.step).padStart(4, '0')}]</span>{' '}
                  <span className="log-action">RL: {e.action}</span> | {e.reason}
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* ── Feature 1: Lane Congestion Heatmap ─────────────────────────── */}
        <div className="card">
          <div className="card-title">
            <AlertOctagon size={15} /> LANE EMISSIONS + CONGESTION HEATMAP
          </div>
          <div className="heatmap-grid">
            <div className="hm-header" />
            <div className="hm-header">LANE 0</div>
            <div className="hm-header">LANE 1</div>
            <div className="hm-header">LANE 2</div>
            {['North', 'South', 'East', 'West'].map(arm => (
              <React.Fragment key={arm}>
                <div style={{ color: 'var(--dim)', fontSize: '0.65rem', display: 'flex', alignItems: 'center', justifyContent: 'flex-end', paddingRight: '10px' }}>
                  {arm}
                </div>
                {[0, 1, 2].map(lane => {
                  const lid = `${arm[0]}_in_${lane}`;
                  const score = data.lane_data[lid]?.emissions_score || 0;
                  const risk = data.congestion_risk?.[lid] || 'LOW';
                  return (
                    <div key={lane} className="hm-cell" style={{ background: getHeatmapColor(score), flexDirection: 'column', gap: '2px' }}>
                      <span style={{ fontSize: '0.7rem', fontFamily: 'Orbitron' }}>{score.toFixed(1)}</span>
                      <CongestionBadge risk={risk} />
                    </div>
                  );
                })}
              </React.Fragment>
            ))}
          </div>
          <div style={{ display: 'flex', justifyContent: 'center', gap: '20px', marginTop: '10px', fontSize: '0.6rem', color: 'var(--dim)' }}>
            <span>■ <span style={{ color: 'var(--neon)' }}>Low CO₂</span></span>
            <span>■ <span style={{ color: '#ffaa00' }}>Medium CO₂</span></span>
            <span>■ <span style={{ color: '#ff4444' }}>High CO₂</span></span>
            <span style={{ borderLeft: '1px solid var(--border)', paddingLeft: '16px' }}>Congestion: LOW / MEDIUM / HIGH</span>
          </div>
        </div>

      </main>
    </div>
  );
};

export default App;
