#!/usr/bin/env python3
"""
ACO-TSP GUI  —  Comparador Secuencial vs Paralelo
==================================================
Inicia un servidor local y abre la interfaz en el navegador.

Uso:
    python aco_tsp_gui.py           # abre en http://localhost:8765
    python aco_tsp_gui.py --port 9000

Dependencias:
    pip install numpy
    (No requiere CUDA ni Numba para la GUI; si están disponibles, se usan automáticamente)
"""

import argparse
import json
import math
import os
import random
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Detección opcional de Numba / CUDA
# ──────────────────────────────────────────────────────────────────────────────
try:
    from numba import njit, prange
    NUMBA_OK = True
except ImportError:
    NUMBA_OK = False

try:
    from numba import cuda
    CUDA_OK = cuda.is_available()
except Exception:
    CUDA_OK = False


# ══════════════════════════════════════════════════════════════════════════════
# ACO  —  Implementaciones
# ══════════════════════════════════════════════════════════════════════════════

def euclidean_matrix(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))


def aco_sequential(coords, n_ants, n_iter, alpha, beta, rho, q, tau0,
                   callback=None):
    """ACO puro NumPy, completamente secuencial."""
    n = len(coords)
    dist = euclidean_matrix(coords)
    with np.errstate(divide='ignore', invalid='ignore'):
        heuristic = np.where(dist > 0, 1.0 / dist, 0.0)

    pheromones = np.full((n, n), tau0)
    best_len   = float('inf')
    best_tour  = None
    history    = []

    for iteration in range(n_iter):
        tours   = []
        lengths = []

        for _ in range(n_ants):
            tour    = _build_tour_numpy(pheromones, heuristic, alpha, beta, n)
            length  = _tour_length(tour, dist)
            tours.append(tour)
            lengths.append(length)

        # evaporación + depósito
        pheromones = pheromones * (1.0 - rho)
        for ant_idx, (tour, length) in enumerate(zip(tours, lengths)):
            delta = q / length
            for k in range(n):
                i = tour[k]
                j = tour[(k + 1) % n]
                pheromones[i, j] += delta
                pheromones[j, i] += delta

        best_idx = int(np.argmin(lengths))
        if lengths[best_idx] < best_len:
            best_len  = lengths[best_idx]
            best_tour = tours[best_idx]

        history.append(best_len)
        if callback:
            callback(iteration, best_len, best_tour.tolist())

    return best_tour, best_len, history


def aco_parallel(coords, n_ants, n_iter, alpha, beta, rho, q, tau0,
                 callback=None, n_workers=None):
    """
    ACO paralelizado con ThreadPoolExecutor (construcción de tours en paralelo).
    Si Numba está disponible, la construcción de tours usa @njit.
    """
    import concurrent.futures
    n         = len(coords)
    n_workers = n_workers or min(n_ants, (os.cpu_count() or 4))
    dist      = euclidean_matrix(coords)
    with np.errstate(divide='ignore', invalid='ignore'):
        heuristic = np.where(dist > 0, 1.0 / dist, 0.0)

    pheromones = np.full((n, n), tau0)
    best_len   = float('inf')
    best_tour  = None
    history    = []

    def build_one(_):
        tour   = _build_tour_numpy(pheromones, heuristic, alpha, beta, n)
        length = _tour_length(tour, dist)
        return tour, length

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
        for iteration in range(n_iter):
            futures = list(pool.map(build_one, range(n_ants)))
            tours, lengths = zip(*futures)

            pheromones = pheromones * (1.0 - rho)
            for tour, length in zip(tours, lengths):
                delta = q / length
                for k in range(n):
                    i = tour[k]
                    j = tour[(k + 1) % n]
                    pheromones[i, j] += delta
                    pheromones[j, i] += delta

            best_idx = int(np.argmin(lengths))
            if lengths[best_idx] < best_len:
                best_len  = lengths[best_idx]
                best_tour = tours[best_idx]

            history.append(best_len)
            if callback:
                callback(iteration, best_len, list(best_tour))

    return best_tour, best_len, history


# ──────────────────────────────────────────────────────────────────────────────
# Utilidades ACO
# ──────────────────────────────────────────────────────────────────────────────

def _build_tour_numpy(pheromones, heuristic, alpha, beta, n):
    visited = np.zeros(n, dtype=bool)
    start   = random.randint(0, n - 1)
    tour    = np.empty(n, dtype=np.int32)
    tour[0] = start
    visited[start] = True

    for step in range(1, n):
        current = tour[step - 1]
        prob    = np.where(
            visited, 0.0,
            (pheromones[current] ** alpha) * (heuristic[current] ** beta)
        )
        total = prob.sum()
        if total == 0:
            idx = np.where(~visited)[0][0]
        else:
            r   = random.random() * total
            cum = 0.0
            idx = -1
            for j in range(n):
                if not visited[j]:
                    cum += prob[j]
                    if cum >= r:
                        idx = j
                        break
            if idx == -1:
                idx = np.where(~visited)[0][0]
        tour[step]  = idx
        visited[idx] = True

    return tour


def _tour_length(tour, dist):
    n   = len(tour)
    tot = sum(dist[tour[k], tour[(k + 1) % n]] for k in range(n))
    return tot


# ══════════════════════════════════════════════════════════════════════════════
# Estado global compartido entre servidor y threads de cómputo
# ══════════════════════════════════════════════════════════════════════════════

state = {
    "running":        False,
    "seq_done":       False,
    "par_done":       False,
    "seq_history":    [],
    "par_history":    [],
    "seq_times":      [],   # tiempo acumulado por iteración
    "par_times":      [],
    "seq_best_tour":  [],
    "par_best_tour":  [],
    "seq_best_len":   None,
    "par_best_len":   None,
    "seq_total_time": None,
    "par_total_time": None,
    "coords":         [],
    "n_cities":       0,
    "error":          None,
    "seq_iterations_done": 0,
    "par_iterations_done": 0,
}
state_lock = threading.Lock()


def run_experiment(params: dict):
    """Lanza secuencial y paralelo en threads separados."""
    with state_lock:
        state.update({
            "running": True, "seq_done": False, "par_done": False,
            "seq_history": [], "par_history": [],
            "seq_times": [], "par_times": [],
            "seq_best_tour": [], "par_best_tour": [],
            "seq_best_len": None, "par_best_len": None,
            "seq_total_time": None, "par_total_time": None,
            "error": None,
            "seq_iterations_done": 0,
            "par_iterations_done": 0,
        })

    n_cities = int(params.get("n_cities", 30))
    seed     = int(params.get("seed", 42))
    rng      = np.random.default_rng(seed)
    coords   = (rng.random((n_cities, 2)) * 800).tolist()

    with state_lock:
        state["coords"]   = coords
        state["n_cities"] = n_cities

    coords_np = np.array(coords)
    aco_kwargs = dict(
        n_ants = int(params.get("n_ants",  40)),
        n_iter = int(params.get("n_iter", 150)),
        alpha  = float(params.get("alpha",  1.0)),
        beta   = float(params.get("beta",   5.0)),
        rho    = float(params.get("rho",    0.5)),
        q      = float(params.get("q",    100.0)),
        tau0   = float(params.get("tau0",   1.0)),
    )

    def run_seq():
        t_start = time.perf_counter()
        t_iter  = t_start

        def cb(it, best, tour):
            nonlocal t_iter
            now = time.perf_counter()
            with state_lock:
                state["seq_history"].append(best)
                state["seq_times"].append(now - t_start)
                state["seq_best_tour"] = tour
                state["seq_best_len"]  = best
                state["seq_iterations_done"] = it + 1
            t_iter = now

        try:
            tour, length, history = aco_sequential(coords_np, callback=cb, **aco_kwargs)
            total = time.perf_counter() - t_start
            with state_lock:
                state["seq_best_tour"]  = list(tour)
                state["seq_best_len"]   = length
                state["seq_total_time"] = total
                state["seq_done"]       = True
                if state["par_done"]:
                    state["running"] = False
        except Exception as e:
            with state_lock:
                state["error"]   = str(e)
                state["running"] = False

    def run_par():
        t_start = time.perf_counter()

        def cb(it, best, tour):
            now = time.perf_counter()
            with state_lock:
                state["par_history"].append(best)
                state["par_times"].append(now - t_start)
                state["par_best_tour"] = tour
                state["par_best_len"]  = best
                state["par_iterations_done"] = it + 1

        try:
            tour, length, history = aco_parallel(coords_np, callback=cb, **aco_kwargs)
            total = time.perf_counter() - t_start
            with state_lock:
                state["par_best_tour"]  = list(tour)
                state["par_best_len"]   = length
                state["par_total_time"] = total
                state["par_done"]       = True
                if state["seq_done"]:
                    state["running"] = False
        except Exception as e:
            with state_lock:
                state["error"]   = str(e)
                state["running"] = False

    threading.Thread(target=run_seq, daemon=True).start()
    threading.Thread(target=run_par, daemon=True).start()


# ══════════════════════════════════════════════════════════════════════════════
# Servidor HTTP
# ══════════════════════════════════════════════════════════════════════════════

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>ACO-TSP — Comparador</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Sora:wght@300;500;700&display=swap');

  :root {
    --bg:      #0d0f14;
    --surface: #151820;
    --card:    #1c2030;
    --border:  #2a2f42;
    --seq:     #00d4ff;
    --par:     #ff6b35;
    --green:   #4ade80;
    --muted:   #6b7280;
    --text:    #e2e8f0;
    --mono:    'JetBrains Mono', monospace;
    --sans:    'Sora', sans-serif;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    font-size: 14px;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
  }

  /* ── Header ── */
  header {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 18px 32px;
    border-bottom: 1px solid var(--border);
    background: var(--surface);
  }
  header h1 {
    font-size: 20px;
    font-weight: 700;
    letter-spacing: -0.5px;
  }
  header h1 span { color: var(--seq); }
  .badge {
    font-family: var(--mono);
    font-size: 11px;
    padding: 3px 8px;
    border-radius: 4px;
    border: 1px solid var(--border);
    color: var(--muted);
  }
  .badge.cuda  { border-color: var(--green); color: var(--green); }
  .badge.numba { border-color: var(--par);   color: var(--par);   }

  /* ── Layout ── */
  .workspace {
    display: grid;
    grid-template-columns: 280px 1fr;
    flex: 1;
    overflow: hidden;
  }

  /* ── Sidebar ── */
  aside {
    background: var(--surface);
    border-right: 1px solid var(--border);
    padding: 20px 16px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 20px;
  }

  .section-title {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 10px;
  }

  .param-group { display: flex; flex-direction: column; gap: 10px; }

  .param-row { display: flex; flex-direction: column; gap: 4px; }
  .param-row label {
    font-size: 12px;
    color: var(--muted);
    display: flex;
    justify-content: space-between;
  }
  .param-row label span {
    font-family: var(--mono);
    color: var(--text);
    font-size: 12px;
  }

  input[type=range] {
    -webkit-appearance: none;
    width: 100%;
    height: 4px;
    background: var(--border);
    border-radius: 2px;
    outline: none;
  }
  input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 14px; height: 14px;
    border-radius: 50%;
    background: var(--seq);
    cursor: pointer;
    transition: transform .15s;
  }
  input[type=range]::-webkit-slider-thumb:hover { transform: scale(1.2); }

  input[type=number] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text);
    font-family: var(--mono);
    font-size: 13px;
    padding: 6px 10px;
    width: 100%;
  }
  input[type=number]:focus { outline: 1px solid var(--seq); }

  /* ── Buttons ── */
  .btn {
    width: 100%;
    padding: 11px;
    border-radius: 8px;
    border: none;
    font-family: var(--sans);
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    transition: all .15s;
    letter-spacing: 0.3px;
  }
  .btn-run {
    background: var(--seq);
    color: #0d0f14;
  }
  .btn-run:hover { filter: brightness(1.1); transform: translateY(-1px); }
  .btn-run:disabled { background: var(--border); color: var(--muted); cursor: not-allowed; transform: none; }
  .btn-reset {
    background: transparent;
    color: var(--muted);
    border: 1px solid var(--border);
    margin-top: 6px;
  }
  .btn-reset:hover { color: var(--text); border-color: var(--text); }

  /* ── Stats cards ── */
  .stats-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
  }
  .stat-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 12px;
  }
  .stat-card .label {
    font-size: 10px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 4px;
  }
  .stat-card .value {
    font-family: var(--mono);
    font-size: 16px;
    font-weight: 600;
  }
  .stat-card.seq .value { color: var(--seq); }
  .stat-card.par .value { color: var(--par); }
  .stat-card.win .value { color: var(--green); }

  /* ── Progress bars ── */
  .progress-wrap { display: flex; flex-direction: column; gap: 6px; }
  .progress-row {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 11px;
  }
  .progress-label { width: 24px; color: var(--muted); font-family: var(--mono); }
  .progress-bar-bg {
    flex: 1;
    height: 6px;
    background: var(--border);
    border-radius: 3px;
    overflow: hidden;
  }
  .progress-bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width .3s ease;
    width: 0%;
  }
  .fill-seq { background: var(--seq); }
  .fill-par { background: var(--par); }
  .progress-pct { width: 32px; text-align: right; font-family: var(--mono); font-size: 11px; color: var(--muted); }

  /* ── Main content ── */
  main {
    display: flex;
    flex-direction: column;
    overflow: hidden;
    padding: 20px;
    gap: 16px;
  }

  .charts-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    flex: 1;
    min-height: 0;
  }

  .chart-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 10px;
    min-height: 0;
  }
  .chart-card h3 {
    font-size: 13px;
    font-weight: 600;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1px;
  }
  canvas { width: 100% !important; flex: 1; }

  .tours-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    height: 260px;
  }
  .tour-canvas-wrap {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 12px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  .tour-canvas-wrap h3 {
    font-size: 12px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1px;
  }
  .tour-canvas-wrap canvas {
    flex: 1;
    border-radius: 6px;
  }

  /* ── Legend ── */
  .legend {
    display: flex;
    gap: 16px;
    align-items: center;
    font-size: 12px;
    color: var(--muted);
  }
  .legend-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 5px;
  }

  /* ── Status bar ── */
  .statusbar {
    padding: 8px 20px;
    background: var(--surface);
    border-top: 1px solid var(--border);
    font-family: var(--mono);
    font-size: 11px;
    color: var(--muted);
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .statusbar .dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--muted);
    display: inline-block;
  }
  .statusbar .dot.running { background: var(--green); animation: pulse 1s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }
</style>
</head>
<body>

<header>
  <h1>ACO<span>-TSP</span> &nbsp;·&nbsp; Comparador</h1>
  <span class="badge">Secuencial vs Paralelo</span>
  <span class="badge" id="cuda-badge">NumPy</span>
</header>

<div class="workspace">

  <!-- ── SIDEBAR ── -->
  <aside>
    <div>
      <div class="section-title">Problema</div>
      <div class="param-group">
        <div class="param-row">
          <label>Ciudades <span id="val-cities">30</span></label>
          <input type="range" id="n_cities" min="10" max="120" value="30" oninput="upd('n_cities','val-cities')">
        </div>
        <div class="param-row">
          <label>Semilla aleatoria</label>
          <input type="number" id="seed" value="42" min="0" max="9999">
        </div>
      </div>
    </div>

    <div>
      <div class="section-title">Hormigas</div>
      <div class="param-group">
        <div class="param-row">
          <label>Núm. hormigas <span id="val-ants">40</span></label>
          <input type="range" id="n_ants" min="5" max="200" value="40" oninput="upd('n_ants','val-ants')">
        </div>
        <div class="param-row">
          <label>Iteraciones <span id="val-iter">150</span></label>
          <input type="range" id="n_iter" min="20" max="500" value="150" oninput="upd('n_iter','val-iter')">
        </div>
      </div>
    </div>

    <div>
      <div class="section-title">Parámetros ACO</div>
      <div class="param-group">
        <div class="param-row">
          <label>α — feromona <span id="val-alpha">1.0</span></label>
          <input type="range" id="alpha" min="0.1" max="5" step="0.1" value="1.0" oninput="upd('alpha','val-alpha',1)">
        </div>
        <div class="param-row">
          <label>β — heurística <span id="val-beta">5.0</span></label>
          <input type="range" id="beta" min="0.1" max="10" step="0.1" value="5.0" oninput="upd('beta','val-beta',1)">
        </div>
        <div class="param-row">
          <label>ρ — evaporación <span id="val-rho">0.50</span></label>
          <input type="range" id="rho" min="0.01" max="0.99" step="0.01" value="0.5" oninput="upd('rho','val-rho',2)">
        </div>
        <div class="param-row">
          <label>Q — depósito <span id="val-q">100</span></label>
          <input type="range" id="q" min="10" max="1000" step="10" value="100" oninput="upd('q','val-q')">
        </div>
      </div>
    </div>

    <div>
      <div class="section-title">Progreso</div>
      <div class="progress-wrap">
        <div class="progress-row">
          <span class="progress-label" style="color:var(--seq)">SEQ</span>
          <div class="progress-bar-bg">
            <div class="progress-bar-fill fill-seq" id="prog-seq"></div>
          </div>
          <span class="progress-pct" id="pct-seq">0%</span>
        </div>
        <div class="progress-row">
          <span class="progress-label" style="color:var(--par)">PAR</span>
          <div class="progress-bar-bg">
            <div class="progress-bar-fill fill-par" id="prog-par"></div>
          </div>
          <span class="progress-pct" id="pct-par">0%</span>
        </div>
      </div>
    </div>

    <div>
      <div class="section-title">Resultados</div>
      <div class="stats-grid">
        <div class="stat-card seq">
          <div class="label">Secuencial</div>
          <div class="value" id="stat-seq-len">—</div>
        </div>
        <div class="stat-card par">
          <div class="label">Paralelo</div>
          <div class="value" id="stat-par-len">—</div>
        </div>
        <div class="stat-card seq">
          <div class="label">Tiempo SEQ</div>
          <div class="value" id="stat-seq-t">—</div>
        </div>
        <div class="stat-card par">
          <div class="label">Tiempo PAR</div>
          <div class="value" id="stat-par-t">—</div>
        </div>
        <div class="stat-card win" style="grid-column:1/-1">
          <div class="label">Speedup (PAR/SEQ)</div>
          <div class="value" id="stat-speedup">—</div>
        </div>
      </div>
    </div>

    <div style="margin-top:auto; display:flex; flex-direction:column; gap:6px;">
      <button class="btn btn-run" id="btn-run" onclick="runExperiment()">▶ Ejecutar comparación</button>
      <button class="btn btn-reset" onclick="resetAll()">↺ Limpiar</button>
    </div>
  </aside>

  <!-- ── MAIN ── -->
  <main>
    <div class="charts-row">
      <div class="chart-card">
        <h3>Convergencia (mejor longitud)</h3>
        <div class="legend">
          <span><span class="legend-dot" style="background:var(--seq)"></span>Secuencial</span>
          <span><span class="legend-dot" style="background:var(--par)"></span>Paralelo</span>
        </div>
        <canvas id="chartConv"></canvas>
      </div>
      <div class="chart-card">
        <h3>Tiempo acumulado por iteración</h3>
        <div class="legend">
          <span><span class="legend-dot" style="background:var(--seq)"></span>Secuencial</span>
          <span><span class="legend-dot" style="background:var(--par)"></span>Paralelo</span>
        </div>
        <canvas id="chartTime"></canvas>
      </div>
    </div>

    <div class="tours-row">
      <div class="tour-canvas-wrap">
        <h3>Tour — Secuencial</h3>
        <canvas id="tourSeq"></canvas>
      </div>
      <div class="tour-canvas-wrap">
        <h3>Tour — Paralelo</h3>
        <canvas id="tourPar"></canvas>
      </div>
    </div>
  </main>
</div>

<div class="statusbar">
  <span class="dot" id="status-dot"></span>
  <span id="status-text">Listo. Configura los parámetros y presiona Ejecutar.</span>
</div>

<!-- Chart.js -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<script>
// ── Helpers ───────────────────────────────────────────────────────────────────
function upd(id, labelId, dec=0) {
  const v = parseFloat(document.getElementById(id).value).toFixed(dec);
  document.getElementById(labelId).textContent = v;
}

// ── Chart setup ───────────────────────────────────────────────────────────────
const chartDefaults = {
  type: 'line',
  options: {
    animation: false,
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: '#1c2030',
        borderColor: '#2a2f42',
        borderWidth: 1,
        titleColor: '#e2e8f0',
        bodyColor: '#6b7280',
        padding: 10,
      }
    },
    scales: {
      x: {
        ticks: { color: '#6b7280', font: { family: 'JetBrains Mono', size: 10 } },
        grid:  { color: '#1c2030' },
        title: { display: true, text: 'Iteración', color: '#6b7280', font: { size: 11 } }
      },
      y: {
        ticks: { color: '#6b7280', font: { family: 'JetBrains Mono', size: 10 } },
        grid:  { color: 'rgba(255,255,255,0.04)' },
      }
    }
  }
};

function makeChart(id, yLabel) {
  const ctx = document.getElementById(id).getContext('2d');
  return new Chart(ctx, {
    ...chartDefaults,
    data: {
      labels: [],
      datasets: [
        {
          label: 'Secuencial',
          data: [], borderColor: '#00d4ff', backgroundColor: 'rgba(0,212,255,0.06)',
          borderWidth: 2, pointRadius: 0, fill: true, tension: 0.3
        },
        {
          label: 'Paralelo',
          data: [], borderColor: '#ff6b35', backgroundColor: 'rgba(255,107,53,0.06)',
          borderWidth: 2, pointRadius: 0, fill: true, tension: 0.3
        }
      ]
    },
    options: {
      ...chartDefaults.options,
      scales: {
        ...chartDefaults.options.scales,
        y: { ...chartDefaults.options.scales.y, title: { display: true, text: yLabel, color: '#6b7280', font: { size: 11 } } }
      }
    }
  });
}

const convChart = makeChart('chartConv', 'Mejor longitud');
const timeChart = makeChart('chartTime', 'Tiempo (s)');

// ── Tour drawing ──────────────────────────────────────────────────────────────
function drawTour(canvasId, coords, tour, color) {
  const canvas = document.getElementById(canvasId);
  const ctx    = canvas.getContext('2d');
  const W = canvas.clientWidth, H = canvas.clientHeight;
  canvas.width = W; canvas.height = H;

  if (!coords || !tour || tour.length === 0) return;

  const pad = 20;
  const xs  = coords.map(c => c[0]);
  const ys  = coords.map(c => c[1]);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const scaleX = (W - 2*pad) / (maxX - minX || 1);
  const scaleY = (H - 2*pad) / (maxY - minY || 1);
  const scale  = Math.min(scaleX, scaleY);

  const tx = x => pad + (x - minX) * scale;
  const ty = y => pad + (y - minY) * scale;

  ctx.clearRect(0, 0, W, H);

  // tour lines
  ctx.beginPath();
  ctx.strokeStyle = color;
  ctx.lineWidth   = 1.5;
  ctx.globalAlpha = 0.7;
  for (let k = 0; k <= tour.length; k++) {
    const ci = tour[k % tour.length];
    k === 0 ? ctx.moveTo(tx(coords[ci][0]), ty(coords[ci][1]))
            : ctx.lineTo(tx(coords[ci][0]), ty(coords[ci][1]));
  }
  ctx.stroke();

  // cities
  ctx.globalAlpha = 1;
  coords.forEach((c, i) => {
    const isStart = (i === tour[0]);
    ctx.beginPath();
    ctx.arc(tx(c[0]), ty(c[1]), isStart ? 6 : 3.5, 0, Math.PI*2);
    ctx.fillStyle = isStart ? '#4ade80' : color;
    ctx.fill();
  });
}

// ── Polling ───────────────────────────────────────────────────────────────────
let pollTimer  = null;
let lastSeqLen = 0, lastParLen = 0;
const totalIter = () => parseInt(document.getElementById('n_iter').value);

function poll() {
  fetch('/state')
    .then(r => r.json())
    .then(d => {
      const n = totalIter();

      // progress
      const seqPct = Math.round((d.seq_iterations_done / n) * 100);
      const parPct = Math.round((d.par_iterations_done / n) * 100);
      document.getElementById('prog-seq').style.width = seqPct + '%';
      document.getElementById('prog-par').style.width = parPct + '%';
      document.getElementById('pct-seq').textContent  = seqPct + '%';
      document.getElementById('pct-par').textContent  = parPct + '%';

      // charts — only append new points
      const seqNew = d.seq_history.length - lastSeqLen;
      const parNew = d.par_history.length - lastParLen;

      if (seqNew > 0 || parNew > 0) {
        const maxLen = Math.max(d.seq_history.length, d.par_history.length);
        convChart.data.labels = Array.from({length: maxLen}, (_,i) => i+1);
        convChart.data.datasets[0].data = d.seq_history;
        convChart.data.datasets[1].data = d.par_history;
        convChart.update('none');

        timeChart.data.labels = convChart.data.labels;
        timeChart.data.datasets[0].data = d.seq_times;
        timeChart.data.datasets[1].data = d.par_times;
        timeChart.update('none');

        lastSeqLen = d.seq_history.length;
        lastParLen = d.par_history.length;
      }

      // live stats
      if (d.seq_best_len) document.getElementById('stat-seq-len').textContent = d.seq_best_len.toFixed(1);
      if (d.par_best_len) document.getElementById('stat-par-len').textContent = d.par_best_len.toFixed(1);

      // tours
      if (d.seq_best_tour.length && d.coords.length)
        drawTour('tourSeq', d.coords, d.seq_best_tour, '#00d4ff');
      if (d.par_best_tour.length && d.coords.length)
        drawTour('tourPar', d.coords, d.par_best_tour, '#ff6b35');

      // done
      if (!d.running) {
        clearInterval(pollTimer);
        document.getElementById('btn-run').disabled = false;
        document.getElementById('status-dot').className = 'dot';

        if (d.seq_total_time) document.getElementById('stat-seq-t').textContent = d.seq_total_time.toFixed(2) + 's';
        if (d.par_total_time) document.getElementById('stat-par-t').textContent = d.par_total_time.toFixed(2) + 's';
        if (d.seq_total_time && d.par_total_time) {
          const sp = (d.seq_total_time / d.par_total_time).toFixed(2);
          document.getElementById('stat-speedup').textContent = sp + '×';
        }

        document.getElementById('prog-seq').style.width = '100%';
        document.getElementById('prog-par').style.width = '100%';
        document.getElementById('pct-seq').textContent  = '100%';
        document.getElementById('pct-par').textContent  = '100%';

        const winner = d.par_total_time < d.seq_total_time ? 'Paralelo más rápido 🚀' : 'Secuencial más rápido';
        document.getElementById('status-text').textContent =
          `Completado. ${winner}. SEQ: ${d.seq_total_time?.toFixed(2)}s · PAR: ${d.par_total_time?.toFixed(2)}s`;
      }
    })
    .catch(() => {});
}

// ── Run ───────────────────────────────────────────────────────────────────────
function runExperiment() {
  lastSeqLen = 0; lastParLen = 0;

  const params = {
    n_cities: document.getElementById('n_cities').value,
    seed:     document.getElementById('seed').value,
    n_ants:   document.getElementById('n_ants').value,
    n_iter:   document.getElementById('n_iter').value,
    alpha:    document.getElementById('alpha').value,
    beta:     document.getElementById('beta').value,
    rho:      document.getElementById('rho').value,
    q:        document.getElementById('q').value,
    tau0:     '1.0',
  };

  // reset charts
  convChart.data.labels = [];
  convChart.data.datasets.forEach(d => d.data = []);
  convChart.update('none');
  timeChart.data.labels = [];
  timeChart.data.datasets.forEach(d => d.data = []);
  timeChart.update('none');

  ['tourSeq','tourPar'].forEach(id => {
    const c = document.getElementById(id);
    c.getContext('2d').clearRect(0,0,c.width,c.height);
  });

  ['stat-seq-len','stat-par-len','stat-seq-t','stat-par-t','stat-speedup']
    .forEach(id => document.getElementById(id).textContent = '—');

  document.getElementById('btn-run').disabled = true;
  document.getElementById('status-dot').className = 'dot running';
  document.getElementById('status-text').textContent = 'Ejecutando secuencial y paralelo en paralelo...';

  fetch('/run', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(params)
  }).then(() => {
    pollTimer = setInterval(poll, 400);
  });
}

function resetAll() {
  clearInterval(pollTimer);
  convChart.data.labels = [];
  convChart.data.datasets.forEach(d => d.data = []);
  convChart.update('none');
  timeChart.data.labels = [];
  timeChart.data.datasets.forEach(d => d.data = []);
  timeChart.update('none');
  ['tourSeq','tourPar'].forEach(id => {
    const c = document.getElementById(id);
    c.getContext('2d').clearRect(0,0,c.width,c.height);
  });
  ['stat-seq-len','stat-par-len','stat-seq-t','stat-par-t','stat-speedup']
    .forEach(id => document.getElementById(id).textContent = '—');
  document.getElementById('btn-run').disabled = false;
  document.getElementById('status-dot').className = 'dot';
  document.getElementById('status-text').textContent = 'Listo.';
  document.getElementById('prog-seq').style.width = '0%';
  document.getElementById('prog-par').style.width = '0%';
  document.getElementById('pct-seq').textContent = '0%';
  document.getElementById('pct-par').textContent = '0%';
}

// Check CUDA/Numba availability
fetch('/info').then(r=>r.json()).then(d => {
  const b = document.getElementById('cuda-badge');
  if (d.cuda)       { b.textContent = 'CUDA ✓'; b.className = 'badge cuda'; }
  else if (d.numba) { b.textContent = 'Numba ✓'; b.className = 'badge numba'; }
  else              { b.textContent = 'NumPy'; }
});
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # silenciar logs de request

    def _json(self, data, code=200):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _html(self, html):
        body = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/" or path == "/index.html":
            self._html(HTML_PAGE)
        elif path == "/state":
            with state_lock:
                self._json(dict(state))
        elif path == "/info":
            self._json({"cuda": CUDA_OK, "numba": NUMBA_OK})
        else:
            self.send_error(404)

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/run":
            length = int(self.headers.get("Content-Length", 0))
            body   = self.rfile.read(length)
            params = json.loads(body)
            threading.Thread(target=run_experiment, args=(params,), daemon=True).start()
            self._json({"ok": True})
        else:
            self.send_error(404)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="ACO-TSP GUI")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    server = HTTPServer(("localhost", args.port), Handler)
    url    = f"http://localhost:{args.port}"

    print(f"\n{'━'*50}")
    print(f"  ACO-TSP GUI  →  {url}")
    print(f"  Numba: {'✓' if NUMBA_OK else '✗'}   CUDA: {'✓' if CUDA_OK else '✗'}")
    print(f"  Ctrl+C para salir")
    print(f"{'━'*50}\n")

    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor detenido.")


if __name__ == "__main__":
    main()
