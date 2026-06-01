# TSP — Ant Colony Optimization (CPU / CUDA)

Solves the **Traveling Salesman Problem** using Ant Colony Optimization. Two execution modes are available: a pure-CPU implementation and a CUDA-parallelized implementation where each ant builds its tour concurrently on the GPU.

---

## Files

| File | Description |
|---|---|
| `tsp_aco.py` | Core solver (CPU and CUDA modes) |
| `tsp_ui.py` | Tkinter graphical interface |
| `cities.txt` | Default city coordinates (50 cities) |
| `run_cpu.cmd` | Run solver in CPU mode |
| `run_cuda.cmd` | Run solver in CUDA mode |
| `run_ui.cmd` | Launch the graphical interface |

---

## Parameters

The following parameters were used for the benchmark runs. All are configurable via command-line arguments or through the UI.

| Parameter | Value | Description |
|---|---|---|
| **Cities** | 50 | Number of cities in the problem instance |
| **Ants** | 50 | Number of ants per iteration |
| **Iterations** | 200 | Number of optimization iterations |
| **Alpha** | 1.0 | Pheromone exponent — controls how strongly ants follow existing trails |
| **Beta** | 3.0 | Heuristic exponent — controls preference for shorter edges |
| **Rho** | 0.1 | Evaporation rate — fraction of pheromone that disappears each iteration |
| **Q** | 100.0 | Pheromone deposit constant — amount deposited proportional to tour quality |
| **Initial pheromone** | 1.0 | Starting pheromone level on all edges |

---

## City Coordinates (`cities.txt`)

50 cities used as the default problem instance.

| # | X | Y | | # | X | Y | | # | X | Y | | # | X | Y | | # | X | Y |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 565 | 575 | | 10 | 1605 | 620 | | 20 | 300 | 465 | | 30 | 420 | 555 | | 40 | 475 | 960 |
| 1 | 25 | 185 | | 11 | 1220 | 580 | | 21 | 520 | 585 | | 31 | 575 | 665 | | 41 | 95 | 260 |
| 2 | 345 | 750 | | 12 | 1465 | 200 | | 22 | 480 | 415 | | 32 | 1150 | 1160 | | 42 | 875 | 920 |
| 3 | 945 | 685 | | 13 | 1530 | 5 | | 23 | 835 | 625 | | 33 | 700 | 580 | | 43 | 700 | 500 |
| 4 | 845 | 655 | | 14 | 845 | 680 | | 24 | 975 | 580 | | 34 | 685 | 595 | | 44 | 555 | 815 |
| 5 | 880 | 660 | | 15 | 725 | 370 | | 25 | 1215 | 245 | | 35 | 685 | 610 | | 45 | 830 | 485 |
| 6 | 25 | 230 | | 16 | 145 | 665 | | 26 | 1320 | 315 | | 36 | 770 | 610 | | 46 | 1170 | 65 |
| 7 | 525 | 1000 | | 17 | 415 | 635 | | 27 | 1250 | 400 | | 37 | 795 | 645 | | 47 | 830 | 610 |
| 8 | 580 | 1175 | | 18 | 510 | 875 | | 28 | 660 | 180 | | 38 | 720 | 635 | | 48 | 605 | 625 |
| 9 | 650 | 1130 | | 19 | 560 | 365 | | 29 | 410 | 250 | | 39 | 760 | 650 | | 49 | 595 | 360 |

---

## Benchmark Results

Both machines ran the same problem instance (50 cities, parameters above) five times each, in CPU and CUDA mode. Times shown are wall-clock seconds for the full run.

### Machine 1 — AMD Ryzen 7 5700G + GTX 1660 Super

| Component | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | **Avg** |
|---|---|---|---|---|---|---|
| AMD Ryzen 7 5700G · 3801 MHz · 8 cores / 16 threads (CPU) | 10.61 s | 10.92 s | 11.27 s | 10.21 s | 10.40 s | **10.68 s** |
| NVIDIA GeForce GTX 1660 Super · 1280 CUDA cores (CUDA) | 1.32 s | 1.31 s | 1.33 s | 1.34 s | 1.32 s | **1.32 s** |

> **Speedup: ~8.1×** faster with CUDA on this machine.

### Machine 2 — Intel Core i7-13700KF + GTX 1660 Super

| Component | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | **Avg** |
|---|---|---|---|---|---|---|
| Intel Core i7-13700KF · 3400 MHz · 16 cores / 24 threads (CPU) | 5.20 s | 5.18 s | 5.17 s | 5.22 s | 5.23 s | **5.20 s** |
| NVIDIA GeForce GTX 1660 Super · 1280 CUDA cores (CUDA) | 0.86 s | 0.84 s | 0.86 s | 0.85 s | 0.85 s | **0.85 s** |

> **Speedup: ~6.1×** faster with CUDA on this machine.

### Summary

| | Machine 1 (Ryzen 7 5700G) | Machine 2 (i7-13700KF) |
|---|---|---|
| CPU avg | 10.68 s | 5.20 s |
| CUDA avg | 1.32 s | 0.85 s |
| **Speedup** | **~8.1×** | **~6.1×** |

CUDA parallelization delivers a significant speedup on both machines. The GPU runs all ants simultaneously — each ant is assigned one thread — whereas the CPU must construct tours sequentially. The GTX 1660 Super achieves this despite being a mid-range GPU, demonstrating that even modest CUDA hardware substantially outperforms a multi-core CPU for this workload.

---

## Requirements

- Python 3.11 or 3.12 (3.13 requires manual Tcl/Tk path fix, already applied)
- `numpy`
- `numba` (for CUDA mode)
- NVIDIA GPU with CUDA driver (for CUDA mode)
- CUDA Toolkit 11.2+ installed (for CUDA mode)
