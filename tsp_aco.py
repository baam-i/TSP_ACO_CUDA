#!/usr/bin/env python3
"""
Traveling Salesman Problem — Ant Colony Optimization
Supports CPU and CUDA (GPU) modes via Numba.

Usage examples:
    python tsp_aco.py --cities 100 --ants 50 --iterations 200 --cpu
    python tsp_aco.py --cities 100 --ants 50 --iterations 200 --cuda
    python tsp_aco.py --file berlin52.tsp --cpu --alpha 1.0 --beta 5.0
    python tsp_aco.py --cities 200 --cuda --quiet
"""

import argparse
import math
import sys
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

# ─── CUDA availability ────────────────────────────────────────────────────────

_CUDA_AVAILABLE = False
_CUDA_ERROR: Optional[str] = None


def _probe_cuda() -> None:
    global _CUDA_AVAILABLE, _CUDA_ERROR
    try:
        from numba import cuda
        if len(cuda.gpus) > 0:
            _CUDA_AVAILABLE = True
        else:
            _CUDA_ERROR = "No CUDA-capable GPU detected by numba"
    except ImportError:
        _CUDA_ERROR = "numba not installed  ->  pip install numba"
    except Exception as exc:
        _CUDA_ERROR = str(exc)


_probe_cuda()

# ─── Data structures ──────────────────────────────────────────────────────────


@dataclass
class ACOParams:
    num_ants: int = 50
    num_iterations: int = 200
    alpha: float = 1.0        # pheromone exponent
    beta: float = 3.0         # heuristic (distance) exponent
    rho: float = 0.1          # evaporation rate  0 < rho < 1
    Q: float = 100.0          # pheromone deposit constant
    initial_pheromone: float = 1.0


@dataclass
class RunStats:
    mode: str
    num_cities: int
    params: ACOParams
    best_tour_length: float
    best_tour: List[int]
    iteration_bests: List[float]
    total_time: float
    avg_iter_ms: float


# ─── TSP instance ─────────────────────────────────────────────────────────────


class TSPInstance:
    """City coordinates + precomputed Euclidean distance matrix."""

    def __init__(self, coords: np.ndarray):
        self.coords = np.asarray(coords, dtype=np.float64)
        self.n = len(self.coords)
        self.distances = self._euclidean()

    def _euclidean(self) -> np.ndarray:
        d = self.coords[:, np.newaxis, :] - self.coords[np.newaxis, :, :]
        return np.sqrt((d ** 2).sum(axis=2))

    @classmethod
    def random(cls, n: int, seed: Optional[int] = None) -> "TSPInstance":
        rng = np.random.default_rng(seed)
        return cls(rng.uniform(0.0, 1000.0, size=(n, 2)))

    @classmethod
    def from_file(cls, path: str) -> "TSPInstance":
        """
        Accepts:
          - TSPLIB format  (NODE_COORD_SECTION block: index x y)
          - Plain text     (one line per city: x y  or  index x y)
        """
        coords: List[List[float]] = []
        in_coords = False
        with open(path) as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#") or line.startswith("!"):
                    continue
                upper = line.upper()
                if upper == "NODE_COORD_SECTION":
                    in_coords = True
                    continue
                if upper in ("EOF", "EDGE_WEIGHT_SECTION", "DISPLAY_DATA_SECTION"):
                    in_coords = False
                    continue
                if ":" in line and not in_coords:
                    continue
                parts = line.split()
                try:
                    if len(parts) == 3:
                        coords.append([float(parts[1]), float(parts[2])])
                    elif len(parts) == 2:
                        coords.append([float(parts[0]), float(parts[1])])
                except ValueError:
                    pass
        if not coords:
            raise ValueError(f"No coordinates found in {path!r}")
        return cls(np.array(coords))


# ─── CPU ACO ──────────────────────────────────────────────────────────────────


class ACO_CPU:
    def __init__(self, instance: TSPInstance, params: ACOParams):
        self.instance = instance
        self.params = params
        self.n = instance.n
        self.dist = instance.distances
        with np.errstate(divide="ignore", invalid="ignore"):
            self.heuristic = np.where(self.dist > 0, 1.0 / self.dist, 0.0)
        self.pheromone = np.full(
            (self.n, self.n), params.initial_pheromone, dtype=np.float64
        )

    def _build_tour(self, rng: np.random.Generator):
        visited = np.zeros(self.n, dtype=bool)
        current = int(rng.integers(self.n))
        tour = [current]
        visited[current] = True
        p = self.params

        for _ in range(self.n - 1):
            prob = (
                self.pheromone[current] ** p.alpha
                * self.heuristic[current] ** p.beta
            )
            prob[visited] = 0.0
            total = prob.sum()
            if total == 0.0:
                nxt = int(rng.choice(np.where(~visited)[0]))
            else:
                prob /= total
                nxt = int(rng.choice(self.n, p=prob))
            tour.append(nxt)
            visited[nxt] = True
            current = nxt

        arr = np.array(tour, dtype=np.int32)
        length = float(self.dist[arr, np.roll(arr, -1)].sum())
        return tour, length

    def solve(self, seed: Optional[int] = None, verbose: bool = True) -> RunStats:
        p = self.params
        rng = np.random.default_rng(seed)
        best_tour: List[int] = []
        best_len = math.inf
        iter_bests: List[float] = []

        t0 = time.perf_counter()

        for it in range(p.num_iterations):
            it_t0 = time.perf_counter()
            tours, lengths = [], []

            for _ in range(p.num_ants):
                tour, length = self._build_tour(rng)
                tours.append(tour)
                lengths.append(length)
                if length < best_len:
                    best_len, best_tour = length, tour[:]

            ib = min(lengths)
            iter_bests.append(ib)

            # Evaporate
            self.pheromone *= 1.0 - p.rho

            # Deposit (vectorized; TSP tours have unique edges so += is safe)
            for tour, length in zip(tours, lengths):
                dep = p.Q / length
                arr = np.array(tour, dtype=np.int32)
                nxt = np.roll(arr, -1)
                self.pheromone[arr, nxt] += dep
                self.pheromone[nxt, arr] += dep

            if verbose:
                ms = (time.perf_counter() - it_t0) * 1000
                print(
                    f"  Iter {it+1:4d}/{p.num_iterations}  "
                    f"iter-best: {ib:10.2f}  global-best: {best_len:10.2f}  "
                    f"{ms:6.1f} ms   ",
                    end="\r",
                )

        if verbose:
            print()

        total = time.perf_counter() - t0
        return RunStats(
            mode="CPU",
            num_cities=self.n,
            params=p,
            best_tour_length=best_len,
            best_tour=best_tour,
            iteration_bests=iter_bests,
            total_time=total,
            avg_iter_ms=total / p.num_iterations * 1000,
        )


# ─── CUDA ACO ─────────────────────────────────────────────────────────────────


def _compile_cuda_kernels():
    """JIT-compile CUDA kernels (called once at solver construction)."""
    from numba import cuda
    from numba.cuda.random import (
        create_xoroshiro128p_states,
        xoroshiro128p_uniform_float32,
    )

    @cuda.jit
    def construct_tours_kernel(
        pheromone, heuristic, tours, tour_lengths, distances,
        rng_states, alpha, beta, n, visited,
    ):
        """
        One thread per ant. Each thread independently builds a complete TSP tour
        using roulette-wheel selection over pheromone x heuristic probabilities.

        Complexity per ant: O(n²) — standard for ACO construction.
        Maximum supported cities: limited by GPU global-memory, not registers.
        """
        ant = cuda.grid(1)
        if ant >= tours.shape[0]:
            return

        # Reset visited flags for this ant
        for i in range(n):
            visited[ant, i] = False

        # Random starting city
        r = xoroshiro128p_uniform_float32(rng_states, ant)
        current = int(r * n)
        if current >= n:
            current = n - 1
        tours[ant, 0] = current
        visited[ant, current] = True

        for step in range(1, n):
            # Compute total weight (denominator)
            total = 0.0
            for j in range(n):
                if not visited[ant, j]:
                    total += (pheromone[current, j] ** alpha) * (heuristic[current, j] ** beta)

            # Roulette-wheel selection
            r = xoroshiro128p_uniform_float32(rng_states, ant)
            threshold = r * total
            cumsum = 0.0
            nxt = -1
            for j in range(n):
                if not visited[ant, j]:
                    cumsum += (pheromone[current, j] ** alpha) * (heuristic[current, j] ** beta)
                    if cumsum >= threshold:
                        nxt = j
                        break

            # Fallback for floating-point rounding edge case
            if nxt < 0:
                for j in range(n):
                    if not visited[ant, j]:
                        nxt = j
                        break

            tours[ant, step] = nxt
            visited[ant, nxt] = True
            current = nxt

        # Tour length
        length = 0.0
        for i in range(n):
            a = tours[ant, i]
            b = tours[ant, (i + 1) % n]
            length += distances[a, b]
        tour_lengths[ant] = length

    return construct_tours_kernel, create_xoroshiro128p_states


class ACO_CUDA:
    def __init__(self, instance: TSPInstance, params: ACOParams):
        if not _CUDA_AVAILABLE:
            raise RuntimeError(f"CUDA unavailable: {_CUDA_ERROR}")
        from numba import cuda

        self.instance = instance
        self.params = params
        self.n = instance.n

        dist32 = instance.distances.astype(np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            heur32 = np.where(dist32 > 0, (1.0 / dist32).astype(np.float32), np.float32(0.0))

        self.d_dist = cuda.to_device(dist32)
        self.d_heur = cuda.to_device(heur32)
        self.pheromone = np.full((self.n, self.n), params.initial_pheromone, dtype=np.float32)

        print("  Compiling CUDA kernels...", end=" ", flush=True)
        self._kernel, self._make_rng = _compile_cuda_kernels()
        print("done")

    def solve(self, seed: Optional[int] = None, verbose: bool = True) -> RunStats:
        from numba import cuda

        p = self.params
        n = self.n
        na = p.num_ants

        d_tours = cuda.device_array((na, n), dtype=np.int32)
        d_lengths = cuda.device_array(na, dtype=np.float32)
        d_visited = cuda.device_array((na, n), dtype=np.bool_)

        rng_seed = seed if seed is not None else int(time.time() * 1e6) & 0x7FFF_FFFF
        rng = self._make_rng(na, seed=rng_seed)

        tpb = min(256, na)
        bpg = math.ceil(na / tpb)

        best_tour: List[int] = []
        best_len = math.inf
        iter_bests: List[float] = []

        t0 = time.perf_counter()

        for it in range(p.num_iterations):
            it_t0 = time.perf_counter()

            d_pheromone = cuda.to_device(self.pheromone)

            self._kernel[bpg, tpb](
                d_pheromone, self.d_heur, d_tours, d_lengths,
                self.d_dist, rng,
                np.float32(p.alpha), np.float32(p.beta), np.int32(n),
                d_visited,
            )
            cuda.synchronize()

            tours = d_tours.copy_to_host()
            lengths = d_lengths.copy_to_host()

            ib = float(lengths.min())
            iter_bests.append(ib)
            if ib < best_len:
                best_len = ib
                best_tour = tours[int(lengths.argmin())].tolist()

            # Pheromone update — evaporate + deposit (CPU, vectorized)
            self.pheromone *= np.float32(1.0 - p.rho)
            for ant in range(na):
                dep = np.float32(p.Q / lengths[ant])
                arr = tours[ant]
                nxt = np.roll(arr, -1)
                self.pheromone[arr, nxt] += dep
                self.pheromone[nxt, arr] += dep

            if verbose:
                ms = (time.perf_counter() - it_t0) * 1000
                print(
                    f"  Iter {it+1:4d}/{p.num_iterations}  "
                    f"iter-best: {ib:10.2f}  global-best: {best_len:10.2f}  "
                    f"{ms:6.1f} ms   ",
                    end="\r",
                )

        if verbose:
            print()

        total = time.perf_counter() - t0
        return RunStats(
            mode="CUDA",
            num_cities=n,
            params=p,
            best_tour_length=best_len,
            best_tour=best_tour,
            iteration_bests=iter_bests,
            total_time=total,
            avg_iter_ms=total / p.num_iterations * 1000,
        )


# ─── Statistics & display ─────────────────────────────────────────────────────


def _convergence_chart(values: List[float], width: int = 55, height: int = 8) -> str:
    if not values:
        return ""

    # Downsample if more values than columns
    if len(values) > width:
        step = len(values) / width
        sampled = [values[min(int(i * step), len(values) - 1)] for i in range(width)]
    else:
        sampled = list(values)

    lo, hi = min(sampled), max(sampled)
    span = hi - lo if hi != lo else 1.0
    n_cols = len(sampled)

    lines: List[str] = []
    for row in range(height, 0, -1):
        threshold = lo + span * row / height
        bar = "".join("#" if v >= threshold else " " for v in sampled)
        if row == height:
            label = f" {hi:.1f}"
        elif row == math.ceil(height / 2):
            label = f" {(lo + hi) / 2:.1f}"
        elif row == 1:
            label = f" {lo:.1f}"
        else:
            label = ""
        lines.append(f"    |{bar[:n_cols]}{label}")

    lines.append(f"    +{'-' * n_cols}")
    lines.append(f"     1{' ' * max(0, n_cols - 12)}{len(values)}")
    return "\n".join(lines)


def print_stats(stats: RunStats) -> None:
    SEP = "=" * 64
    p = stats.params
    bests = stats.iteration_bests

    improvement_pct = (bests[0] - bests[-1]) / bests[0] * 100 if bests[0] else 0.0

    # Last iteration where global best improved
    running = bests[0]
    conv_iter = 0
    for i, b in enumerate(bests):
        if b < running:
            running, conv_iter = b, i

    throughput = p.num_ants / (stats.avg_iter_ms / 1000)

    print(f"\n{SEP}")
    print(f"  TSP-ACO RESULTS  --  {stats.mode} MODE")
    print(SEP)

    print("\n  PROBLEM")
    print(f"    Cities              : {stats.num_cities}")

    print("\n  PARAMETERS")
    print(f"    Ants                : {p.num_ants}")
    print(f"    Iterations          : {p.num_iterations}")
    print(f"    Alpha  (pheromone)  : {p.alpha}")
    print(f"    Beta   (heuristic)  : {p.beta}")
    print(f"    Rho    (evaporation): {p.rho}")
    print(f"    Q      (deposit)    : {p.Q}")
    print(f"    Initial pheromone   : {p.initial_pheromone}")

    print("\n  RESULTS")
    print(f"    Best tour length    : {stats.best_tour_length:.4f}")
    print(f"    First iteration best: {bests[0]:.4f}")
    print(f"    Last  iteration best: {bests[-1]:.4f}")
    print(f"    Improvement         : {improvement_pct:.2f}%")
    print(f"    Converged at iter   : {conv_iter + 1}")

    print("\n  PERFORMANCE")
    print(f"    Total time          : {stats.total_time:.3f} s")
    print(f"    Avg per iteration   : {stats.avg_iter_ms:.2f} ms")
    print(f"    Throughput          : {throughput:,.0f} ant-tours/s")

    print("\n  CONVERGENCE  (tour length over iterations -- lower is better)")
    print(_convergence_chart(bests))

    n_show = min(stats.num_cities, 20)
    suffix = f" ... +{stats.num_cities - n_show} more" if stats.num_cities > n_show else ""
    tour_preview = " -> ".join(str(c) for c in stats.best_tour[:n_show])
    print(f"\n  BEST TOUR  (first {n_show} cities)")
    print(f"    {tour_preview}{suffix}")

    print(f"\n{SEP}\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Traveling Salesman Problem — Ant Colony Optimization (CPU / CUDA)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    mg = ap.add_mutually_exclusive_group()
    mg.add_argument("--cpu", action="store_true", help="Force CPU mode")
    mg.add_argument(
        "--cuda", action="store_true",
        help="Force CUDA (GPU) mode — requires numba and a CUDA GPU",
    )

    ap.add_argument(
        "--cities", type=int, default=50,
        help="Number of random cities (ignored when --file is given)",
    )
    ap.add_argument(
        "--file", default=None,
        help="Path to city coordinate file (TSPLIB or plain 'x y' per line)",
    )

    ap.add_argument("--ants", type=int, default=50, metavar="N")
    ap.add_argument("--iterations", type=int, default=200, metavar="N")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="Pheromone importance exponent")
    ap.add_argument("--beta", type=float, default=3.0,
                    help="Heuristic (inverse distance) importance exponent")
    ap.add_argument("--rho", type=float, default=0.1,
                    help="Pheromone evaporation rate  (0 < rho < 1)")
    ap.add_argument("--Q", type=float, default=100.0,
                    help="Pheromone deposit constant")
    ap.add_argument("--initial-pheromone", type=float, default=1.0,
                    help="Initial pheromone level on all edges")
    ap.add_argument("--seed", type=int, default=None, help="Random seed")
    ap.add_argument("--quiet", action="store_true",
                    help="Suppress per-iteration progress line")

    return ap


# ─── Entry point ──────────────────────────────────────────────────────────────


def main() -> None:
    args = build_parser().parse_args()

    # ── Mode selection ────────────────────────────────────────────────────────
    if args.cuda and not _CUDA_AVAILABLE:
        print(f"[ERROR] CUDA requested but unavailable: {_CUDA_ERROR}", file=sys.stderr)
        sys.exit(1)

    if args.cuda:
        mode = "cuda"
    elif args.cpu:
        mode = "cpu"
    else:
        mode = "cuda" if _CUDA_AVAILABLE else "cpu"
        print(f"[auto] CUDA available={_CUDA_AVAILABLE} -> selecting {mode.upper()} mode")

    print(f"Mode: {mode.upper()}")
    if mode == "cuda":
        print(f"  CUDA: {_CUDA_AVAILABLE}")

    # ── Load / generate instance ──────────────────────────────────────────────
    if args.file:
        print(f"Loading cities from  {args.file} ...")
        instance = TSPInstance.from_file(args.file)
    else:
        print(f"Generating {args.cities} random cities  (seed={args.seed}) ...")
        instance = TSPInstance.random(args.cities, seed=args.seed)

    print(f"Cities: {instance.n}")

    # ── Parameters ────────────────────────────────────────────────────────────
    params = ACOParams(
        num_ants=args.ants,
        num_iterations=args.iterations,
        alpha=args.alpha,
        beta=args.beta,
        rho=args.rho,
        Q=args.Q,
        initial_pheromone=args.initial_pheromone,
    )

    print(
        f"ACO  {params.num_ants} ants x {params.num_iterations} iterations  "
        f"(alpha={params.alpha}  beta={params.beta}  rho={params.rho}  Q={params.Q})\n"
    )

    # ── Solve ─────────────────────────────────────────────────────────────────
    solver: ACO_CUDA | ACO_CPU
    if mode == "cuda":
        solver = ACO_CUDA(instance, params)
    else:
        solver = ACO_CPU(instance, params)

    stats = solver.solve(seed=args.seed, verbose=not args.quiet)
    print_stats(stats)


if __name__ == "__main__":
    main()
