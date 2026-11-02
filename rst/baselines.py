"""Classical reverse-stress-testing baselines, all held to the same plausibility budget."""
import math
import torch
from .env import StressEnv, CHOL, run_path


def monte_carlo_worst(env: StressEnv, scales, n=100_000, batch=25_000, seed=0):
    """Simulate paths from the model; keep those inside the plausibility budget; return the worst."""
    g = torch.Generator().manual_seed(seed)
    B = scales.shape[0]
    best = torch.full((B,), -math.inf, dtype=torch.float64)
    best_Z = torch.zeros(B, env.cfg.T, 4, dtype=torch.float64)
    for _ in range(n // batch):
        Z = torch.randn(B, batch, env.cfg.T, 4, generator=g, dtype=torch.float64)
        sc = scales.unsqueeze(1).expand(B, batch, 5).reshape(-1, 5)
        L, cost = run_path(env, sc, Z.reshape(-1, env.cfg.T, 4))
        L = torch.where(cost <= env.B_total, L, torch.full_like(L, -math.inf)).view(B, batch)
        v, i = L.max(1)
        upd = v > best
        best = torch.where(upd, v, best)
        best_Z[upd] = Z[torch.arange(B), i][upd]
    return best, best_Z


# Stylized 10-day historical episodes: (index %, tech %, 10y yield bp, index IV end level).
# Approximate magnitudes for illustration, not exact market data.
HISTORICAL = {
    "Oct-2008 crash": (-18.0, -20.0, -40.0, 0.55),
    "Aug-2015 deval": (-11.0, -12.0, -20.0, 0.40),
    "Feb-2018 Volmageddon": (-10.0, -9.0, +10.0, 0.37),
    "Mar-2020 COVID": (-25.0, -22.0, -80.0, 0.75),
    "2022 rates shock": (-8.0, -12.0, +60.0, 0.32),
}


def scenario_path(env: StressEnv, scales, moves, k: float):
    """Spread k * (historical 10-day move) evenly over the horizon and back out the implied z path."""
    c = env.cfg
    T = c.T
    eq, te, yb, vend = moves
    per = torch.tensor([
        math.log(1 + k * eq / 100) / T,
        math.log(1 + k * te / 100) / T,
        k * yb / 1e4 / T,
        k * math.log(vend / c.v0) / T,
    ], dtype=torch.float64)
    B = scales.shape[0]
    env.reset(scales)
    Zs, total = [], torch.zeros(B, dtype=torch.float64)
    for _ in range(T):
        sig = env.daily_sigma()
        move = per.expand(B, 4).clone()
        move[:, 0] += 0.5 * sig[:, 0] ** 2  # undo the Ito drift in the simulator
        move[:, 1] += 0.5 * sig[:, 1] ** 2
        eps = move / sig
        z = torch.linalg.solve_triangular(CHOL, eps.T, upper=False).T
        Zs.append(z)
        total = total + env.step_z(z)
    Z = torch.stack(Zs, 1)
    return total, (Z ** 2).sum((1, 2)), Z


def historical_rescaled(env: StressEnv, scales, moves):
    """Scale a historical scenario up or down until it sits exactly on the plausibility budget."""
    lo, hi = 0.0, 5.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        _, cost, _ = scenario_path(env, scales[:1], moves, mid)
        if cost.item() > env.B_total:
            hi = mid
        else:
            lo = mid
    L, cost, Z = scenario_path(env, scales, moves, lo)
    return L, lo, Z


def gradient_worst(env: StressEnv, scales, static=False, restarts=16, iters=300, lr=0.05, seed=0):
    """Gradient ascent on the path (autograd through the simulator), multi-start.

    static=True: classical "maximum loss on the ellipsoid": the whole budget is an
    instantaneous day-1 shock, followed by quiet days.
    static=False: full 10-day path ("oracle" - needs a differentiable pricer and a
    fresh optimization for every portfolio).
    """
    g = torch.Generator().manual_seed(seed)
    B, T = scales.shape[0], env.cfg.T
    sc = scales.repeat_interleave(restarts, 0)
    mask = torch.ones(1, T, 1, dtype=torch.float64)
    if static:
        mask[:, 1:] = 0.0
    # Smooth parametrization of the budget ball: Z = sqrt(budget) * sigmoid(rho) * U / |U|
    U = torch.randn(B * restarts, T, 4, generator=g, dtype=torch.float64).requires_grad_(True)
    rho = torch.full((B * restarts, 1, 1), 3.0, dtype=torch.float64, requires_grad=True)

    def make_Z():
        Um = U * mask
        return math.sqrt(env.B_total) * torch.sigmoid(rho) * Um / Um.pow(2).sum((1, 2), keepdim=True).sqrt()

    opt = torch.optim.Adam([U, rho], lr=lr)
    for _ in range(iters):
        L, _ = run_path(env, sc, make_Z())
        opt.zero_grad()
        (-L.sum()).backward()
        opt.step()
    with torch.no_grad():
        Z = make_Z()
        L, _ = run_path(env, sc, Z)
        L = L.view(B, restarts)
        v, i = L.max(1)
        bestZ = Z.view(B, restarts, T, 4)[torch.arange(B), i]
    return v, bestZ
