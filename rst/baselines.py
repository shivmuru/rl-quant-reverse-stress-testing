"""Classical reverse-stress-testing baselines, all held to the same plausibility budget."""
import math
import torch
from .env import StressEnv, CHOL, CAL, NU, run_path, day_cost, norm_for_cost


def monte_carlo_worst(env: StressEnv, scales, n=100_000, batch=25_000, seed=0):
    """Simulate paths from the fitted (Student-t) model; keep those inside the budget; return the worst."""
    g = torch.Generator().manual_seed(seed)
    B = scales.shape[0]
    best = torch.full((B,), -math.inf, dtype=torch.float64)
    best_Z = torch.zeros(B, env.cfg.T, 4, dtype=torch.float64)
    for _ in range(n // batch):
        Z = torch.randn(B, batch, env.cfg.T, 4, generator=g, dtype=torch.float64)
        W = torch.distributions.Chi2(torch.tensor(NU, dtype=torch.float64)).sample((B, batch, env.cfg.T, 1))
        Z = Z * torch.sqrt((NU - 2) / W)  # unit-variance multivariate t
        sc = scales.unsqueeze(1).expand(B, batch, 5).reshape(-1, 5)
        L, cost = run_path(env, sc, Z.reshape(-1, env.cfg.T, 4))
        L = torch.where(cost <= env.B_total, L, torch.full_like(L, -math.inf)).view(B, batch)
        v, i = L.max(1)
        upd = v > best
        best = torch.where(upd, v, best)
        best_Z[upd] = Z[torch.arange(B), i][upd]
    return best, best_Z


# Worst 10-day S&P 500 drawdown inside each crisis window, measured from the data
# by calibrate.py: (index %, tech %, 10y yield bp, VIX end/start ratio).
HISTORICAL = {
    name: (h["index_pct"], h["tech_pct"], h["yield_bp"], h["vix_end"] / h["vix_start"])
    for name, h in CAL["historical_scenarios"].items()
}


def scenario_path(env: StressEnv, scales, moves, k: float):
    """Spread k * (real 10-day crisis move) evenly over the horizon and back out the implied z path."""
    c = env.cfg
    T = c.T
    eq, te, yb, vol_ratio = moves
    per = torch.tensor([
        math.log(1 + k * eq / 100) / T,
        math.log(1 + k * te / 100) / T,
        k * yb / 1e4 / T,
        k * math.log(vol_ratio) / T,
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
    return total, day_cost(Z).sum(1), Z


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

    static=True: classical "maximum loss" stress test: the whole budget is an
    instantaneous day-1 shock, followed by quiet days.
    static=False: full 10-day path ("oracle" - needs a differentiable pricer and a
    fresh optimization for every portfolio).
    """
    g = torch.Generator().manual_seed(seed)
    B, T = scales.shape[0], env.cfg.T
    sc = scales.repeat_interleave(restarts, 0)
    # Exact parametrization of the budget set: day t spends cost
    # budget * sigmoid(rho) * softmax(a)_t in direction U_t / |U_t|.
    U = torch.randn(B * restarts, T, 4, generator=g, dtype=torch.float64).requires_grad_(True)
    a = torch.zeros(B * restarts, T, dtype=torch.float64, requires_grad=True)
    rho = torch.full((B * restarts, 1), 3.0, dtype=torch.float64, requires_grad=True)
    neg = torch.zeros(1, T, dtype=torch.float64)
    if static:
        neg[:, 1:] = -torch.inf

    def make_Z():
        c = env.B_total * torch.sigmoid(rho) * torch.softmax(a + neg, dim=1)
        return norm_for_cost(c).unsqueeze(-1) * U / U.norm(dim=-1, keepdim=True)

    opt = torch.optim.Adam([U, a, rho], lr=lr)
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
