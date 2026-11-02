"""Batched, differentiable reverse-stress-testing environment.

A 4-factor market (equity index, tech stock, 10y yield, index implied vol) with
conditionally heteroskedastic dynamics: the daily volatility of every factor
scales with the current implied-vol level, so the *order* in which shocks arrive
changes how plausible a path is.

An adversary chooses standardized innovations z_t (one 4-vector per day). The
plausibility of a whole path is its Gaussian negative log-likelihood,
sum_t ||z_t||^2, and the adversary must stay inside a fixed budget (by default
the 99% chi-square quantile with 4*T degrees of freedom). The adversary's reward
is the portfolio's loss.

The portfolio is nonlinear and path dependent: long equities and bonds, a
*daily delta-hedged* short straddle on the index (short gamma and vega), and
unhedged short puts on the tech stock. Position sizes can be rescaled (and
flipped) per episode so one policy learns to attack many books.

Everything is written in torch so the same simulator is used by PPO (no gradients
needed) and by the gradient-based "oracle" baseline (autograd through the path).
"""
from dataclasses import dataclass
import math
import torch
from scipy.stats import chi2

SQ252 = math.sqrt(252.0)

# factor order: [index log-return, tech log-return, 10y yield change, log implied vol]
CORR = torch.tensor([
    [1.00, 0.80, 0.20, -0.75],
    [0.80, 1.00, 0.15, -0.65],
    [0.20, 0.15, 1.00, -0.15],
    [-0.75, -0.65, -0.15, 1.00],
], dtype=torch.float64)
CHOL = torch.linalg.cholesky(CORR)
FACTOR_NAMES = ["Index", "Tech", "10y yield", "Index IV"]
POSITION_NAMES = ["Long index", "Long tech", "Long 10y bonds",
                  "Short index straddle (Δ-hedged)", "Short tech 25Δ puts"]


@dataclass
class MarketConfig:
    T: int = 10                 # horizon in trading days
    S0: float = 100.0           # index level
    Q0: float = 100.0           # tech stock level
    y0: float = 0.04            # 10y yield
    v0: float = 0.18            # index ATM implied vol
    tech_vol_mult: float = 1.5  # tech vol = mult * index vol
    yield_daily_vol: float = 0.0007
    volvol_daily: float = 0.07  # daily std of log implied vol
    r: float = 0.04
    straddle_days: int = 21
    put_days: int = 30
    confidence: float = 0.99
    # base notionals in $m
    eq_notional: float = 50.0
    tech_notional: float = 20.0
    bond_notional: float = 100.0
    bond_duration: float = 8.5
    bond_convexity: float = 90.0
    straddle_notional: float = 200.0
    put_notional: float = 60.0

    @property
    def budget(self) -> float:
        return float(chi2.ppf(self.confidence, 4 * self.T))


def ncdf(x):
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def bs(S, K, vol, tau, r):
    """Black-Scholes call, put and call delta (vectorized)."""
    tau = torch.clamp(tau, min=1e-6)
    sd = vol * torch.sqrt(tau)
    d1 = (torch.log(S / K) + (r + 0.5 * vol * vol) * tau) / sd
    d2 = d1 - sd
    disc = torch.exp(-r * tau)
    call = S * ncdf(d1) - K * disc * ncdf(d2)
    put = K * disc * ncdf(-d2) - S * ncdf(-d1)
    return call, put, ncdf(d1)


class StressEnv:
    """Vectorized environment; every tensor has leading batch dimension B."""

    obs_dim = 13
    act_dim = 5  # 4 shock-direction components + 1 logit for share of remaining budget

    def __init__(self, cfg: MarketConfig = MarketConfig(), device="cpu"):
        self.cfg = cfg
        self.B_total = cfg.budget
        self.dev = device
        c = cfg
        # 25-delta tech put strike
        tv = c.tech_vol_mult * c.v0
        tau_p = c.put_days / 252
        d1 = 0.6744897501960817  # N^-1(0.75)
        self.K_put = c.Q0 * math.exp(-(d1 * tv * math.sqrt(tau_p) - (c.r + 0.5 * tv * tv) * tau_p))
        self.K_str = c.S0

    # ----------------------------------------------------------------- state
    def reset(self, scales: torch.Tensor):
        """scales: (B,5) multipliers on the base position sizes (negative = flipped)."""
        c = self.cfg
        B = scales.shape[0]
        f = lambda v: torch.full((B,), v, dtype=torch.float64, device=self.dev)
        self.scales = scales.to(torch.float64)
        self.S, self.Q, self.y, self.v = f(c.S0), f(c.Q0), f(c.y0), f(c.v0)
        self.t = 0
        self.spent = f(0.0)
        self.cum_loss = f(0.0)
        self.n_eq = self.scales[:, 0] * c.eq_notional / c.S0
        self.n_tech = self.scales[:, 1] * c.tech_notional / c.Q0
        self.bond = self.scales[:, 2] * c.bond_notional
        self.n_str = self.scales[:, 3] * c.straddle_notional / c.S0
        self.n_put = self.scales[:, 4] * c.put_notional / c.Q0
        self._mark()
        self.hedge = self.n_str * (2 * self.str_delta - 1.0)  # offset straddle delta
        return self.obs()

    def _mark(self):
        c = self.cfg
        tau_s = torch.full_like(self.S, (c.straddle_days - self.t) / 252)
        tau_p = torch.full_like(self.S, (c.put_days - self.t) / 252)
        call, put, dcall = bs(self.S, self.K_str, self.v, tau_s, c.r)
        self.str_val = call + put
        self.str_delta = dcall  # straddle delta = 2*dcall - 1
        _, tput, _ = bs(self.Q, self.K_put, c.tech_vol_mult * self.v, tau_p, c.r)
        self.put_val = tput

    def daily_sigma(self):
        c = self.cfg
        return torch.stack([
            self.v / SQ252,
            c.tech_vol_mult * self.v / SQ252,
            torch.full_like(self.v, c.yield_daily_vol),
            torch.full_like(self.v, c.volvol_daily),
        ], dim=1)

    def obs(self):
        c = self.cfg
        rem = (self.B_total - self.spent) / self.B_total
        return torch.cat([
            torch.stack([
                torch.full_like(self.S, self.t / c.T),
                rem,
                torch.log(self.S / c.S0) * 10,
                torch.log(self.Q / c.Q0) * 10,
                (self.y - c.y0) * 100,
                torch.log(self.v / c.v0),
                self.cum_loss / 20.0,
                (self.n_str * (2 * self.str_delta - 1) - self.hedge) * c.S0 / 100,  # residual straddle delta
            ], dim=1),
            self.scales,
        ], dim=1).float()

    # ------------------------------------------------------------------ step
    def step_z(self, z: torch.Tensor):
        """Advance one day with standardized innovations z (B,4). Returns the day's loss ($m)."""
        c = self.cfg
        z = z.to(torch.float64)
        sig = self.daily_sigma()
        move = (z @ CHOL.T.to(z.device)) * sig
        S0, Q0, y0 = self.S, self.Q, self.y
        str0, put0 = self.str_val, self.put_val
        S1 = S0 * torch.exp(move[:, 0] - 0.5 * sig[:, 0] ** 2)
        Q1 = Q0 * torch.exp(move[:, 1] - 0.5 * sig[:, 1] ** 2)
        y1 = y0 + move[:, 2]
        v1 = torch.clamp(self.v * torch.exp(move[:, 3]), 0.05, 2.0)
        dy = y1 - y0
        self.S, self.Q, self.y, self.v = S1, Q1, y1, v1
        self.t += 1
        self._mark()
        comps = torch.stack([
            self.n_eq * (S1 - S0),
            self.n_tech * (Q1 - Q0),
            self.bond * (-c.bond_duration * dy + 0.5 * c.bond_convexity * dy * dy),
            -self.n_str * (self.str_val - str0) + self.hedge * (S1 - S0),
            -self.n_put * (self.put_val - put0),
        ], dim=1)
        self.last_loss_by_position = -comps  # (B,5) loss attribution, $m
        pnl = comps.sum(1)
        # rebalance the straddle's delta hedge at end of day
        self.hedge = self.n_str * (2 * self.str_delta - 1.0)
        loss = -pnl
        self.cum_loss = self.cum_loss + loss
        self.spent = self.spent + (z * z).sum(1)
        return loss

    def action_to_z(self, a: torch.Tensor):
        """Map raw policy output (B,5) to a budget-respecting shock z (B,4)."""
        a = a.to(torch.float64)
        d = a[:, :4]
        d = d / (d.norm(dim=1, keepdim=True) + 1e-8)
        frac = torch.sigmoid(a[:, 4])
        if self.t == self.cfg.T - 1:
            frac = torch.ones_like(frac)  # spend what is left on the last day
        rem = torch.clamp(self.B_total - self.spent, min=0.0)
        return d * torch.sqrt(frac * rem).unsqueeze(1)

    def step(self, a: torch.Tensor):
        z = self.action_to_z(a)
        loss = self.step_z(z)
        done = self.t >= self.cfg.T
        return self.obs(), loss, done, z


def sample_scales(B, generator=None, low=-1.5, high=1.5):
    """Random long/short books: each position size is base * U[low, high] (negative = flipped)."""
    return low + (high - low) * torch.rand(B, 5, generator=generator, dtype=torch.float64)


def run_path(env: StressEnv, scales: torch.Tensor, Z: torch.Tensor):
    """Run full paths Z (B,T,4); returns total loss (B,) and the plausibility cost (B,)."""
    env.reset(scales)
    total = torch.zeros(scales.shape[0], dtype=torch.float64)
    for t in range(env.cfg.T):
        total = total + env.step_z(Z[:, t])
    return total, (Z.to(torch.float64) ** 2).sum((1, 2))
