"""Calibrate the 4-factor stress model to daily market data (2000 onward).

Inputs (data/):  spx.csv, qqq.csv, vix.csv (Yahoo via yfinance), dgs10.csv (FRED).
Output:          rst/calibration.json, read by rst/env.py as the default model.

Model being calibrated (see rst/env.py), with v_t = VIX_t / 100:
    index log-return   = idx_vol_scale  * v_{t-1}/sqrt(252) * e1
    tech  log-return   = tech_vol_scale * v_{t-1}/sqrt(252) * e2
    10y yield change   = yield_daily_vol                    * e3
    log VIX change     = volvol_daily                       * e4
    corr(e) = CORR,  e ~ N(0, CORR)

Plausibility: decorrelated daily innovations z_t follow a unit-variance multivariate
Student-t (nu fitted by maximum likelihood). A path's cost is sum_t (nu+4) log(1 + |z_t|^2/(nu-2)),
i.e. twice its negative log-likelihood, and the budget is the empirical 99th percentile of that
cost over real 10-day windows. Hard historical bounds also apply: each factor's daily move
is capped at its largest real move since 2000 (raw and vol-adjusted), each factor's
cumulative 10-day move stays inside its real 10-day range, and the VIX stays inside its
historical range.

Also produces:
  * the largest 10-day S&P 500 drawdown inside each historical crisis window
    (replaces the hand-set scenarios), and
  * a backtest showing why a Gaussian budget fails: how often real 10-day windows
    exceed the Gaussian chi-square 99% budget (should be ~1%; it is far higher).
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import chi2, kurtosis

DATA = Path("data")
T = 10
LAST_DATE = "2026-10-30"  # exclude any partial (intraday) session after the last full close

CRISES = {  # search windows; the worst 10-day S&P drop inside each is used
    "Sep-2001 9/11 reopening": ("2001-09-01", "2001-10-31"),
    "Jul-2002 dot-com bottom": ("2002-06-15", "2002-08-15"),
    "Oct-2008 GFC": ("2008-09-01", "2008-12-31"),
    "Aug-2011 US downgrade": ("2011-07-15", "2011-09-30"),
    "Aug-2015 China deval": ("2015-08-01", "2015-09-30"),
    "Feb-2018 Volmageddon": ("2018-01-20", "2018-03-15"),
    "Mar-2020 COVID": ("2020-02-15", "2020-04-15"),
    "2022 rates shock": ("2022-01-01", "2022-12-31"),
    "Apr-2025 tariff shock": ("2025-03-20", "2025-05-15"),
}


def load():
    px = lambda f, col: pd.read_csv(DATA / f, index_col=0, parse_dates=True)[col]
    df = pd.concat({
        "spx": px("spx.csv", "Close"),
        "qqq": px("qqq.csv", "Adj Close"),  # total return (dividends) for the ETF
        "vix": px("vix.csv", "Close"),
        "y10": px("dgs10.csv", "DGS10") / 100.0,
    }, axis=1, join="inner").sort_index()
    return df.loc[:LAST_DATE].dropna()


def standardize(df, p):
    """Daily model innovations eps_t (before decorrelation)."""
    sig_eq = (df.vix.shift(1) / 100.0) / math.sqrt(252)
    out = pd.DataFrame({
        "idx": np.log(df.spx).diff() / (p["idx_vol_scale"] * sig_eq),
        "tech": np.log(df.qqq).diff() / (p["tech_vol_scale"] * sig_eq),
        "y10": df.y10.diff() / p["yield_daily_vol"],
        "vol": np.log(df.vix).diff() / p["volvol_daily"],
    })
    return out.dropna()


def main():
    df = load()
    sig_eq = (df.vix.shift(1) / 100.0) / math.sqrt(252)
    r_idx = np.log(df.spx).diff()
    r_tech = np.log(df.qqq).diff()
    p = {
        # realized vol as a fraction of VIX-implied vol (VIX > realized on average)
        "idx_vol_scale": float((r_idx / sig_eq).std()),
        "tech_vol_scale": float((r_tech / sig_eq).std()),
        "yield_daily_vol": float(df.y10.diff().std()),
        "volvol_daily": float(np.log(df.vix).diff().std()),
    }
    eps = standardize(df, p)
    corr = eps.corr().values
    assert np.all(np.linalg.eigvalsh(corr) > 0), "correlation matrix not positive definite"
    L = np.linalg.cholesky(corr)

    # ---------------------------------------------------------- tail diagnostics
    z = np.linalg.solve(L, eps.values.T).T  # decorrelated innovations
    tails = {
        "excess_kurtosis": dict(zip(eps.columns, kurtosis(eps.values, axis=0).round(2).tolist())),
        "pct_daily_moves_beyond_3sd": dict(zip(eps.columns, ((eps.abs() > 3).mean() * 100).round(2).tolist())),
        "gaussian_pct_beyond_3sd": 0.27,
    }

    # ------------------------------------------ Student-t plausibility measure
    # Fit nu of a unit-variance multivariate t to the decorrelated daily innovations.
    d = 4
    r2 = (z ** 2).sum(1)

    def t_loglik(nu):
        s2 = (nu - 2) / nu  # unit variance
        return (gammaln((nu + d) / 2) - gammaln(nu / 2) - d / 2 * np.log(nu * np.pi * s2)
                - (nu + d) / 2 * np.log1p(r2 / (nu * s2))).sum()

    grid = np.arange(2.2, 60.0, 0.05)
    nu = float(grid[int(np.argmax([t_loglik(n) for n in grid]))])
    # path cost = 2 x negative log-likelihood, up to a constant (-> ||z||^2 as nu -> inf)
    t_cost = pd.Series((nu + d) * np.log1p(r2 / (nu - 2)), index=eps.index)
    g_cost = pd.Series(r2, index=eps.index)

    # ---------------------------------------- budgets and their backtests
    gauss_budget = float(chi2.ppf(0.99, d * T))
    g_roll = g_cost.rolling(T).sum().dropna()
    t_roll = t_cost.rolling(T).sum().dropna()
    budget = float(t_roll.quantile(0.99))  # empirical 99th pct of real 10-day path costs
    backtest = {
        "gaussian_chi2_budget_99pct": gauss_budget,
        "pct_10d_windows_over_gaussian_budget": float((g_roll.iloc[::T] > gauss_budget).mean() * 100),
        "student_t_nu": nu,
        "loglik_gain_t_vs_gaussian": float(t_loglik(nu) - t_loglik(1e7)),
        "budget_99pct": budget,
        "pct_10d_windows_over_budget": float((t_roll.iloc[::T] > budget).mean() * 100),
        "n_nonoverlapping_windows": int(len(t_roll.iloc[::T])),
    }
    # Hard per-day caps: no factor moves more in one day (in vol-adjusted units)
    # than its largest move since 2000.
    eps_cap = eps.abs().max().round(3).tolist()
    # ...and in raw terms: largest real daily move per factor.
    raw = pd.DataFrame({"idx": r_idx, "tech": r_tech, "y10": df.y10.diff(),
                        "vol": np.log(df.vix).diff()}).dropna()
    raw_cap = raw.abs().max().round(6).tolist()
    # Each factor's cumulative move over the horizon stays inside its real 10-day range.
    ten = pd.DataFrame({
        "idx": np.log(df.spx).diff(T), "tech": np.log(df.qqq).diff(T), "y10": df.y10.diff(T),
    }).dropna()
    ten_day_range = {k: [round(float(ten[k].min()), 6), round(float(ten[k].max()), 6)] for k in ten}
    vix_range = [round(float(df.vix.min()), 2), round(float(df.vix.max()), 2)]

    # ------------------------------------------------- historical crisis moves
    hist = {}
    log_spx = np.log(df.spx)
    for name, (a, b) in CRISES.items():
        w = df.loc[a:b]
        drop = (log_spx.shift(-T) - log_spx).loc[w.index].dropna()
        t0 = drop.idxmin()
        i0 = df.index.get_loc(t0)
        s, e = df.iloc[i0], df.iloc[i0 + T]
        hist[name] = {
            "start": str(t0.date()), "end": str(df.index[i0 + T].date()),
            "index_pct": round((e.spx / s.spx - 1) * 100, 2),
            "tech_pct": round((e.qqq / s.qqq - 1) * 100, 2),
            "yield_bp": round((e.y10 - s.y10) * 1e4, 1),
            "vix_start": round(float(s.vix), 2), "vix_end": round(float(e.vix), 2),
            # plausibility cost of the real path under the calibrated model
            "model_cost": round(float(t_cost.loc[df.index[i0 + 1]:df.index[i0 + T]].sum()), 1),
        }

    last = df.index[-1]
    out = {
        "sample": {"start": str(df.index[0].date()), "end": str(last.date()), "n_days": int(len(df))},
        "params": {k: round(v, 6) for k, v in p.items()},
        "corr": np.round(corr, 4).tolist(),
        "factors": list(eps.columns),
        "initial_state": {"as_of": str(last.date()), "vix": float(df.vix.iloc[-1]), "y10": float(df.y10.iloc[-1])},
        "student_t_nu": nu,
        "budget": budget,
        "eps_cap": eps_cap,
        "raw_daily_cap": raw_cap,
        "ten_day_range": ten_day_range,
        "vix_range": vix_range,
        "tails": tails,
        "budget_backtest": backtest,
        "historical_scenarios": hist,
    }
    Path("rst/calibration.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ["sample", "params", "initial_state", "eps_cap", "raw_daily_cap", "ten_day_range", "vix_range", "budget_backtest"]}, indent=1))
    plot(g_roll, t_roll, gauss_budget, budget, hist, nu)
    for k, v in hist.items():
        print(f"{k:26s} {v['start']}  idx {v['index_pct']:7.2f}%  tech {v['tech_pct']:7.2f}%  "
              f"y {v['yield_bp']:6.1f}bp  VIX {v['vix_start']:5.1f}->{v['vix_end']:5.1f}  cost {v['model_cost']:6.1f}")


def plot(g_roll, t_roll, gauss_budget, budget, hist, nu):
    """Distribution of real 10-day path costs under the Gaussian and Student-t measures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, INK2, GRID, BG = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    plt.rcParams.update({
        "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
        "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
        "axes.titleweight": "bold", "axes.titlesize": 11, "axes.titlecolor": INK, "axes.titlelocation": "left",
    })
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, roll, b, title, col in [
        (axs[0], g_roll, gauss_budget, "Gaussian likelihood: 99% budget is far too tight", "#eb6834"),
        (axs[1], t_roll, budget, f"Student-t likelihood (ν = {nu:.1f}), empirical 99% budget", "#2a78d6"),
    ]:
        x = roll.iloc[::T]
        ax.hist(x.clip(upper=x.quantile(0.995) * 1.3), bins=60, color=col, edgecolor=BG, linewidth=0.5)
        ax.axvline(b, color=INK, lw=1.2, ls="--")
        pct = (x > b).mean() * 100
        ax.text(b, ax.get_ylim()[1] * 0.92, f"  99% budget = {b:.0f}\n  {pct:.1f}% of real windows above",
                color=INK, fontsize=9, va="top")
        ax.set_title(title)
        ax.set_xlabel("Plausibility cost of a real 10-day window (2000-2026)")
    axs[0].set_ylabel("Number of non-overlapping windows")
    fig.tight_layout()
    Path("results").mkdir(exist_ok=True)
    fig.savefig("results/fig0_calibration.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
