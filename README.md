# RL Reverse Stress Testing

**Using reinforcement learning to find the most damaging *plausible* market path for a nonlinear portfolio.**

Classical stress testing asks *"what do we lose if the market does X?"*. Reverse stress testing asks the opposite question: *"what is the worst thing the market can plausibly do to this book?"*. This project trains a PPO agent, implemented from scratch in PyTorch, to answer that question for a book of equities, bonds and options. It then compares the agent against the standard toolkit (Monte Carlo, rescaled historical crises, the static worst case) under an identical plausibility constraint.

## The problem

**Market.** A stylized 4-factor model, simulated daily over a 10-day horizon:

| Factor | Dynamics |
|---|---|
| Equity index | log-normal, daily vol = implied vol / √252 |
| Tech stock | log-normal, 1.5× the index's vol |
| 10y yield | arithmetic, ~7bp/day |
| Index implied vol | log-normal, 7%/day vol-of-vol |

Shocks are correlated (equity–vol −0.75, index–tech +0.80, etc.), and the model is **heteroskedastic**: the daily volatility of the equity factors scales with the current implied-vol level. This makes the *order* of shocks matter. Once vol has risen, a large equity drop costs less likelihood.

**Portfolio.** A deliberately nonlinear, path-dependent book:

- Long index and long tech stock
- Long 10y bonds (duration + convexity)
- **Short index ATM straddle, delta-hedged daily** (short gamma, short vega)
- **Short 25Δ puts on the tech stock**, unhedged

**Plausibility budget.** The adversary picks standardized shocks `z_t ∈ ℝ⁴` for each day. A path's plausibility is its Gaussian negative log-likelihood, `Σ_t ‖z_t‖²`, and every method must keep this inside the **99% χ² quantile with 40 degrees of freedom** (≈63.7). Every scenario reported here is therefore a 1-in-100 event by the model's own likelihood, so all methods are compared on equal footing.

## Methods compared

| Method | What it does |
|---|---|
| **Monte Carlo** | Simulate 100k paths, keep those inside the budget, report the worst |
| **Historical (rescaled)** | Five stylized 10-day crisis moves (2008, 2015, 2018, 2020, 2022), scaled up or down to sit exactly on the budget |
| **Static worst case** | The textbook "maximum loss on the ellipsoid": the whole budget spent as one day-1 shock |
| **RL agent (PPO)** | Sequential policy: each day it picks a shock direction and how much of the remaining budget to spend |
| **Gradient oracle** | Multi-start Adam through the differentiable simulator over the full path. Used as an upper-bound reference; requires a differentiable pricer and a fresh optimization per portfolio |

The RL agent is trained on **randomized long/short books** (each position's size drawn from base × U[−1.5, 1.5]). One trained policy can therefore stress-test a portfolio it has never seen, in a single forward pass, with no gradients through the pricer.

## Results

All numbers are 10-day losses in $m at the same 99% plausibility budget, produced by `evaluate.py`. Full output is in `results/metrics.json`.

### Base portfolio

![Method comparison](results/fig1_method_comparison.png)

| Method | Worst loss found | Time |
|---|---|---|
| Static worst case (day-1 shock) | $20.4m | 9.7 s |
| Monte Carlo (100k paths) | $31.9m | 0.5 s |
| Historical, worst of 5 (rescaled) | $40.1m | — |
| **RL agent (1 rollout)** | **$76.7m** | **0.01 s** |
| Gradient oracle (full path) | $76.7m | 7.9 s |

The agent finds **2.4× the Monte Carlo worst case** and **1.9× the worst rescaled historical crisis**, matching the oracle to within 0.1%.

### Why: sequencing

![Scenario anatomy](results/fig2_scenario_anatomy.png)

The agent learned to **push implied vol up first and crash equities second**. Under the heteroskedastic model, once vol is high a large equity drop "costs" less likelihood, so spending budget on vol early buys a much larger crash later. Meanwhile the daily-hedged short straddle bleeds gamma losses every day. A static day-1 shock can't use this, and Monte Carlo almost never samples such a coordinated path.

![Loss attribution](results/fig3_attribution.png)

### Generalization to unseen books

The agent was trained on random long/short books and tested on 100 books it never saw (different seed). Because positions can be flipped, a single "crash" scenario doesn't work: for many books it produces a *gain*.

![Generalization](results/fig4_generalization.png)

| Method (100 held-out books) | Mean loss | Mean % of oracle | Time per book |
|---|---|---|---|
| Fixed crash scenario (base book's worst path) | −$0.2m | — | — |
| Historical, worst of 5 (rescaled) | $8.1m | — | — |
| Static worst case | $10.3m | 36% | 134 ms |
| Monte Carlo (20k paths) | $12.8m | 44% | 74 ms |
| **RL agent (1 rollout)** | **$30.6m** | **97%** (median 99%, worst 71%) | **0.2 ms** |
| Gradient oracle | $31.2m | 100% | 136 ms |

One trained policy reaches **97% of the oracle on unseen portfolios, about 600× faster per portfolio**. It beats the Monte Carlo worst case on all 100 books and needs no gradients through the pricer, so the same approach would work with black-box pricing libraries where the oracle can't run.

![Training curve](results/fig5_training.png)

## Repository layout

```
rst/
  env.py         # batched, differentiable market + portfolio simulator
  ppo.py         # PPO actor-critic, GAE, clipped objective (from scratch)
  baselines.py   # Monte Carlo, historical, static worst case, gradient oracle
train.py         # trains the agent  -> results/policy.pt
evaluate.py      # runs every method -> results/metrics.json + figures
```

## Running it

```bash
pip install -r requirements.txt
python train.py      # ~10 min on a laptop CPU
python evaluate.py   # ~3 min
```

## Limitations and next steps

- **The model is stylized, not calibrated.** Correlations, vols and the historical scenario magnitudes are hand-set to be realistic, not estimated from data. Next step: calibrate to daily SPX, QQQ, DGS10 and VIX history (correlations from standardized returns, vol-of-vol and the vol-to-realized-vol link from VIX).
- **Gaussian innovations.** Fat tails (Student-t) would change both the budget and which paths are plausible.
- **No vol mean-reversion.** The agent exploits this: its worst path pushes implied vol very high. A mean-reverting vol process would make that path less plausible, so the agent's scenario partly reflects a weakness of the model.
- **Black-Scholes repricing with a flat vol surface.** Adding skew dynamics would matter for the short puts.
