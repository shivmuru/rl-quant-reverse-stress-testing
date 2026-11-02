"""Evaluate the trained adversary against classical reverse-stress-test baselines.

Writes results/metrics.json and the figures in results/.
"""
import json
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rst.env import StressEnv, sample_scales, run_path, POSITION_NAMES
from rst.ppo import ActorCritic, rollout
from rst import baselines as bl

torch.manual_seed(0)
OUT = "results"
# Categorical palette (validated default, light mode) - assigned in fixed order
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.titleweight": "bold", "axes.titlesize": 11, "axes.titlecolor": INK, "axes.titlelocation": "left",
})

env = StressEnv()
ac = ActorCritic(env.obs_dim, env.act_dim)
ac.load_state_dict(torch.load(f"{OUT}/policy.pt"))
ac.eval()


def rl_attack(scales, samples=0):
    """Deterministic policy, or best of `samples` stochastic rollouts per portfolio."""
    if samples == 0:
        _, _, _, R, _, Z = rollout(env, ac, scales, deterministic=True)
        return R.double().sum(0), Z
    B = scales.shape[0]
    sc = scales.repeat_interleave(samples, 0)
    _, _, _, R, _, Z = rollout(env, ac, sc)
    L = R.double().sum(0).view(B, samples)
    v, i = L.max(1)
    return v, Z.view(B, samples, env.cfg.T, 4)[torch.arange(B), i]


def timed(f):
    t = time.time()
    r = f()
    return r, time.time() - t


def trace(scales, Z):
    """Day-by-day factor levels, cumulative loss and loss attribution for one path."""
    env.reset(scales)
    lv = [[env.S.item(), env.Q.item(), env.y.item(), env.v.item()]]
    cum, attr = [0.0], []
    for t in range(env.cfg.T):
        l = env.step_z(Z[:, t])
        lv.append([env.S.item(), env.Q.item(), env.y.item(), env.v.item()])
        cum.append(cum[-1] + l.item())
        attr.append(env.last_loss_by_position[0].tolist())
    return np.array(lv), np.array(cum), np.array(attr), (Z[0] ** 2).sum(-1).numpy()


# --------------------------------------------------------------- base portfolio
base = torch.ones(1, 5, dtype=torch.float64)
res, paths, times = {}, {}, {}
(v, Z), times["Monte Carlo (100k paths)"] = timed(lambda: bl.monte_carlo_worst(env, base))
res["Monte Carlo (100k paths)"] = v.item(); paths["Monte Carlo"] = Z
hist = {}
for name, m in bl.HISTORICAL.items():
    L, k, Zh = bl.historical_rescaled(env, base, m)
    hist[name] = (L.item(), k)
worst_hist = max(hist, key=lambda n: hist[n][0])
res["Historical (worst of 5, rescaled)"] = hist[worst_hist][0]
(v, Z), times["Static worst case (day-1 shock)"] = timed(lambda: bl.gradient_worst(env, base, static=True))
res["Static worst case (day-1 shock)"] = v.item()
(v, Z), times["RL policy (1 rollout)"] = timed(lambda: rl_attack(base))
res["RL policy (1 rollout)"] = v.item(); paths["RL"] = Z
(v, Z), times["RL policy (best of 256)"] = timed(lambda: rl_attack(base, 256))
res["RL policy (best of 256)"] = v.item(); paths["RL best"] = Z
(v, Z), times["Gradient oracle (full path)"] = timed(lambda: bl.gradient_worst(env, base))
res["Gradient oracle (full path)"] = v.item(); paths["Oracle"] = Z

print("\nBase portfolio: loss at the 99% plausibility budget ($m)")
for k, v in res.items():
    print(f"  {k:36s} {v:7.2f}   {times.get(k, float('nan')):7.2f}s")
print("  historical detail:", {k: (round(a, 2), round(b, 2)) for k, (a, b) in hist.items()})

# ------------------------------------------------------ held-out portfolios
g = torch.Generator().manual_seed(12345)  # training used seed 0
N = 100
test = bl_scales = sample_scales(N, g)
held = {}
(mc, _), t_mc = timed(lambda: bl.monte_carlo_worst(env, test, n=20_000, batch=2_000))
hist_l = torch.stack([bl.scenario_path(env, test, m, hist[n][1])[0] for n, m in bl.HISTORICAL.items()]).max(0).values
(st, _), t_st = timed(lambda: bl.gradient_worst(env, test, static=True))
(rl1, _), t_rl1 = timed(lambda: rl_attack(test))
(rl64, _), t_rl64 = timed(lambda: rl_attack(test, 64))
(orc, _), t_orc = timed(lambda: bl.gradient_worst(env, test))
# "One scenario for everyone": the base portfolio's worst path, reused for every book
fixed, _ = run_path(env, test, paths["Oracle"].expand(N, -1, -1))
held_out = {
    "Monte Carlo (20k paths)": (mc, t_mc),
    "Historical (worst of 5, rescaled)": (hist_l, 0.0),
    "Fixed crash scenario (base book's worst)": (fixed, 0.0),
    "Static worst case (day-1 shock)": (st, t_st),
    "RL policy (1 rollout)": (rl1, t_rl1),
    "RL policy (best of 64)": (rl64, t_rl64),
    "Gradient oracle (full path)": (orc, t_orc),
}
print(f"\n{N} held-out portfolios")
held_summary = {}
for k, (v, t) in held_out.items():
    s = {"mean_loss": v.mean().item(), "mean_pct_of_oracle": (v / orc).mean().item() * 100,
         "beats_mc_pct": (v > mc).double().mean().item() * 100,
         "beats_hist_pct": (v > hist_l).double().mean().item() * 100,
         "ms_per_portfolio": t / N * 1000}
    held_summary[k] = s
    print(f"  {k:36s} mean ${s['mean_loss']:6.2f}m  {s['mean_pct_of_oracle']:5.1f}% of oracle  "
          f"beats MC {s['beats_mc_pct']:5.1f}%  {s['ms_per_portfolio']:8.1f} ms/portfolio")

json.dump({"base_portfolio": res, "base_times_s": times,
           "historical_detail": {k: {"loss": a, "scale": b} for k, (a, b) in hist.items()},
           "held_out": held_summary, "budget": env.B_total}, open(f"{OUT}/metrics.json", "w"), indent=2)

# ------------------------------------------------------------------ figures
# 1. Base portfolio comparison (horizontal bars, sorted)
order = sorted(res, key=res.get)
fig, ax = plt.subplots(figsize=(8, 3.6))
cols = [C[0] if k.startswith("RL") else "#a3a29d" for k in order]
ax.barh(order, [res[k] for k in order], color=cols, height=0.62)
for i, k in enumerate(order):
    ax.text(res[k] + 0.8, i, f"${res[k]:.1f}m", va="center", color=INK, fontsize=9)
ax.set_xlabel("10-day loss at equal plausibility (99% likelihood budget), $m")
ax.set_title("Worst plausible loss found, base portfolio")
ax.grid(axis="y", visible=False)
ax.set_xlim(0, max(res.values()) * 1.15)
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_method_comparison.png", dpi=160); plt.close(fig)

# 2. Anatomy of the RL scenario vs Monte Carlo worst
lv_rl, cum_rl, attr_rl, sp_rl = trace(base, paths["RL"])
lv_mc, cum_mc, attr_mc, sp_mc = trace(base, paths["Monte Carlo"])
days = np.arange(env.cfg.T + 1)
fig, axs = plt.subplots(2, 3, figsize=(12, 6.2))
panels = [(0, "Index level", 1.0), (1, "Tech stock level", 1.0), (2, "10y yield (%)", 100.0), (3, "Index implied vol (%)", 100.0)]
for ax, (j, title, mult) in zip(axs.flat[:4], panels):
    ax.plot(days, lv_rl[:, j] * mult, color=C[0], lw=2, marker="o", ms=4, label="RL scenario")
    ax.plot(days, lv_mc[:, j] * mult, color=C[1], lw=2, marker="o", ms=4, label="Monte Carlo worst")
    ax.set_title(title); ax.set_xlabel("Day")
axs.flat[0].legend(frameon=False, fontsize=9)
ax = axs.flat[4]
ax.plot(days, cum_rl, color=C[0], lw=2, marker="o", ms=4, label="RL scenario")
ax.plot(days, cum_mc, color=C[1], lw=2, marker="o", ms=4, label="Monte Carlo worst")
ax.set_title("Cumulative loss ($m)"); ax.set_xlabel("Day")
ax = axs.flat[5]
ax.bar(days[1:] - 0.2, sp_rl, width=0.4, color=C[0], label="RL scenario")
ax.bar(days[1:] + 0.2, sp_mc, width=0.4, color=C[1], label="Monte Carlo worst")
ax.set_title("Plausibility budget spent per day (‖z‖²)"); ax.set_xlabel("Day")
fig.suptitle("How the RL adversary builds its scenario (base portfolio, same plausibility budget)",
             x=0.01, ha="left", fontweight="bold", color=INK)
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_scenario_anatomy.png", dpi=160); plt.close(fig)

# 3. Loss attribution by position
fig, ax = plt.subplots(figsize=(8, 3.4))
y = np.arange(5)
ax.barh(y + 0.2, attr_rl.sum(0), height=0.38, color=C[0], label=f"RL scenario (total ${cum_rl[-1]:.1f}m)")
ax.barh(y - 0.2, attr_mc.sum(0), height=0.38, color=C[1], label=f"Monte Carlo worst (total ${cum_mc[-1]:.1f}m)")
ax.set_yticks(y, POSITION_NAMES); ax.invert_yaxis()
ax.axvline(0, color=INK2, lw=0.8); ax.grid(axis="y", visible=False)
ax.set_xlabel("Loss contribution, $m (negative = gain)")
ax.set_title("Where the loss comes from")
ax.legend(frameon=False, fontsize=9, loc="lower right")
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_attribution.png", dpi=160); plt.close(fig)

# 4. Held-out generalization: RL vs oracle, MC vs oracle
fig, ax = plt.subplots(figsize=(6, 5.4))
lo = min(0.0, mc.min().item(), fixed.min().item()) * 1.08
lim = [lo, orc.max().item() * 1.08]
ax.plot(lim, lim, color=INK2, lw=0.8, ls="--")
ax.text(lim[1] * 0.97, lim[1] * 0.90, "= oracle", ha="right", color=INK2, fontsize=9)
ax.axhline(0, color=INK2, lw=0.6)
ax.scatter(orc, fixed, s=26, color=C[2], edgecolor="#fcfcfb", lw=1, label="Fixed crash scenario")
ax.scatter(orc, mc, s=26, color=C[1], edgecolor="#fcfcfb", lw=1, label="Monte Carlo worst (20k)")
ax.scatter(orc, rl1, s=26, color=C[0], edgecolor="#fcfcfb", lw=1, label="RL policy (1 rollout)")
ax.set_xlim(0, lim[1]); ax.set_ylim(lim)
ax.set_xlabel("Gradient oracle loss, $m"); ax.set_ylabel("Method loss, $m")
ax.set_title(f"{N} unseen long/short books")
ax.legend(frameon=False, fontsize=9, loc="upper left")
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_generalization.png", dpi=160); plt.close(fig)

# 5. Training curve
h = json.load(open(f"{OUT}/train_history.json"))
fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(h, color=C[0], lw=2)
ax.set_xlabel("PPO iteration (1,024 random portfolios each)"); ax.set_ylabel("Mean episode loss, $m")
ax.set_title("Training: adversary loss on random portfolios")
fig.tight_layout(); fig.savefig(f"{OUT}/fig5_training.png", dpi=160); plt.close(fig)
print("figures written")
