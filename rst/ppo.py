"""Minimal PPO (written from scratch) for the batched stress environment.

Episodes are synchronized and have fixed length T, so rollouts are plain (T, B) tensors.
"""
import time
import torch
import torch.nn as nn
from .env import StressEnv, sample_scales


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()

        def mlp(out):
            return nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, out))
        self.pi = mlp(act_dim)
        self.v = mlp(1)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

    def dist(self, obs):
        return torch.distributions.Normal(self.pi(obs), self.log_std.exp())

    def value(self, obs):
        return self.v(obs).squeeze(-1)


def rollout(env: StressEnv, ac: ActorCritic, scales, deterministic=False):
    obs = env.reset(scales)
    O, A, LP, R, V, Zs = [], [], [], [], [], []
    for _ in range(env.cfg.T):
        with torch.no_grad():
            d = ac.dist(obs)
            a = d.mean if deterministic else d.sample()
            lp = d.log_prob(a).sum(-1)
            v = ac.value(obs)
        nobs, loss, done, z = env.step(a)
        O.append(obs); A.append(a); LP.append(lp); V.append(v); R.append(loss.float()); Zs.append(z)
        obs = nobs
    return (torch.stack(O), torch.stack(A), torch.stack(LP), torch.stack(R),
            torch.stack(V), torch.stack(Zs, 1))


def train(iters=600, n_envs=1024, lr=3e-4, epochs=6, minibatch=2048, clip=0.2,
          lam=0.95, ent_coef=1e-3, reward_scale=0.1, seed=0, log_every=25, log=print):
    torch.manual_seed(seed)
    env = StressEnv()
    ac = ActorCritic(env.obs_dim, env.act_dim)
    opt = torch.optim.Adam(ac.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    history = []
    t0 = time.time()
    for it in range(iters):
        scales = sample_scales(n_envs, g)
        O, A, LP, R, V, _ = rollout(env, ac, scales)
        R = R * reward_scale
        T = R.shape[0]
        # GAE (gamma = 1: we care about total loss over the horizon)
        adv = torch.zeros_like(R)
        last = torch.zeros(n_envs)
        for t in reversed(range(T)):
            nv = V[t + 1] if t + 1 < T else torch.zeros(n_envs)
            delta = R[t] + nv - V[t]
            last = delta + lam * last
            adv[t] = last
        ret = adv + V
        O, A, LP, adv, ret = (x.reshape(T * n_envs, *x.shape[2:]) for x in (O, A, LP, adv, ret))
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        N = O.shape[0]
        for _ in range(epochs):
            perm = torch.randperm(N)
            for s in range(0, N, minibatch):
                idx = perm[s:s + minibatch]
                d = ac.dist(O[idx])
                nlp = d.log_prob(A[idx]).sum(-1)
                ratio = (nlp - LP[idx]).exp()
                pg = -torch.min(ratio * adv[idx], ratio.clamp(1 - clip, 1 + clip) * adv[idx]).mean()
                vl = 0.5 * (ac.value(O[idx]) - ret[idx]).pow(2).mean()
                ent = d.entropy().sum(-1).mean()
                loss = pg + 0.5 * vl - ent_coef * ent
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), 0.5)
                opt.step()
        ep_loss = (R.sum(0) / reward_scale).mean().item()
        history.append(ep_loss)
        if it % log_every == 0 or it == iters - 1:
            log(f"iter {it:4d}  mean episode loss ${ep_loss:6.2f}m  "
                f"std {ac.log_std.exp().mean():.3f}  {time.time()-t0:5.0f}s")
    return ac, history
