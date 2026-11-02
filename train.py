"""Train the PPO adversary on randomized portfolios and save it to results/."""
import json
import torch
from rst.ppo import train

if __name__ == "__main__":
    ac, hist = train(iters=1200)
    torch.save(ac.state_dict(), "results/policy.pt")
    json.dump(hist, open("results/train_history.json", "w"))
