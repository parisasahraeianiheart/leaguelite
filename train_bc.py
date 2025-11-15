import argparse
import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from .env import LeagueLiteEnv
from .models import BehaviorCloningPolicy
from .scripted_policy import scripted_expert


DEMO_PATH = "logs/bc_demos.npz"


def generate_demos(episodes: int = 500):
    env = LeagueLiteEnv()
    obs_list, act_list = [], []

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action = scripted_expert(obs)
            next_obs, reward, done, truncated, _ = env.step(action)
            done_flag = done or truncated

            obs_list.append(obs)
            act_list.append(action)

            obs = next_obs
            ep_reward += reward

        print(f"[BC Demo] Episode {ep} | Return: {ep_reward:.2f}")

    os.makedirs("logs", exist_ok=True)
    np.savez(DEMO_PATH, obs=np.array(obs_list), act=np.array(act_list))
    print(f"Saved demos to {DEMO_PATH}")


def train_bc(epochs: int = 20, batch_size: int = 256, lr: float = 3e-4):
    data = np.load(DEMO_PATH)
    obs = data["obs"]
    act = data["act"]

    obs_dim = obs.shape[1]
    act_dim = int(act.max()) + 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BehaviorCloningPolicy(obs_dim, act_dim).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    dataset_size = len(obs)
    idxs = np.arange(dataset_size)

    for epoch in range(epochs):
        np.random.shuffle(idxs)
        losses = []

        for start in range(0, dataset_size, batch_size):
            end = start + batch_size
            batch_idx = idxs[start:end]

            batch_obs = torch.as_tensor(obs[batch_idx], dtype=torch.float32, device=device)
            batch_act = torch.as_tensor(act[batch_idx], dtype=torch.long, device=device)

            logits = model(batch_obs)
            loss = F.cross_entropy(logits, batch_act)

            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        print(f"[BC] Epoch {epoch} | Loss: {np.mean(losses):.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/bc_leaguelite.pt")
    print("Saved BC model to checkpoints/bc_leaguelite.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["generate", "train"], required=True)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    if args.mode == "generate":
        generate_demos(episodes=args.episodes)
    else:
        train_bc(epochs=args.epochs)
