import argparse
import os
import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical

from .env import LeagueLiteEnv
from .models import ActorCritic
from .telemetry import TelemetryLogger


def run_ppo(episodes: int = 1000, gamma=0.99, lam=0.95, clip_ratio=0.2,
           lr=3e-4, update_every=1024, epochs=4, batch_size=256):

    env = LeagueLiteEnv()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActorCritic(obs_dim, act_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    logger = TelemetryLogger(prefix="ppo")

    obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []

    def compute_advantages(rewards, values, dones, last_value):
        adv = []
        gae = 0.0
        values = values + [last_value]
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            adv.insert(0, gae)
        returns = [a + v for a, v in zip(adv, values[:-1])]
        return np.array(adv, dtype=np.float32), np.array(returns, dtype=np.float32)

    step_count = 0
    for ep in range(episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        done = False

        while not done:
            step_count += 1
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits, value = model(obs_t)
            dist = Categorical(logits=logits)
            action = dist.sample()
            logp = dist.log_prob(action)

            next_obs, reward, done, truncated, info = env.step(action.item())
            done_flag = done or truncated

            # store
            obs_buf.append(obs)
            act_buf.append(action.item())
            logp_buf.append(logp.item())
            rew_buf.append(reward)
            val_buf.append(value.item())
            done_buf.append(float(done_flag))

            logger.log_step(
                {
                    "episode": ep,
                    "step": step_count,
                    "reward": reward,
                    "done": done_flag,
                    "agent_hp": info["agent_hp"],
                    "enemy_hp": info["enemy_hp"],
                    "distance": info["distance"],
                }
            )

            obs = next_obs
            ep_reward += reward

            # Update PPO when buffer is large enough
            if len(obs_buf) >= update_every:
                with torch.no_grad():
                    last_obs_t = torch.as_tensor(
                        obs, dtype=torch.float32, device=device
                    ).unsqueeze(0)
                    _, last_val = model(last_obs_t)
                    last_val = last_val.item()

                adv, ret = compute_advantages(
                    rew_buf, val_buf, done_buf, last_val
                )

                obs_tensor = torch.as_tensor(np.array(obs_buf), dtype=torch.float32, device=device)
                act_tensor = torch.as_tensor(np.array(act_buf), dtype=torch.int64, device=device)
                logp_old_tensor = torch.as_tensor(np.array(logp_buf), dtype=torch.float32, device=device)
                adv_tensor = torch.as_tensor(adv, dtype=torch.float32, device=device)
                ret_tensor = torch.as_tensor(ret, dtype=torch.float32, device=device)

                adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)

                dataset_size = len(obs_buf)
                idxs = np.arange(dataset_size)

                for _ in range(epochs):
                    np.random.shuffle(idxs)
                    for start in range(0, dataset_size, batch_size):
                        end = start + batch_size
                        batch_idx = idxs[start:end]

                        batch_obs = obs_tensor[batch_idx]
                        batch_act = act_tensor[batch_idx]
                        batch_logp_old = logp_old_tensor[batch_idx]
                        batch_adv = adv_tensor[batch_idx]
                        batch_ret = ret_tensor[batch_idx]

                        logits, value = model(batch_obs)
                        dist = Categorical(logits=logits)
                        logp = dist.log_prob(batch_act)

                        ratio = torch.exp(logp - batch_logp_old)
                        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * batch_adv
                        actor_loss = -(torch.min(ratio * batch_adv, clip_adv)).mean()

                        value_loss = ((batch_ret - value.squeeze()) ** 2).mean()
                        entropy_loss = dist.entropy().mean()

                        loss = actor_loss + 0.5 * value_loss - 0.01 * entropy_loss

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []

        print(f"Episode {ep} | Return: {ep_reward:.2f}")

    logger.close()
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/ppo_leaguelite.pt")
    print("Saved PPO model to checkpoints/ppo_leaguelite.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    args = parser.parse_args()
    run_ppo(episodes=args.episodes)
