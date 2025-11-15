# leaguelite
Multi-Agent RL &amp; Imitation Learning for MOBA-Style Game AI

# LeagueLite: Multi-Agent RL & Imitation Learning for MOBA-Style Game AI

LeagueLite is a small research-style project that explores **reinforcement learning** and
**imitation learning** in a simplified **MOBA-style environment**.

The goal is to demonstrate:

- Training **autonomous agents** to understand game state and make decisions.
- Using **on-policy RL (PPO)** and **imitation learning (behavior cloning)**.
- Building **reusable training & evaluation pipelines**.
- Generating **predictive telemetry features** from gameplay logs.
- Applying basic **responsible-AI / UX principles** like stability and interpretability.

This project is meant as a portfolio piece for game AI / ML roles (e.g. ML Bots / Game AI teams).

---

## Environment overview

- 2D grid map.
- Our agent: HP, mana, basic attack, heal, move.
- Enemy: scripted policy (aggressive chaser).
- Objective: stay alive, deal damage, avoid dying.
- Observations: normalized game state vector (positions, HP, distance, cooldown).
- Actions:
  - `0` = stay
  - `1` = move toward enemy
  - `2` = move away
  - `3` = attack
  - `4` = heal

Rewards:
- Positive for damage dealt, surviving, strategic positioning.
- Negative for taking damage and dying.
- Optional reward shaping for distance, safe positioning.

---

## Key components

- `env.py` – MOBA-style grid environment (`LeagueLiteEnv`).
- `telemetry.py` – episode logging to CSV for later analysis.
- `models.py` – PyTorch models for PPO (actor-critic) and behavior cloning.
- `scripted_policy.py` – simple “expert” policy used to generate demonstrations.
- `train_ppo.py` – PPO training loop on the environment.
- `train_bc.py` – behavior cloning from scripted policy trajectories.

---

## Installation

```bash
git clone https://github.com/<your-username>/LeagueLite.git
cd LeagueLite
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
