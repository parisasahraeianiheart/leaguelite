import numpy as np
import gym
from gym import spaces


class LeagueLiteEnv(gym.Env):
    """
    Simplified MOBA-style 2D grid environment.

    Agent controls a single champion:
      - move toward / away from enemy
      - stay
      - attack
      - heal

    Observation: vector of normalized game state.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, grid_size=7, max_steps=100):
        super().__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps

        # Agent & enemy stats
        self.max_hp = 100
        self.max_mana = 50

        # Observation: [ax, ay, ex, ey, a_hp, e_hp, a_mana, dist, attack_cd]
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(9,),
            dtype=np.float32,
        )

        # Actions: 0=stay, 1=move_toward, 2=move_away, 3=attack, 4=heal
        self.action_space = spaces.Discrete(5)

        self.reset()

    # ---------- Core API ----------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0

        # Agent starts bottom-left, enemy top-right
        self.agent_pos = np.array([0, 0], dtype=np.int32)
        self.enemy_pos = np.array([self.grid_size - 1, self.grid_size - 1], dtype=np.int32)

        self.agent_hp = self.max_hp
        self.enemy_hp = self.max_hp
        self.agent_mana = self.max_mana // 2
        self.attack_cooldown = 0

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        self.steps += 1

        # Cooldown decay
        if self.attack_cooldown > 0:
            self.attack_cooldown -= 1

        # Enemy scripted action
        self._enemy_step()

        # Agent action
        reward = 0.0
        done = False

        if action == 1:
            self._move_toward_enemy()
        elif action == 2:
            self._move_away_from_enemy()
        elif action == 3:
            reward += self._agent_attack()
        elif action == 4:
            reward += self._agent_heal()

        # Distance-based shaping (encourage healthy positioning)
        dist = self._distance()
        # small reward for staying in mid-range
        reward += -0.01 * abs(dist - 2.0)

        # Terminal conditions
        if self.enemy_hp <= 0:
            reward += 5.0
            done = True

        if self.agent_hp <= 0:
            reward -= 5.0
            done = True

        if self.steps >= self.max_steps:
            done = True

        obs = self._get_obs()
        info = {
            "agent_hp": self.agent_hp,
            "enemy_hp": self.enemy_hp,
            "agent_mana": self.agent_mana,
            "distance": dist,
        }
        return obs, reward, done, False, info

    # ---------- Internal helpers ----------

    def _get_obs(self):
        ax, ay = self.agent_pos / (self.grid_size - 1)
        ex, ey = self.enemy_pos / (self.grid_size - 1)
        a_hp = self.agent_hp / self.max_hp
        e_hp = self.enemy_hp / self.max_hp
        a_mana = self.agent_mana / self.max_mana
        dist = self._distance() / (np.sqrt(2) * (self.grid_size - 1))
        cd = self.attack_cooldown / 3.0
        return np.array([ax, ay, ex, ey, a_hp, e_hp, a_mana, dist, cd], dtype=np.float32)

    def _distance(self):
        return float(np.linalg.norm(self.agent_pos - self.enemy_pos, ord=1))

    def _enemy_step(self):
        # Simple aggressive script: move toward agent, sometimes attack
        if self.enemy_hp <= 0:
            return

        # Move closer in Manhattan distance
        direction = np.sign(self.agent_pos - self.enemy_pos)
        self.enemy_pos += direction
        self.enemy_pos = np.clip(self.enemy_pos, 0, self.grid_size - 1)

        # If adjacent: enemy attacks
        if self._distance() <= 1.0:
            dmg = 8
            self.agent_hp = max(0, self.agent_hp - dmg)

    def _move_toward_enemy(self):
        direction = np.sign(self.enemy_pos - self.agent_pos)
        self.agent_pos += direction
        self.agent_pos = np.clip(self.agent_pos, 0, self.grid_size - 1)

    def _move_away_from_enemy(self):
        direction = np.sign(self.agent_pos - self.enemy_pos)
        self.agent_pos += direction
        self.agent_pos = np.clip(self.agent_pos, 0, self.grid_size - 1)

    def _agent_attack(self):
        if self.attack_cooldown > 0:
            return -0.1  # wasted action

        if self._distance() <= 1.0:
            dmg = 12
            self.enemy_hp = max(0, self.enemy_hp - dmg)
            self.attack_cooldown = 2
            return 1.0  # reward for successful hit
        return -0.1

    def _agent_heal(self):
        if self.agent_mana < 5:
            return -0.1
        self.agent_mana -= 5
        heal = 10
        old_hp = self.agent_hp
        self.agent_hp = min(self.max_hp, self.agent_hp + heal)
        return 0.2 if self.agent_hp > old_hp else -0.1

    # Optional text render
    def render(self, mode="human"):
        print(
            f"Step {self.steps} | Agent HP {self.agent_hp}, Enemy HP {self.enemy_hp}, "
            f"Agent Pos {self.agent_pos}, Enemy Pos {self.enemy_pos}"
        )
