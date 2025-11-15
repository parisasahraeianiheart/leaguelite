import numpy as np


def scripted_expert(obs: np.ndarray, grid_size: int = 7) -> int:
    """
    Very simple heuristic "expert" for behavior cloning data.

    Obs: [ax, ay, ex, ey, a_hp, e_hp, a_mana, dist, cd]
    """
    ax, ay, ex, ey, a_hp, e_hp, a_mana, dist, cd = obs

    # Prefer attacking if adjacent and cooldown ready
    if dist < (1.5 / (np.sqrt(2) * (grid_size - 1))) and cd < 0.01:
        return 3  # attack

    # If low HP, heal when mana available
    if a_hp < 0.4 and a_mana > 0.2:
        return 4  # heal

    # If enemy far away, move toward
    if dist > 0.5:
        return 1  # move toward

    # Otherwise, keep some distance
    return 2  # move away
