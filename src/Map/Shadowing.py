import numpy as np
import os
import tqdm
from src.Map.Map import load_map


def bresenham(x0, y0, x1, y1, obstacles, shadow_map):
    if obstacles[y0, x0]:
        return
    x_dist = abs(x0 - x1)
    y_dist = -abs(y0 - y1)
    x_step = 1 if x1 > x0 else -1
    y_step = 1 if y1 > y0 else -1

    error = x_dist + y_dist

    # shadowed = False
    shadow_map[y0, x0] = False

    while x0 != x1 or y0 != y1:
        if 2 * error - y_dist > x_dist - 2 * error:
            # horizontal step
            error += y_dist
            x0 += x_step
        else:
            # vertical step
            error += x_dist
            y0 += y_step

        if obstacles[y0, x0]:
            # shadowed = True
            return

        # if shadowed:
        shadow_map[y0, x0] = False


def calculate_shadowing(map_path, save_as):
    total_map = load_map(map_path)
    obstacles = total_map.obstacles
    size = total_map.obstacles.shape[0]
    total = size * size

    total_shadow_map = np.ones((size, size, size, size), dtype=bool)
    with tqdm.tqdm(total=total) as pbar:
        for i, j in np.ndindex(total_map.obstacles.shape):
            shadow_map = np.ones((size, size), dtype=bool)

            for x in range(size):
                bresenham(i, j, x, 0, obstacles, shadow_map)
                bresenham(i, j, x, size - 1, obstacles, shadow_map)
                bresenham(i, j, 0, x, obstacles, shadow_map)
                bresenham(i, j, size - 1, x, obstacles, shadow_map)

            total_shadow_map[j, i] = shadow_map
            pbar.update(1)

    np.save(save_as, total_shadow_map)
    return total_shadow_map


def load_or_create_shadowing(map_path):
    shadow_file_name = os.path.splitext(map_path)[0] + "_shadowing.npy"
    if os.path.exists(shadow_file_name):
        return np.load(shadow_file_name)
    else:
        return calculate_shadowing(map_path, shadow_file_name)
