"""Find a short reference band around obstacles observed in the current scan."""

from heapq import heappop, heappush

import numpy as np


def lidar_reference(start, goal, points, radius, segments):
    """Return a sampled local route, or None if it makes no progress."""
    resolution = 0.04
    extent = 50
    size = 2 * extent + 1
    origin = start[:2] - extent * resolution
    source = (extent, extent)

    blocked = np.zeros((size, size), dtype=bool)
    nearby = points[np.all((points >= origin - radius)
                           & (points <= origin + (size - 1) * resolution + radius), axis=1)]
    cells = np.rint((nearby - origin) / resolution).astype(int)
    padding = int(np.ceil(radius / resolution)) + 1
    for dx in range(-padding, padding + 1):
        for dy in range(-padding, padding + 1):
            indices = cells + (dx, dy)
            in_grid = np.all((indices >= 0) & (indices < size), axis=1)
            indices = indices[in_grid]
            centers = origin + resolution * indices
            hits = np.sum((centers - nearby[in_grid]) ** 2, axis=1) <= radius ** 2
            blocked[tuple(indices[hits].T)] = True
    if blocked[source]:
        return None

    direction = goal - start[:2]
    span = np.max(np.abs(direction))
    if span == 0:
        return None
    scale = min(1.0, (extent - 1) * resolution / span)
    target = np.rint((start[:2] + direction * scale - origin) / resolution).astype(int)
    tx, ty = map(int, target)
    neighbors = (
        (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
        (1, 1, 2**0.5), (-1, 1, 2**0.5), (1, -1, 2**0.5), (-1, -1, 2**0.5),
    )
    frontier = [(0.0, 0.0, source)]
    costs = {source: 0.0}
    parents = {}
    nearest = source
    best_distance = float('inf')
    while frontier:
        _, cost, cell = heappop(frontier)
        if cost != costs[cell]:
            continue
        x, y = cell
        distance = (x - tx) ** 2 + (y - ty) ** 2
        if distance < best_distance:
            nearest, best_distance = cell, distance
        if distance == 0:
            break
        for dx, dy, length in neighbors:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < size and 0 <= ny < size) or blocked[nx, ny]:
                continue
            if blocked[nx, y] or blocked[x, ny]:
                continue  # A diagonal may not cut the corner of an occupied cell.
            neighbor = (nx, ny)
            next_cost = cost + length
            if next_cost < costs.get(neighbor, float('inf')):
                costs[neighbor] = next_cost
                parents[neighbor] = cell
                heuristic = ((nx - tx) ** 2 + (ny - ty) ** 2) ** 0.5
                heappush(frontier, (next_cost + heuristic, next_cost, neighbor))

    if nearest == source:
        return None
    path = [nearest]
    while path[-1] != source:
        path.append(parents[path[-1]])
    xy = origin + resolution * np.array(path[::-1])
    length = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    if length[-1] < 0.15:
        return None
    samples = np.linspace(0, min(0.8, length[-1]), segments + 1)
    route = np.column_stack([np.interp(samples, length, axis) for axis in xy.T])
    tangent = np.gradient(route, axis=0)
    yaw = np.unwrap(np.arctan2(tangent[:, 1], tangent[:, 0]))
    yaw += 2 * np.pi * np.round((start[2] - yaw[0]) / (2 * np.pi))
    yaw[0] = start[2]
    return np.column_stack((route, yaw))
