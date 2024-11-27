import heapq
import numpy as np
import cv2 as cv

from scipy.interpolate import CubicSpline
from scipy.interpolate import BPoly

class Node:
    def __init__(self, position, cost, heuristic):
        self.position = position
        self.cost = cost
        self.heuristic = heuristic
        self.total_cost = cost + heuristic
        self.parent = None

    @classmethod
    def from_coordinates(cls, row, col, cost, heuristic):
        return cls((row, col), cost, heuristic)

    def __lt__(self, other):
        return self.total_cost < other.total_cost

def heuristic(node, goal):
    return abs(node.position[0] - goal[0]) + abs(node.position[1] - goal[1])

def astar(grid, start, goal):
    # rows, cols = len(grid), len(grid[0])
    rows, cols = grid.shape[0], grid.shape[1]
    open_set = []
    closed_set = set()

    start_node = Node.from_coordinates(*start, 0, heuristic(Node.from_coordinates(*start, 0, 0), goal))
    heapq.heappush(open_set, start_node)

    while open_set:
        current = heapq.heappop(open_set)

        if current.position == goal:
            path = [current.position]
            while current.parent:
                current = current.parent
                path.append(current.position)
            return path[::-1]

        closed_set.add(current.position)

        for i, j in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, 1), (1, -1), (-1, -1)]:
            neighbor_row, neighbor_col = current.position[0] + i, current.position[1] + j
            neighbor_position = (neighbor_row, neighbor_col)

            if (
                0 <= neighbor_row < rows
                and 0 <= neighbor_col < cols
                and grid[neighbor_row][neighbor_col] == 0
                and neighbor_position not in closed_set
            ):
                neighbor = Node.from_coordinates(
                    neighbor_row,
                    neighbor_col,
                    current.cost + 1,
                    heuristic(Node.from_coordinates(neighbor_row, neighbor_col, 0, 0), goal),
                )

                if neighbor_position not in (node.position for node in open_set):
                    neighbor.parent = current
                    heapq.heappush(open_set, neighbor)
                else:
                    existing_neighbor = next(node for node in open_set if node.position == neighbor_position)
                    if neighbor.total_cost < existing_neighbor.total_cost:
                        existing_neighbor.parent = current
                        existing_neighbor.cost = neighbor.cost
                        existing_neighbor.total_cost = neighbor.total_cost

    return None  # Unable to reach the goal

def interpolate_curve(astar_path, kind = 3, point_num = 100):
    points_x, points_y = [], []
    for p in astar_path:
        points_x.append(p[0])
        points_y.append(p[1])
    
    x = np.array(points_x)
    y = np.array(points_y)
    t = np.linspace(0, 1, len(x))
    t_interp = np.linspace(0, 1, point_num)
    if kind == 3:
        cs_x = CubicSpline(t, x)
        cs_y = CubicSpline(t, y)
        x_interp3 = cs_x(t_interp)
        y_interp3 = cs_y(t_interp)
        return list(x_interp3), list(y_interp3)
    elif kind == 5:
        coeffs_x = np.polyfit(t, x, 5)
        coeffs_y = np.polyfit(t, y, 5)
        spline_x = BPoly.from_derivatives(t, [[np.polyval(coeffs_x, ti)] for ti in t])
        spline_y = BPoly.from_derivatives(t, [[np.polyval(coeffs_y, ti)] for ti in t])
        x_interp5 = spline_x(t_interp)
        y_interp5 = spline_y(t_interp)
        return list(x_interp5), list(y_interp5)
    else:
        print("kind only is 3 or 5!")

def astar_search(global_map, start_pose_visual, target_pose_visual):
    origin_shape = global_map.shape[0] #ToDo: support rectangular maps
    start_pose_visual[0] = int(round(start_pose_visual[0] * 160/origin_shape))
    target_pose_visual[0] = int(round(target_pose_visual[0] * 160/origin_shape))
    start_pose_visual[1] = int(round(start_pose_visual[1] * 160/origin_shape))
    target_pose_visual[1] = int(round(target_pose_visual[1] * 160/origin_shape))
    global_map = cv.resize(global_map, dsize=(160, 160), interpolation=cv.INTER_CUBIC)
    global_map[global_map < 255] = 1
    global_map[global_map == 255] = 0
    # Inflate obstacles based on vehicle size
    dilated_global_map = np.zeros_like(global_map)
    obstacle_indices = np.where(global_map == 1)
    for x, y in zip(*obstacle_indices):
        min_x, max_x = max(0, x-5), min(dilated_global_map.shape[0], x+6) # The actual length and the actual ratio of the (160, 160) image are about 1:10, that is, 1m is equal to 10 pixels.
        min_y, max_y = max(0, y-5), min(dilated_global_map.shape[1], y+6)

        m, n = np.mgrid[min_x:max_x, min_y:max_y]
        mask = ((m - start_pose_visual[0])**2 + (n - start_pose_visual[1])**2) >= 36

        dilated_global_map[m[mask], n[mask]] = 1
    path = astar(dilated_global_map, tuple(start_pose_visual), tuple(target_pose_visual))

    if path is None or len(path) == 1:
        return [[0] * 100, [0] * 100]
    else:
        path_x, path_y = interpolate_curve(path, 3, 100)
        for i in range(len(path_x)):
            path_x[i] = int(round(path_x[i] * origin_shape/160))
            path_y[i] = int(round(path_y[i] * origin_shape/160))
        return [path_x, path_y]