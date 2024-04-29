import numpy as np
import time
from heuristics import path_distance, euclidean_heuristic
from astar import astar

def is_collision(point, obstacle_map):
    x, y = point
    return obstacle_map[x, y] == 1

def edge_collision(start, end, obstacle_map):
    # Extracting x, y coordinates of start and end points
    x1, y1 = start
    x2, y2 = end

    # Calculate the number of steps needed to iterate from start to end
    steps = max(abs(x2 - x1), abs(y2 - y1))
    if steps == 0:
        return is_collision(start, obstacle_map)

    points = zip(np.linspace(x1, x2, steps + 1, dtype=int),
                 np.linspace(y1, y2, steps + 1, dtype=int))

    for point in points:
        if is_collision(point, obstacle_map):
            return True
    return False



def generate_random_points(map_size, obstacle_map, num_points):
    points = []
    while len(points) < num_points:
        point = (np.random.randint(map_size[0]), np.random.randint(map_size[1]))
        if not is_collision(point, obstacle_map):
            points.append(point)

    return points


def find_k_nearest_neighbors(point, points, k):
    distances = [(np.linalg.norm(np.array(point) - np.array(p)), p) for p in points]
    distances.sort()
    return [p for _, p in distances[:k]]


def build_roadmap(start, end, map_size, obstacle_map, num_points, k, reconnect = False):
    points = generate_random_points(map_size, obstacle_map, num_points)
    points.append(start)
    points.append(end)
    roadmap = {}
    for point in points:
        neighbors = find_k_nearest_neighbors(point, points, k)
        roadmap[point] = [neighbor for neighbor in neighbors if not edge_collision(point, neighbor, obstacle_map)]
        
    if reconnect:
        for point in points:
            for neighbor in find_k_nearest_neighbors(point, points, k):
                if neighbor not in roadmap[point] and not edge_collision(point, neighbor, obstacle_map):
                    roadmap[point].append(neighbor)
        
    return roadmap


def PRM(start, goal, map_size, obstacle_map, num_points, k):
  
    algo_start = time.time()
    roadmap = build_roadmap(tuple(start), tuple(goal), map_size, obstacle_map, num_points, k)
    algo_end = time.time()

    optimal_path = astar(tuple(start), tuple(goal), roadmap)
    print("The algorithm takes: ", algo_end - algo_start)


    min_dist = euclidean_heuristic(start, goal)

    path_dist = path_distance(optimal_path)

    normalized_ratio = path_dist/min_dist

    return roadmap, optimal_path, normalized_ratio



def PRM_star(start, goal, map_size, obstacle_map, num_points, k):
    
    algo_start = time.time()
    roadmap = build_roadmap(tuple(start), tuple(goal), map_size, obstacle_map, num_points, k, reconnect=True)

    optimal_path = astar(tuple(start), tuple(goal), roadmap)
    algo_end = time.time()
    print("The algorithm takes: ", algo_end - algo_start)

    min_dist = euclidean_heuristic(start, goal)

    path_dist = path_distance(optimal_path)

    normalized_ratio = path_dist/min_dist

    return roadmap, optimal_path, normalized_ratio




