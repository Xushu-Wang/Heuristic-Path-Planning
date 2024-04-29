import numpy as np
import time
from heuristics import path_distance, euclidean_heuristic

class Node:
    def __init__(self, position):
        self.position = position
        self.parent = None
        self.cost = 0.0

def is_collision(point, obstacle_map):
    x, y = point
    return obstacle_map[x, y] == 1

def nearest_neighbor(node_list, new_node):
    distances = [np.linalg.norm(np.array(new_node.position) - np.array(n.position)) for n in node_list]
    nearest_index = np.argmin(distances)
    return node_list[nearest_index]

def steer(from_node, to_node, max_distance):

    direction = np.array(to_node.position) - np.array(from_node.position)
    distance = np.linalg.norm(direction)
    if distance > max_distance:
        direction = direction / distance * max_distance
    new_position = np.array(from_node.position) + direction

    return Node(new_position)

def is_path_clear(from_node, to_node, obstacle_map):
    line = np.linspace(from_node.position, to_node.position, num=100)
    for point in line:
        if is_collision(point.astype(int), obstacle_map):
            return False
    return True


def rewire(node_list, new_node, max_distance, obstacle_map):
    for node in node_list:
        if node == new_node or node.parent == new_node:
            continue
        if is_path_clear(new_node, node, obstacle_map):
            new_cost = new_node.cost + np.linalg.norm(np.array(new_node.position) - np.array(node.position))
            if new_cost < node.cost:
                node.parent = new_node
                node.cost = new_cost
                  

def rrt_star(start, goal, map_size, obstacle_map, max_iter, max_distance):
    algo_start = time.time()
    node_list = [Node(start)]

    for _ in range(max_iter):
        rand_point = np.random.randint(map_size[0]), np.random.randint(map_size[1])
        nearest_node = nearest_neighbor(node_list, Node(rand_point))
        new_node = steer(nearest_node, Node(rand_point), max_distance)

        if is_path_clear(nearest_node, new_node, obstacle_map):
            new_node.parent = nearest_node
            new_node.cost = nearest_node.cost + np.linalg.norm(np.array(nearest_node.position) - np.array(new_node.position))

            near_nodes = [node for node in node_list if np.linalg.norm(np.array(node.position) - np.array(new_node.position)) < max_distance]
            for near_node in near_nodes:
                if is_path_clear(near_node, new_node, obstacle_map):
                    new_cost = near_node.cost + np.linalg.norm(np.array(near_node.position) - np.array(new_node.position))
                    if new_cost < new_node.cost:
                        new_node.parent = near_node
                        new_node.cost = new_cost

            node_list.append(new_node)
            rewire(node_list, new_node, max_distance, obstacle_map)

    goal_node = nearest_neighbor(node_list, Node(goal))
    if np.linalg.norm(np.array(goal_node.position) - np.array(goal)) > max_distance:
        goal_node = steer(goal_node, Node(goal), max_distance)
        if is_path_clear(nearest_node, goal_node, obstacle_map):
            goal_node.parent = nearest_node
            goal_node.cost = nearest_node.cost + np.linalg.norm(np.array(nearest_node.position) - np.array(goal_node.position))
            node_list.append(goal_node)
            rewire(node_list, goal_node, max_distance, obstacle_map)

    if goal_node.parent is None:
        return None

    path = []
    current_node = goal_node
    while current_node is not None:
        path.append(current_node.position)
        current_node = current_node.parent
    path.reverse()
    
    path_dist = path_distance(path)
    min_dist = euclidean_heuristic(start, goal)
    normalized_ratio = path_dist/min_dist
    
    algo_end = time.time()

    print("The algorithm takes: ", algo_end - algo_start)

    return path, normalized_ratio
