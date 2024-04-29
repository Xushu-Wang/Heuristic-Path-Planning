import torch
import numpy as np
from neural_astar.planner import NeuralAstar
from neural_astar.utils.training import load_from_ptl_checkpoint
from prm import build_roadmap
from heuristics import euclidean_heuristic
import heapq


device = "cuda" if torch.cuda.is_available() else "cpu"

neural_astar = NeuralAstar(encoder_arch='CNN').to(device)
neural_astar.load_state_dict(load_from_ptl_checkpoint("/content"))


def convert_to_tensors(grid, start_point, end_point):
    # Convert numpy array to PyTorch tensor
    grid = np.expand_dims(grid, axis = 0)
    grid_tensor = torch.tensor(grid, dtype=torch.float32)

    # Create starting map
    start_map = np.zeros_like(grid)
    start_map[0, start_point[0], start_point[1]] = 1
    start_map_tensor = torch.tensor(start_map, dtype=torch.float32)

    # Create ending map
    end_map = np.zeros_like(grid)

    end_map[0, end_point[0], end_point[1]] = 1
    end_map_tensor = torch.tensor(end_map, dtype=torch.float32)

    # Add a dimension to simulate batch dimension
    grid_tensor = grid_tensor.unsqueeze(0)
    start_map_tensor = start_map_tensor.unsqueeze(0)
    end_map_tensor = end_map_tensor.unsqueeze(0)

    return grid_tensor, start_map_tensor, end_map_tensor





def prm_astar(start, goal, map_size, obstacle_map, num_points, k):
    
    roadmap = build_roadmap(tuple(start), tuple(goal), map_size, obstacle_map, num_points, k, reconnect=True)

    frontier = [(0, start)]
    came_from = {}
    cost_so_far = {start: 0}

    while frontier:
        _, current = heapq.heappop(frontier)

        if current == goal:
            break

        for neighbor in roadmap[current]:
            
            grid_instance, start_instance, end_instance = convert_to_tensors(obstacle_map, neighbor, goal)
            
            heuristic = neural_astar(grid_instance, start_instance, end_instance)

            new_cost = cost_so_far[current] + heuristic
            
            if neighbor not in cost_so_far or new_cost < cost_so_far.get(neighbor, float('inf')):
                cost_so_far[neighbor] = new_cost
                priority = new_cost + euclidean_heuristic(goal, neighbor, epsilon=0.1)
                heapq.heappush(frontier, (priority, neighbor))
                came_from[neighbor] = current

    path = [goal]
    while path[-1] != start:
        path.append(came_from[path[-1]])
    path.reverse()
    return path
    



