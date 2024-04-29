import heapq
from heuristics import euclidean_heuristic, mixed_heuristic
import torch



def astar(start, goal, roadmap):
    frontier = [(0, start)]
    came_from = {}
    cost_so_far = {start: 0}

    while frontier:
        _, current = heapq.heappop(frontier)

        if current == goal:
            break

        for neighbor in roadmap[current]:
            new_cost = cost_so_far[current] + euclidean_heuristic(neighbor, current, epsilon=0.1)
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