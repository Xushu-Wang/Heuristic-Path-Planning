from dataset import MapDataset, MapSample
from utils import plot_path, plot_roadmap, plot_roadmap_with_path
from prm import PRM, PRM_star
from rrt import rrt_star
from heuristics import path_distance, euclidean_heuristic



if __name__ == "__main__":
    
    import sys, types
    sys.modules['dataset'] = types.ModuleType('dataset')
    sys.modules['dataset.map_sample'] = types.ModuleType('map_sample')
    sys.modules['dataset.map_sample'].__dict__['MapSample'] = MapSample
    
    dataset = MapDataset('/Users/andywang/Desktop/data')

    total_time = 0
    total_ratio = 0
    
    for i in range(5):
        map, start, end, path = dataset[i].numpy()

        map_size = (100, 100)
        obstacle_map = map
        start = start
        goal = end
        num_points = 1500
        k = 10

        roadmap, optimal_path, ratio = PRM_star(start, goal, map_size, obstacle_map, num_points, k)

        plot_roadmap_with_path(roadmap, obstacle_map, start, goal, optimal_path)

        print("Optimal Path:", optimal_path)
        print("Optimal Path Distance: ", path_distance(optimal_path))
        print("Ratio: ", path_distance(optimal_path)/euclidean_heuristic(start, goal))

        total_ratio += path_distance(optimal_path)/euclidean_heuristic(start, goal)

    print(total_time/5)
    print(total_ratio/5)