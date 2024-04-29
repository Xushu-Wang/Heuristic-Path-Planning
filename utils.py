import matplotlib.pyplot as plt


def plot_roadmap(roadmap, obstacle_map, start, goal):
    fig, ax = plt.subplots()

    # Plot obstacle map
    ax.imshow(obstacle_map, cmap='binary')

    # Plot start and goal points
    ax.plot(start[1], start[0], 'go', markersize=8, label='Start')
    ax.plot(goal[1], goal[0], 'ro', markersize=8, label='Goal')

    # Plot roadmap edges
    for point, neighbors in roadmap.items():
        for neighbor in neighbors:
            ax.plot([point[1], neighbor[1]], [point[0], neighbor[0]], 'b-', linewidth=0.5)

    ax.legend()
    ax.set_title('PRM Roadmap')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True)
    plt.gca().invert_yaxis()
    plt.show()
    
    plt.savefig(fname = f'PRM Roadmap Starting at {start[0], start[1]}', dpi = 100)



def plot_path(obstacle_map, start, goal, path):
    fig, ax = plt.subplots()

    # Plot obstacle map
    ax.imshow(obstacle_map, cmap='binary')

    # Plot start and goal points
    ax.plot(start[1], start[0], 'go', markersize=8, label='Start')
    ax.plot(goal[1], goal[0], 'ro', markersize=8, label='Goal')

    # Plot path
    if path:
        path_x = [point[1] for point in path]
        path_y = [point[0] for point in path]
        ax.plot(path_x, path_y, 'g-', linewidth=2, label='Optimal Path')

    ax.legend()
    ax.set_title('RRT* with Path')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True)
    plt.gca().invert_yaxis()
    plt.show()
    
    plt.savefig(fname = f'RRT* Starting at {start[0], start[1]}', dpi = 100)





def plot_roadmap_with_path(roadmap, obstacle_map, start, goal, path):
    fig, ax = plt.subplots()

    # Plot obstacle map
    ax.imshow(obstacle_map, cmap='binary')

    # Plot start and goal points
    ax.plot(start[1], start[0], 'go', markersize=8, label='Start')
    ax.plot(goal[1], goal[0], 'ro', markersize=8, label='Goal')

    # Plot roadmap edges
    for point, neighbors in roadmap.items():
        for neighbor in neighbors:
            ax.plot([point[1], neighbor[1]], [point[0], neighbor[0]], 'b-', linewidth=0.5)

    # Plot path
    if path:
        path_x = [point[1] for point in path]
        path_y = [point[0] for point in path]
        ax.plot(path_x, path_y, 'g-', linewidth=2, label='Optimal Path')

    ax.legend()
    ax.set_title('PRM Roadmap with Path')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True)
    plt.gca().invert_yaxis()
    plt.show()
    
    plt.savefig(fname = f'PRM Roadmap Starting at {start[0], start[1]}', dpi = 100)

    