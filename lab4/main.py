import random
import numpy as np
import matplotlib.pyplot as plt
import time

# HELPER FUNCTIONS


def distance(city1, city2):
    """Calculate Euclidean distance between two cities"""
    return np.linalg.norm(city1 - city2)


def route_distance(cities, route):
    """Calculate total distance of a route (returns to start city)"""
    total = 0
    for i in range(len(route)):
        total += distance(cities[route[i]],
                          cities[route[(i + 1) % len(route)]])
    return total


def build_distance_matrix(cities):
    """Precompute distance matrix for efficiency"""
    num_cities = len(cities)
    dist_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                dist_matrix[i][j] = distance(cities[i], cities[j])
    return dist_matrix


# ANT CONSTRUCTION
def construct_solution(num_cities, pheromone, dist_matrix, alpha, beta):
    """
    Single ant builds a complete route using probabilistic city selection:
    1. Start from a random city
    2. At each step, choose next city based on pheromone and heuristic (1/distance)
    3. Continue until all cities are visited
    """
    start = random.randint(0, num_cities - 1)
    visited = [False] * num_cities
    route = [start]
    visited[start] = True

    for _ in range(num_cities - 1):
        current = route[-1]

        # Calculate probabilities for unvisited cities
        probabilities = []
        candidates = []

        for j in range(num_cities):
            if not visited[j]:
                # pheromone attraction
                tau = pheromone[current][j] ** alpha
                # heuristic (visibility)
                eta = (1.0 / dist_matrix[current][j]) ** beta
                probabilities.append(tau * eta)
                candidates.append(j)

        # Normalize and select next city
        total = sum(probabilities)
        probabilities = [p / total for p in probabilities]
        next_city = random.choices(candidates, weights=probabilities, k=1)[0]

        route.append(next_city)
        visited[next_city] = True

    return route


# PHEROMONE UPDATE
def update_pheromones(pheromone, all_routes, cities, evaporation_rate, q):
    """
    Update pheromone matrix:
    1. Evaporate existing pheromones by factor (1 - evaporation_rate)
    2. Deposit new pheromones on each route proportional to route quality (Q / distance)
    """
    pheromone *= (1 - evaporation_rate)

    for route in all_routes:
        dist = route_distance(cities, route)
        deposit = q / dist
        for i in range(len(route)):
            a = route[i]
            b = route[(i + 1) % len(route)]
            pheromone[a][b] += deposit
            pheromone[b][a] += deposit  # symmetric TSP

# CITY PARSING FROM FILE


def load_cities_from_file(filepath):
    """
    Load cities from a file. Supported format:

    1. Simple XY (one city per line):
       x y
       10.5 20.3
       ...

    Returns: np.ndarray of shape (N, 2)
    """
    cities = []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) == 2:
            try:
                cities.append([float(parts[0]), float(parts[1])])
            except ValueError:
                continue  # skip lines that don't parse as numbers

    if len(cities) < 3:
        raise ValueError(
            f"File must contain at least 3 cities, found {len(cities)}")

    return np.array(cities)


def get_user_input():
    """Get parameters from user with validation"""

    print("\n" + "="*50)
    print("TSP ANT COLONY OPTIMIZATION CONFIGURATION")
    print("="*50)

    # --- City source ---
    print("\n📍 City source:")
    print("  [1] Generate random cities")
    print("  [2] Load cities from file")

    while True:
        choice = input("Choose option (1/2): ").strip()
        if choice in ("", "1", "2"):
            break
        print("❌ Please enter 1 or 2")

    city_source = "random" if choice in ("", "1") else "file"
    city_file = None
    num_cities_from_file = None

    if city_source == "file":
        while True:
            path = input("Enter path to city file: ").strip()
            if not path:
                print("❌ Path cannot be empty")
                continue
            try:
                test_cities = load_cities_from_file(path)
                num_cities_from_file = len(test_cities)
                print(f"✅ Loaded {num_cities_from_file} cities from '{path}'")
                city_file = path
                break
            except FileNotFoundError:
                print(f"❌ File not found: {path}")
            except ValueError as e:
                print(f"❌ {e}")

    # Default values
    defaults = {
        'num_cities': num_cities_from_file if num_cities_from_file else 20,
        'num_ants': 40,
        'iterations': 200,
        'alpha': 1.0,
        'beta': 5.0,
        'evaporation_rate': 0.5,
        'q': 100.0
    }

    print(f"\nDefault parameters:")
    if city_source == "random":
        print(f"  • Number of cities: {defaults['num_cities']}")
    print(f"  • Number of ants:      {defaults['num_ants']}")
    print(f"  • Number of iterations:{defaults['iterations']}")
    print(f"  • Alpha (pheromone):   {defaults['alpha']}")
    print(f"  • Beta  (heuristic):   {defaults['beta']}")
    print(f"  • Evaporation rate:    {defaults['evaporation_rate']}")
    print(f"  • Q (pheromone const): {defaults['q']}")

    print("\n" + "-"*50)
    user_input = input(
        "Do you want to use default parameters? (y/n): ").strip().lower()

    if user_input == 'y':
        print("\n✅ Using default parameters")
        params = dict(defaults)
        params['city_source'] = city_source
        params['city_file'] = city_file
        return params

    print("\n📝 Enter custom parameters (press Enter to keep default value)")
    print("-"*50)

    params = {}

    # Number of cities (only for random mode)
    if city_source == "random":
        while True:
            try:
                value = input(
                    f"Number of cities [{defaults['num_cities']}]: ").strip()
                if value == "":
                    params['num_cities'] = defaults['num_cities']
                    break
                value = int(value)
                if value > 2:
                    params['num_cities'] = value
                    break
                else:
                    print("❌ Number of cities must be greater than 2")
            except ValueError:
                print("❌ Please enter a valid integer")

    # Number of ants
    while True:
        try:
            value = input(f"Number of ants [{defaults['num_ants']}]: ").strip()
            if value == "":
                params['num_ants'] = defaults['num_ants']
                break
            value = int(value)
            if value > 0:
                params['num_ants'] = value
                break
            else:
                print("❌ Number of ants must be positive")
        except ValueError:
            print("❌ Please enter a valid integer")

    # Number of iterations
    while True:
        try:
            value = input(
                f"Number of iterations [{defaults['iterations']}]: ").strip()
            if value == "":
                params['iterations'] = defaults['iterations']
                break
            value = int(value)
            if value > 0:
                params['iterations'] = value
                break
            else:
                print("❌ Number of iterations must be positive")
        except ValueError:
            print("❌ Please enter a valid integer")

    # Alpha
    while True:
        try:
            value = input(
                f"Alpha - pheromone influence (>0) [{defaults['alpha']}]: ").strip()
            if value == "":
                params['alpha'] = defaults['alpha']
                break
            value = float(value)
            if value > 0:
                params['alpha'] = value
                break
            else:
                print("❌ Alpha must be positive")
        except ValueError:
            print("❌ Please enter a valid number")

    # Beta
    while True:
        try:
            value = input(
                f"Beta - heuristic influence (>0) [{defaults['beta']}]: ").strip()
            if value == "":
                params['beta'] = defaults['beta']
                break
            value = float(value)
            if value > 0:
                params['beta'] = value
                break
            else:
                print("❌ Beta must be positive")
        except ValueError:
            print("❌ Please enter a valid number")

    # Evaporation rate
    while True:
        try:
            value = input(
                f"Evaporation rate (0.0-1.0) [{defaults['evaporation_rate']}]: ").strip()
            if value == "":
                params['evaporation_rate'] = defaults['evaporation_rate']
                break
            value = float(value)
            if 0 < value < 1:
                params['evaporation_rate'] = value
                break
            else:
                print("❌ Evaporation rate must be between 0 and 1 (exclusive)")
        except ValueError:
            print("❌ Please enter a valid number")

    # Q
    while True:
        try:
            value = input(
                f"Q - pheromone constant (>0) [{defaults['q']}]: ").strip()
            if value == "":
                params['q'] = defaults['q']
                break
            value = float(value)
            if value > 0:
                params['q'] = value
                break
            else:
                print("❌ Q must be positive")
        except ValueError:
            print("❌ Please enter a valid number")

    print("\n✅ Custom parameters set successfully")
    params['city_source'] = city_source
    params['city_file'] = city_file
    return params


def format_elapsed(seconds):
    """Format elapsed time nicely"""
    if seconds < 60:
        return f"{seconds:.3f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.3f}s"


def main():
    # GET USER PARAMETERS
    params = get_user_input()

    # Extract parameters
    NUM_CITIES = params['num_cities']
    NUM_ANTS = params['num_ants']
    ITERATIONS = params['iterations']
    ALPHA = params['alpha']
    BETA = params['beta']
    EVAPORATION_RATE = params['evaporation_rate']
    CITY_SOURCE = params['city_source']
    CITY_FILE = params['city_file']
    Q = params['q']

    # Display configuration
    print("\n" + "="*50)
    print("🐜 STARTING TSP ANT COLONY OPTIMIZATION")
    print("="*50)
    print(f"Configuration:")
    print(f"  • Number of cities:    {NUM_CITIES}")
    print(f"  • Number of ants:      {NUM_ANTS}")
    print(f"  • Number of iterations:{ITERATIONS}")
    print(f"  • Alpha (pheromone):   {ALPHA}")
    print(f"  • Beta  (heuristic):   {BETA}")
    print(f"  • Evaporation rate:    {EVAPORATION_RATE}")
    print(f"  • Q (pheromone const): {Q}")
    print("="*50 + "\n")

    # CITY GENERATION / LOADING
    if CITY_SOURCE == "file":
        cities = load_cities_from_file(CITY_FILE)
        print(f"📂 Cities loaded from file: {CITY_FILE}")
    else:
        rng = np.random.default_rng()
        cities = rng.uniform(0.0, 1.0, size=(NUM_CITIES, 2)) * 1000
        print("🎲 Random cities generated")

    # PRECOMPUTE DISTANCES
    dist_matrix = build_distance_matrix(cities)

    # INITIALIZE PHEROMONES (uniform small value)
    pheromone = np.ones((NUM_CITIES, NUM_CITIES)) * (1.0 / NUM_CITIES)

    best_route = None
    best_distance = float("inf")

    # Setup plot
    plt.ion()
    fig, ax = plt.subplots()

    # Start timer
    start_time = time.perf_counter()

    # MAIN ALGORITHM
    for iteration in range(ITERATIONS):
        all_routes = []

        # Each ant constructs a solution
        for _ in range(NUM_ANTS):
            route = construct_solution(
                NUM_CITIES, pheromone, dist_matrix, ALPHA, BETA)
            all_routes.append(route)

        # Update pheromones based on all routes
        update_pheromones(pheromone, all_routes, cities, EVAPORATION_RATE, Q)

        # Track best solution
        current_best = min(all_routes, key=lambda r: route_distance(cities, r))
        current_distance = route_distance(cities, current_best)

        if current_distance < best_distance:
            best_distance = current_distance
            best_route = current_best
            elapsed = time.perf_counter() - start_time
            print(
                f"✨ Iteration {iteration:3d}: best distance = {best_distance:.2f}  [{format_elapsed(elapsed)}]")

        # Visualization
        if iteration % 5 == 0:
            ax.clear()
            best_cities = cities[best_route]

            # Pheromone heatmap (edge thickness ~ pheromone strength)
            max_pher = pheromone.max()
            for i in range(NUM_CITIES):
                for j in range(i + 1, NUM_CITIES):
                    strength = pheromone[i][j] / max_pher
                    if strength > 0.3:  # only draw strong edges to avoid clutter
                        ax.plot(
                            [cities[i, 0], cities[j, 0]],
                            [cities[i, 1], cities[j, 1]],
                            color='orange', alpha=strength * 0.4,
                            linewidth=strength * 2
                        )

            # Plot all cities
            ax.scatter(cities[1:, 0], cities[1:, 1],
                       c='blue', label='Other cities')
            ax.scatter(cities[0, 0], cities[0, 1], c='red',
                       s=100, label='Start city', zorder=5)

            # Plot best route
            ax.plot(
                np.append(best_cities[:, 0], best_cities[0, 0]),
                np.append(best_cities[:, 1], best_cities[0, 1]),
                'g-', alpha=0.8, linewidth=1.5
            )

            ax.set_title(
                f"Iteration {iteration}/{ITERATIONS} — Best Distance: {best_distance:.2f}")
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            ax.grid(True, alpha=0.3)
            plt.pause(0.01)

    # Stop timer
    total_time = time.perf_counter() - start_time

    # FINAL RESULT
    print("\n" + "="*50)
    print("🎉 BEST ROUTE FOUND")
    print("="*50)
    print(f"Route length: {best_distance:.2f}")
    print(f"Route order:  {best_route}")
    print(f"Execution time: {format_elapsed(total_time)}")
    print("="*50)

    plt.title(f"FINAL: Best Found Route (Length: {best_distance:.2f}) | Time: {format_elapsed(total_time)}")
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    print("🐜 ANT COLONY OPTIMIZATION FOR TRAVELING SALESMAN PROBLEM")
    print("="*50)
    main()
