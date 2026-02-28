import random
import time
import numpy as np
import matplotlib.pyplot as plt

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


def fitness(cities, route):
    """Calculate fitness (inverse of distance - shorter routes have higher fitness)"""
    return 1 / route_distance(cities, route)


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


# INITIAL POPULATION
def create_population(num_cities, population_size):
    """Create random initial population"""
    population = []
    base = list(range(num_cities))
    for _ in range(population_size):
        individual = base[:]
        random.shuffle(individual)
        population.append(individual)
    return population


# SELECTION (tournament)
def tournament_selection(cities, population, tournament_size):
    """Select best individual from random tournament"""
    tournament = random.sample(population, tournament_size)
    tournament.sort(key=lambda x: route_distance(cities, x))
    return tournament[0]


# CROSSOVER (Order Crossover OX)
def crossover(parent1, parent2, num_cities):
    """
    Order Crossover (OX):
    1. Select random segment from parent1
    2. Copy segment to child
    3. Fill remaining positions with genes from parent2 in order
    """
    start, end = sorted(random.sample(range(num_cities), 2))
    child = [None] * num_cities
    child[start:end] = parent1[start:end]

    pointer = 0
    for gene in parent2:
        if gene not in child:
            while child[pointer] is not None:
                pointer += 1
            child[pointer] = gene

    return child


# MUTATION (swap two cities)
def mutate(route, mutation_rate, num_cities):
    """Swap two random cities with probability MUTATION_RATE"""
    if random.random() < mutation_rate:
        i, j = random.sample(range(num_cities), 2)
        route[i], route[j] = route[j], route[i]


def get_user_input():
    """Get parameters from user with validation"""

    print("\n" + "="*50)
    print("TSP GENETIC ALGORITHM CONFIGURATION")
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
        'population_size': 100,
        'generations': 200,
        'mutation_rate': 0.02,
        'tournament_size': 5
    }

    print(f"\nDefault parameters:")
    if city_source == "random":
        print(f"  • Number of cities: {defaults['num_cities']}")
    print(f"  • Population size: {defaults['population_size']}")
    print(f"  • Number of generations: {defaults['generations']}")
    print(f"  • Mutation rate: {defaults['mutation_rate']}")
    print(f"  • Tournament size: {defaults['tournament_size']}")

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
    else:
        params['num_cities'] = num_cities_from_file

    # Population size
    while True:
        try:
            value = input(
                f"Population size [{defaults['population_size']}]: ").strip()
            if value == "":
                params['population_size'] = defaults['population_size']
                break
            value = int(value)
            if value > 0:
                params['population_size'] = value
                break
            else:
                print("❌ Population size must be positive")
        except ValueError:
            print("❌ Please enter a valid integer")

    # Number of generations
    while True:
        try:
            value = input(
                f"Number of generations [{defaults['generations']}]: ").strip()
            if value == "":
                params['generations'] = defaults['generations']
                break
            value = int(value)
            if value > 0:
                params['generations'] = value
                break
            else:
                print("❌ Number of generations must be positive")
        except ValueError:
            print("❌ Please enter a valid integer")

    # Mutation rate
    while True:
        try:
            value = input(
                f"Mutation rate (0.0-1.0) [{defaults['mutation_rate']}]: ").strip()
            if value == "":
                params['mutation_rate'] = defaults['mutation_rate']
                break
            value = float(value)
            if 0 <= value <= 1:
                params['mutation_rate'] = value
                break
            else:
                print("❌ Mutation rate must be between 0 and 1")
        except ValueError:
            print("❌ Please enter a valid number")

    # Tournament size
    while True:
        try:
            value = input(
                f"Tournament size [{defaults['tournament_size']}]: ").strip()
            if value == "":
                params['tournament_size'] = defaults['tournament_size']
                break
            value = int(value)
            if value > 1:
                params['tournament_size'] = value
                break
            else:
                print("❌ Tournament size must be greater than 1")
        except ValueError:
            print("❌ Please enter a valid integer")

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
    POPULATION_SIZE = params['population_size']
    GENERATIONS = params['generations']
    MUTATION_RATE = params['mutation_rate']
    TOURNAMENT_SIZE = params['tournament_size']
    CITY_SOURCE = params['city_source']
    CITY_FILE = params['city_file']

    # Display configuration
    print("\n" + "="*50)
    print("🚀 STARTING TSP GENETIC ALGORITHM")
    print("="*50)
    print(f"Configuration:")
    if CITY_SOURCE == "file":
        print(f"  • City source: file ({CITY_FILE})")
    else:
        print(f"  • City source: random")
    print(f"  • Number of cities: {NUM_CITIES}")
    print(f"  • Population size: {POPULATION_SIZE}")
    print(f"  • Number of generations: {GENERATIONS}")
    print(f"  • Mutation rate: {MUTATION_RATE}")
    print(f"  • Tournament size: {TOURNAMENT_SIZE}")
    print("="*50 + "\n")

    # CITY GENERATION / LOADING
    if CITY_SOURCE == "file":
        cities = load_cities_from_file(CITY_FILE)
        print(f"📂 Cities loaded from file: {CITY_FILE}")
    else:
        rng = np.random.default_rng()
        cities = rng.uniform(0.0, 1.0, size=(NUM_CITIES, 2)) * 1000
        print("🎲 Random cities generated")

    # MAIN ALGORITHM
    population = create_population(NUM_CITIES, POPULATION_SIZE)
    best_route = None
    best_distance = float("inf")

    # Setup plot
    plt.ion()
    fig, ax = plt.subplots()

    # Start timer
    start_time = time.perf_counter()

    for generation in range(GENERATIONS):
        new_population = []

        # Create new generation
        for _ in range(POPULATION_SIZE):
            # Selection
            parent1 = tournament_selection(cities, population, TOURNAMENT_SIZE)
            parent2 = tournament_selection(cities, population, TOURNAMENT_SIZE)

            # Crossover
            child = crossover(parent1, parent2, NUM_CITIES)

            # Mutation
            mutate(child, MUTATION_RATE, NUM_CITIES)

            new_population.append(child)

        population = new_population

        # Track best solution
        current_best = min(population, key=lambda x: route_distance(cities, x))
        current_distance = route_distance(cities, current_best)

        if current_distance < best_distance:
            best_distance = current_distance
            best_route = current_best
            elapsed = time.perf_counter() - start_time
            print(
                f"✨ Generation {generation:3d}: best distance = {best_distance:.2f}  [{format_elapsed(elapsed)}]")

        # Visualization
        if generation % 5 == 0:
            ax.clear()
            best_cities = cities[best_route]

            ax.scatter(cities[1:, 0], cities[1:, 1],
                       c='blue', label='Other cities')
            ax.scatter(cities[0, 0], cities[0, 1], c='red',
                       s=100, label='Start city', zorder=5)

            ax.plot(
                np.append(best_cities[:, 0], best_cities[0, 0]),
                np.append(best_cities[:, 1], best_cities[0, 1]),
                'g-', alpha=0.7
            )

            ax.set_title(
                f"Generation {generation}/{GENERATIONS} - Best Distance: {best_distance:.2f}")
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
    print(f"Route length:     {best_distance:.2f}")
    print(f"Route order:      {best_route}")
    print(f"Execution time: {format_elapsed(total_time)}")
    print("="*50)

    plt.title(
        f"FINAL: Best Found Route (Length: {best_distance:.2f}) | Time: {format_elapsed(total_time)}")
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    print("🧬 GENETIC ALGORITHM FOR TRAVELING SALESMAN PROBLEM")
    print("="*50)
    main()
