import random
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
        total += distance(cities[route[i]], cities[route[(i + 1) % len(route)]])
    return total


def fitness(cities, route):
    """Calculate fitness (inverse of distance - shorter routes have higher fitness)"""
    return 1 / route_distance(cities, route)


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
    
    # Default values
    defaults = {
        'num_cities': 20,
        'population_size': 100,
        'generations': 300,
        'mutation_rate': 0.02,
        'tournament_size': 5
    }
    
    print(f"\nDefault parameters:")
    print(f"  • Number of cities: {defaults['num_cities']}")
    print(f"  • Population size: {defaults['population_size']}")
    print(f"  • Number of generations: {defaults['generations']}")
    print(f"  • Mutation rate: {defaults['mutation_rate']}")
    print(f"  • Tournament size: {defaults['tournament_size']}")
    
    print("\n" + "-"*50)
    user_input = input("Do you want to use default parameters? (y/n): ").strip().lower()
    
    if user_input == 'y':
        print("\n✅ Using default parameters")
        return defaults
    
    print("\n📝 Enter custom parameters (press Enter to keep default value)")
    print("-"*50)
    
    params = {}
    
    # Number of cities
    while True:
        try:
            value = input(f"Number of cities [{defaults['num_cities']}]: ").strip()
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
    
    # Population size
    while True:
        try:
            value = input(f"Population size [{defaults['population_size']}]: ").strip()
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
            value = input(f"Number of generations [{defaults['generations']}]: ").strip()
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
            value = input(f"Mutation rate (0.0-1.0) [{defaults['mutation_rate']}]: ").strip()
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
            value = input(f"Tournament size [{defaults['tournament_size']}]: ").strip()
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
    return params


def main():
    # GET USER PARAMETERS
    params = get_user_input()
    
    # Extract parameters
    NUM_CITIES = params['num_cities']
    POPULATION_SIZE = params['population_size']
    GENERATIONS = params['generations']
    MUTATION_RATE = params['mutation_rate']
    TOURNAMENT_SIZE = params['tournament_size']
    
    # Display configuration
    print("\n" + "="*50)
    print("🚀 STARTING TSP GENETIC ALGORITHM")
    print("="*50)
    print(f"Configuration:")
    print(f"  • Number of cities: {NUM_CITIES}")
    print(f"  • Population size: {POPULATION_SIZE}")
    print(f"  • Number of generations: {GENERATIONS}")
    print(f"  • Mutation rate: {MUTATION_RATE}")
    print(f"  • Tournament size: {TOURNAMENT_SIZE}")
    print("="*50 + "\n")

    # CITY GENERATION
    rng = np.random.default_rng()
    cities = rng.uniform(0.0, 1.0, size=(NUM_CITIES, 2)) * 100

    # MAIN ALGORITHM
    population = create_population(NUM_CITIES, POPULATION_SIZE)
    best_route = None
    best_distance = float("inf")

    # Setup plot
    plt.ion()
    fig, ax = plt.subplots()

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
            print(f"✨ Generation {generation:3d}: best distance = {best_distance:.2f}")

        # Visualization
        if generation % 5 == 0:
            ax.clear()
            best_cities = cities[best_route]

            # Plot all cities except the first one in blue
            ax.scatter(cities[1:, 0], cities[1:, 1], c='blue', label='Other cities')
            
            # Plot the first city (index 0) in red
            ax.scatter(cities[0, 0], cities[0, 1], c='red', s=100, label='Start city', zorder=5)
            
            # Plot the route
            ax.plot(
                np.append(best_cities[:, 0], best_cities[0, 0]),
                np.append(best_cities[:, 1], best_cities[0, 1]),
                'g-', alpha=0.7
            )

            ax.set_title(f"Generation {generation}/{GENERATIONS} - Best Distance: {best_distance:.2f}")
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            ax.grid(True, alpha=0.3)
            # ax.legend(loc='upper right')
            plt.pause(0.01)
    
    # FINAL RESULT
    print("\n" + "="*50)
    print("🎉 BEST ROUTE FOUND")
    print("="*50)
    print(f"Route length: {best_distance:.2f}")
    print(f"Route order: {best_route}")
    print("="*50)

    plt.title(f"FINAL: Best Found Route (Length: {best_distance:.2f})")
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    print("🧬 GENETIC ALGORITHM FOR TRAVELING SALESMAN PROBLEM")
    print("="*50)
    main()