from py_compile import main
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
def create_population():
    """Create random initial population"""
    population = []
    base = list(range(NUM_CITIES))
    for _ in range(POPULATION_SIZE):
        individual = base[:]
        random.shuffle(individual)
        population.append(individual)
    return population


# SELECTION (tournament)
def tournament_selection(cities, population):
    """Select best individual from random tournament"""
    tournament = random.sample(population, TOURNAMENT_SIZE)
    tournament.sort(key=lambda x: route_distance(cities, x))
    return tournament[0]


# CROSSOVER (Order Crossover OX)
def crossover(parent1, parent2):
    """
    Order Crossover (OX):
    1. Select random segment from parent1
    2. Copy segment to child
    3. Fill remaining positions with genes from parent2 in order
    """
    start, end = sorted(random.sample(range(NUM_CITIES), 2))
    child = [None] * NUM_CITIES
    child[start:end] = parent1[start:end]

    pointer = 0
    for gene in parent2:
        if gene not in child:
            while child[pointer] is not None:
                pointer += 1
            child[pointer] = gene

    return child


# MUTATION (swap two cities)
def mutate(route):
    """Swap two random cities with probability MUTATION_RATE"""
    if random.random() < MUTATION_RATE:
        i, j = random.sample(range(NUM_CITIES), 2)
        route[i], route[j] = route[j], route[i]


# DEFAULT PARAMETERS
NUM_CITIES = 20
POPULATION_SIZE = 100
GENERATIONS = 300
MUTATION_RATE = 0.02
TOURNAMENT_SIZE = 5

def main():
    global NUM_CITIES, POPULATION_SIZE, GENERATIONS, MUTATION_RATE, TOURNAMENT_SIZE

    # USER INPUT
    print("Do you want to use default parameters? (y/n)")
    user_input = input("> ").strip().lower()
    if user_input == 'n':
        print("Enter number value for parameter or d for default value.")
        value = int(input("number of cities: "))
        NUM_CITIES = value if value > 0 else NUM_CITIES
        value = int(input("population size: "))
        POPULATION_SIZE = value if value > 0 else POPULATION_SIZE
        value = int(input("number of generations: "))
        GENERATIONS = value if value > 0 else GENERATIONS
        value = float(input("mutation rate (0.0-1.0): "))
        MUTATION_RATE = value if 0 <= value <= 1 else MUTATION_RATE
        value = int(input("tournament size: "))
        TOURNAMENT_SIZE = value if value > 0 else TOURNAMENT_SIZE
    
    print(f"\nTSP Parameters: NUM_CITIES={NUM_CITIES}, POPULATION_SIZE={POPULATION_SIZE}, GENERATIONS={GENERATIONS}, MUTATION_RATE={MUTATION_RATE}, TOURNAMENT_SIZE={TOURNAMENT_SIZE}\n")

    # CITY GENERATION
    cities = np.random.rand(NUM_CITIES, 2) * 100

    # MAIN ALGORITHM
    population = create_population()
    best_route = None
    best_distance = float("inf")

    plt.ion()
    fig, ax = plt.subplots()

    for generation in range(GENERATIONS):
        new_population = []

        # Create new generation
        for _ in range(POPULATION_SIZE):
            # Selection
            parent1 = tournament_selection(cities, population)
            parent2 = tournament_selection(cities, population)

            # Crossover
            child = crossover(parent1, parent2)
            
            # Mutation
            mutate(child)

            new_population.append(child)

        population = new_population

        # Track best solution
        current_best = min(population, key=lambda x: route_distance(cities, x))
        current_distance = route_distance(cities, current_best)

        if current_distance < best_distance:
            best_distance = current_distance
            best_route = current_best
            print(f"Generation {generation}, best distance: {best_distance:.2f}")

        # Visualization
        if generation % 5 == 0:
            ax.clear()
            best_cities = cities[current_best]

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

            ax.set_title(f"Generation {generation}, distance: {best_distance:.2f}")
            # ax.legend()
            plt.pause(0.01)
    
    # FINAL RESULT
    print("\nBEST ROUTE FOUND")
    print("Route length:", best_distance)

    plt.title("Best Found Route")
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    print("Running Genetic Algorithm for TSP...\n")
    main()