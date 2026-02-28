import numpy as np
import matplotlib.pyplot as plt

FILE_NAME = "points.txt"


def generate_random_points(num_points, x_range=(0, 100), y_range=(0, 100)):
    points = np.random.rand(num_points, 2)
    points[:, 0] = points[:, 0] * (x_range[1] - x_range[0]) + x_range[0]
    points[:, 1] = points[:, 1] * (y_range[1] - y_range[0]) + y_range[0]
    return points


def get_user_input():
    # Default values
    defaults = {
        'num_points': 50,
        'max_coord': 1000,
    }

    print(f"\nDefault parameters:")
    print(f"  • Number of points: {defaults['num_points']}")
    print(f"  • Max coordinate: {defaults['max_coord']}")
    print("\n" + "-"*50)
    user_input = input(
        "Do you want to use default parameters? (y/n): ").strip().lower()

    if user_input == 'y':
        print("\n✅ Using default parameters")
        params = dict(defaults)
        return params

    print("\n📝 Enter custom parameters (press Enter to keep default value)")
    print("-"*50)

    params = {}

    # Number of points
    while True:
        try:
            value = input(
                f"Number of points [{defaults['num_points']}]: ").strip()
            if value == "":
                params['num_points'] = defaults['num_points']
                break
            value = int(value)
            if value > 0:
                params['num_points'] = value
                break
            else:
                print("❌ Number of points must be positive")
        except ValueError:
            print("❌ Please enter a valid integer")

    # Max coordinate
    while True:
        try:
            value = input(
                f"Max coordinate [{defaults['max_coord']}]: ").strip()
            if value == "":
                params['max_coord'] = defaults['max_coord']
                break
            value = int(value)
            if value > 0:
                params['max_coord'] = value
                break
            else:
                print("❌ Max coordinate must be positive")
        except ValueError:
            print("❌ Please enter a valid integer")

    print("\n✅ Custom parameters set successfully")
    return params


def main():
    # GET USER PARAMETERS
    params = get_user_input()

    points = generate_random_points(
        params['num_points'], x_range=(0, params['max_coord']), y_range=(0, params['max_coord']))
    np.savetxt(FILE_NAME, points, fmt="%.2f", header="x y", comments="")
    print(
        f"Generated [{params['num_points']}] random points and saved to {FILE_NAME}")
    
    # Optional: Plot the generated points
    fig, ax = plt.subplots()
    ax.scatter(points[1:, 0], points[1:, 1], c='blue', label='Other cities')
    ax.scatter(points[0, 0], points[0, 1], c='red',
               s=100, label='Start city', zorder=5)
    plt.title(f"Generated {params['num_points']} Random Points")
    plt.show()


if __name__ == "__main__":
    print(f"Running {__file__} to generate random points...")
    print("="*50)

    main()
