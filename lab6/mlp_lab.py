from sklearn.neural_network import MLPClassifier
import numpy as np

# 1. Preparing training data
# Each array in X contains client data: [Age, Income in thousands of UAH]
X_train = np.array([
    [25, 40],   # 25 years old, 40k
    [45, 120],  # 45 years old, 120k
    [20, 15],   # 20 years old, 15k
    [35, 60],   # 35 years old, 60k
    [50, 150],  # 50 years old, 150k
    [19, 10]    # 19 years old, 10k
])

# Corresponding labels (y): 1 - loan approved, 0 - denied
y_train = np.array([1, 1, 0, 1, 1, 0])

# 2. Creating and training a multilayer perceptron
# We use one hidden layer with 5 neurons, which is enough for this simple task
mlp = MLPClassifier(hidden_layer_sizes=(5,), max_iter=2000, random_state=42)
mlp.fit(X_train, y_train)

# 3. Making predictions for new clients
# Imagine this data came from a frontend through a POST request
new_clients = np.array([
    [28, 50],  # New client 1: 28 years old, 50k income
    [21, 12]   # New client 2: 21 years old, 12k income
])

predictions = mlp.predict(new_clients)

# 4. Printing a readable result
for i, client in enumerate(new_clients):
    status = "APPROVED" if predictions[i] == 1 else "DENIED"
    print(f"Client {i+1} (Age: {client[0]}, Income: {client[1]}k) -> Result: {status}")