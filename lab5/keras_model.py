import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# 1. Loading and normalizing the MNIST data
# Pixel values range from 0 to 255, so we divide by 255.0 to get a range of 0-1
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 2. Building the model (declarative approach)
model = keras.models.Sequential([
    keras.layers.Input(shape=(28, 28)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'), # Hidden layer
    keras.layers.Dense(10, activation='softmax') # Output layer (10 digits)
])

# 3. Compiling the model (choosing the optimizer and loss function)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4. Training the model
print("[LOG] Starting learning on Keras...")
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# 5. Evaluating accuracy
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
print(f'\n[LOG] Accuracy on test data: {test_acc:.4f}')

# 6. Plotting the loss graph
plt.plot(history.history['loss'], label='Втрати на навчанні')
plt.plot(history.history['val_loss'], label='Втрати на валідації')
plt.xlabel('Епоха')
plt.ylabel('Втрати (Loss)')
plt.legend()
plt.title('Графік функції втрат (Keras)')
plt.show()