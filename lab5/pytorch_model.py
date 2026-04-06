import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 1. Loading data through DataLoader
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# 2. Defining the neural network class
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x) # Softmax is applied inside CrossEntropyLoss
        return x

model = Net()

# 3. Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Training loop
epochs = 5
losses = []
val_losses = []
print("Starting learning on PyTorch...")
for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in trainloader:
        optimizer.zero_grad() # Clearing gradients
        
        outputs = model(images) # Forward pass
        loss = criterion(outputs, labels) # Calculating loss
        
        loss.backward() # Backpropagation
        optimizer.step() # Updating weights
        
        running_loss += loss.item()
    epoch_loss = running_loss / len(trainloader)
    losses.append(epoch_loss)
    print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

    val_running_loss = 0.0
    model.eval()
    with torch.no_grad():
        for images, labels in testloader:
            outputs = model(images)
            val_loss = criterion(outputs, labels)
            val_running_loss += val_loss.item()
    model.train()

    epoch_val_loss = val_running_loss / len(testloader)
    val_losses.append(epoch_val_loss)
    print(f"Epoch {epoch+1}, Validation Loss: {epoch_val_loss:.4f}")

# 5. Calculating accuracy
correct = 0
total = 0
with torch.no_grad(): # Disabling gradient calculations for testing
    for images, labels in testloader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'\nAccuracy on test data: {100 * correct / total:.2f}%')

# 6. Plotting the loss graph
plt.plot(range(1, epochs + 1), losses, label='Втрати на навчанні')
plt.plot(range(1, epochs + 1), val_losses, label='Втрати на валідації')
plt.xlabel('Епоха')
plt.ylabel('Втрати (Loss)')
plt.legend()
plt.title('Графік функції втрат (PyTorch)')
plt.show()