# Import modules
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
import torchvision.datasets as datasets

# Import the MNIST dataset
train = datasets.MNIST(root='.', train=True)
test = datasets.MNIST(root='.', train=False)

# Two-layer neural network
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Define the neural network layers
        self.layer1 = nn.Linear(28**2, 64)
        self.layer2 = nn.Linear(64, 10)

    # Define how the input is passed through the network
    def forward(self, x):
        x = x.view(-1, 28**2)
        x = self.layer1(x)
        x = functional.relu(x)
        x = self.layer2(x)
        return x

# Create an instance of this neural network architecture
net = Net()

# Define our criterion and optimizer to train the neural network
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001)

# Train the neural network
for step in range(100):
    training_batch = np.random.choice(len(train), 100)
    optimizer.zero_grad()
    output = net(train.data[training_batch].float())
    loss = criterion(output, train.targets[training_batch])
    loss.backward()
    optimizer.step()

# Plot the neural network performance on the test set
fig, ax = plt.subplots(nrows=3, ncols=3)
test_batch = np.random.choice(len(test), 9)
output = net(test.data[test_batch].float())
for i in range(9):
    ax[i//3, i%3].imshow(test.data[test_batch[i]])
    ax[i//3, i%3].set_title(str(test.targets[test_batch[i]].item()))
plt.show()
