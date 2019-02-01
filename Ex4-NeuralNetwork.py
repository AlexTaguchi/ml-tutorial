# Let's do a simple linear regression, like in the last example, but
# with machine learning. Let's import the necessary modules. "torch"
# is the PyTorch machine learning module
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


# Let's generate some noisy y = 2x data like last time, except using the
# PyTorch module instead of numpy. Don't worry about what "view" is doing,
# it's just a reshaping trick to make x a column vector instead of a row
# vector
x = torch.arange(0, 10, 0.1).view(-1, 1)
y = 2 * x
y += torch.randn((100, 1))


# Our neural network needs to be trained to predict y from x. It basically
# needs to find the optimal value of m in y = mx + b. The neural network
# is always defined as a class where the first 3 lines are always the same:
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Here's the first section in the class we need to modify.
        # We'll make it a single-layer neural network, where "nn.Linear(1, 1)"
        # means the layer accepts 1 input, and spits out 1 output. In this
        # very simple case, the input is x, and it is multiplied by a constant
        # to predict y.
        self.m = nn.Linear(1, 1)

    # In addition to the "__init__" function, you always define a "forward"
    # function which defines how the input is passed through the network.
    def forward(self, input):

        # Here's the second section we need to modify. We will pass the input
        # through the neural network, and get the output.
        return self.m(input)


# The general template for writing ANY neural network in PyTorch is as follows:
#
# class Net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # ADD LAYERS HERE
#
#     def forward(self, input):
#         # DO STUFF WITH THE LAYERS HERE


# Create an instance of this neural network architecture. The neural network
# doesn't really "exist" yet until we run this command. The class definition
# above is only a blue-print on how to make the neural network. Here we actually
# create it.
net = Net()

# Define our criterion by which we will optimize the neural network. We'll
# use mean-squared error (MSE) here.
criterion = nn.MSELoss()

# Choose an algorithm to optimize the MSE, where "SGD" is a popular optimization
# algorithm, and "lr" is the tunable learning rate parameter
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

# Train the neural network for ten steps. Everything in the for loop (except for
# the print statement) is standard, and you can basically just copy-paste it to
# train other neural networks if you want the standard options.
for step in range(10):
    optimizer.zero_grad()
    output = net(x)
    loss = criterion(output, y)
    print(loss[0])
    loss.backward()
    optimizer.step()

# Plot the result. The ".numpy()" is because we are converting PyTorch tensor objects
# to numpy objects. Matplotlib can't work with tensors, and only knows how to plot
# numpy objects. The ".detach()" is more complicated, but just has to do with the need
# to disassociate the variable from the neural network before it can be treated as
# an independent variable that no longer needs to be optimized.
plt.plot(x.numpy(), y.numpy())
plt.plot(x.numpy(), output.detach().numpy())
plt.show()
