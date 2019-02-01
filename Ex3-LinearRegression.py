# Let's fit some data to a line. First, as you'll usually end up doing,
# import numpy and matplotlib.pyplot
import numpy as np
import matplotlib.pyplot as plt

# Let's make some ideal data corresponding to y = 2x
x = np.arange(0, 10, 0.1)
y = 2 * x

# Let's add some noise to y. Let's first make 100 points of noise
# (standard deviation of 1 about a mean of 0). We'll verify
# the noise vector is 100 points long with print(len(noise)).
noise = np.random.randn(100)
print(len(noise))

# Add the noise to the y-variable. Note that "y += noise" is shorthand
# for "y = y + noise"
y += noise

# Ok, let's plot the noisy data, but don't show it yet
plt.plot(x, y)

# Let's do a linear (1st order) fit. Numpy's polyfit returns the vector
# [m, b] from the equation y = mx + b. For higher order fits, just change
# the 1 to 2, 3, 4...
fit = np.polyfit(x, y, 1)
print('slope:')
print(fit[0])
print('intercept')
print(fit[1])

# Plot the fit, and show the graph
plt.plot(x, fit[0] * x + fit[1])
plt.show()
