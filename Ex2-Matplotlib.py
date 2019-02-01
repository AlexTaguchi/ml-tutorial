# Numpy is the number one tool for data processing in python. The number one
# visualization tool is matplotlib. It's actually named as such because it's
# actually just a port of the plotting tools in Matlab. This time, we'll
# import sub-module "pyplot" of matplotlib, because it's the only one we
# need. We'll also rename it as "plt" for convenience.
import matplotlib.pyplot as plt

# We also need numpy
import numpy as np

# Let's make a cosine curve. First we'll define the x-axis. You can make a
# linear range of points with the helpful "np.arange" function. Just pass in
# the first value, last value, and step and it'll make a vector of the values.
# For example, np.arange(0, 10, 2) will give you a vector [0, 2, 4, 6, 8].
# Why not [0, 2, 4, 6, 8, 10]? It's because np.arange always excludes the last
# value. Don't worry about why this is for now, but know that it has something
# to do with the fact that python is "zero-indexed".
x = np.arange(0, 10, 2)
print('x:')
print(x)

# With only 5 values, this will make for a pretty ugly cosine curve. Let's
# make it a vector 50 long.
x = np.arange(0, 10, 0.2)

# Let's get the y-values
y = np.cos(x)

# Time to plot it. Technically, the plt.plot() command only creates the plot
# object without showing it (unless you're in what's called "interactive
# mode"). Generally, you need to call plt.show() to get the plot objects to
# appear.
plt.plot(x, y)
plt.show()