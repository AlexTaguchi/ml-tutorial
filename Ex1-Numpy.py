# The first step in any python script is to import "modules". Modules are
# packages that give you access to more specialized functions not available
# in native python. Native python is pretty bare-bones; you can't even perform
# a square-root operation in native python! Let's import the "numpy" module,
# and rename it so we can refer to it as "np" instead of the longer "numpy"
import numpy as np

# Now we have access to a lot of math and matrix operations. Let's assign a value
# to a variable and print the square-root of its value.
a = 16
print('The square-root of 16:')
print(np.sqrt(a))

# Now let's make a matrix of values, and print the square of each. Note that
# a matrix is defined a bit differently than in Matlab. It may look confusing at
# first, but it's actually more explicit than in Matlab and generalizes better
# when you go to higher dimensional matrices.
b = np.array([[1, 2],
              [3, 4]])
print('The square of 1, 2, 3, and 4:')
print(b**2)

# Let's take the square-root of the element at the second row, first column (3).
# You can access this by typing b[1, 0]. Why not b[2, 1]? It's because python
# uses "zero-indexing". That means the first index of any array or matrix is in
# the "zero position". Matlab is "one-indexed". Confusing, but you'll get used
# to it.
print('The square-root of 3:')
print(np.sqrt(b[1, 0]))