import numpy as np
import matplotlib.pyplot as plt
import math

z_e = np.load('z.npy')
y = np.load('y.npy')
print(y)
plt.scatter(z_e[:, 0], z_e[:, 1], c=y)
plt.show()


