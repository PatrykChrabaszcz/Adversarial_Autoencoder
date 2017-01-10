import numpy as np
import matplotlib.pyplot as plt

z_e = np.load('z.npy')
y = np.load('y.npy')
print(y)
plt.scatter(z_e[:, 5], z_e[:, 10], c=y)
plt.show()
