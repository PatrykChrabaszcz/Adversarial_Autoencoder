import numpy as np
import matplotlib.pyplot as plt

z_e = np.load('z.npy')
y = np.load('y.npy')
print(y)
plt.scatter(z_e[:, 3], z_e[:, 3], c=y)
plt.show()
