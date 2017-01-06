import numpy as np
import matplotlib.pyplot as plt
from datasets import CelebA, MNIST
z1 = np.load('z1.npy')
y1 = np.load('y1.npy')
z2 = np.load('z2.npy')
y2 = np.load('y2.npy')
zs = np.load('z_sampled.npy')
ys = np.load('y_sampled.npy')

z = np.append(z2, zs, axis=0)
y = np.append(y2, ys, axis=0)

plt.scatter(z[:, 0], z[:, 1], c=y)

plt.legend
plt.show()

