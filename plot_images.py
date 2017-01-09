import matplotlib.pyplot as plt
import numpy as np
from src.datasets import MNIST
x_im = np.load("x_rec.npy")
xhq_im = np.load("xhq_rec.npy")

data = MNIST(mean=False)
batch = data.train_images[:100]
batch = np.reshape(batch, [-1, 28, 28])

for i in range(10):
    im = np.reshape(x_im[i], [28, 28])

    plt.imshow(batch[i], cmap='gray')
    plt.show()

    plt.imshow(im, cmap='gray')
    plt.show()



