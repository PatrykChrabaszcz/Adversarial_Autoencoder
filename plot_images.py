import matplotlib.pyplot as plt
import numpy as np
from src.datasets import MNIST, CelebA
print("Loading decoded images ...")
x_im = np.load("x_rec.npy")
xhq_im = np.load("xhq_rec.npy")


print("Loading dataset ... ")
data = MNIST(mean=False)
batch = data.train_images[:128]
batch = np.resize(batch, [-1, 28, 28])
x_im = np.reshape(x_im, [-1, 28, 28])
for i in range(12):
    im = np.reshape(x_im[i], [28, 28])

    plt.imshow(batch[i], cmap='gray')
    plt.show()

    plt.imshow(x_im[i], cmap='gray')
    plt.show()



