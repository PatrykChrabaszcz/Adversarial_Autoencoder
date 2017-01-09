import matplotlib.pyplot as plt
import numpy as np
from src.datasets import MNIST
x_im = np.load("x_rec.npy")
xhq_im = np.load("xhq_rec.npy")

data = MNIST()
batch = MNIST(mean=False).train_images[:100]
batch = np.reshape(batch, [-1, 28, 28])

for i in range(10):
    im = x_im[i] + data.mean_image
    #imhq = xhq_im[i]
    im = np.reshape(im, [28, 28])
    #imhq = np.reshape(imhq, [28*50, 28*50])
    plt.imshow(batch[i], cmap='gray')
    plt.show()
    plt.imshow(im, cmap='gray')
    plt.show()



