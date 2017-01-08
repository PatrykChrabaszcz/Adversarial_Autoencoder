import matplotlib.pyplot as plt
import numpy as np

x_im = np.load("x_rec.npy")
xhq_im = np.load("xhq_rec.npy")

for i in range(10):
    im = x_im[i]
    imhq = xhq_im[i]
    im = np.reshape(im, [28, 28])
    imhq = np.reshape(imhq, [28*4, 28*4])
    plt.imshow(im, cmap='gray')
    plt.show()
#plt.imshow(imhq, cmap='gray')
#plt.show()

