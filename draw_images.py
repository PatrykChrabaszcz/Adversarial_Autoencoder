import numpy as np
from itertools import combinations

from src.datasets import MNIST, CelebA
from src.model_mnist_dense import ModelDenseMnist
from src.model_mnist_conv import ModelConvMnist
from src.model_mnist_hq import ModelHqMnist
from src.model_celeb_conv import ModelConvCeleb
from src.solver import Solver
import tensorflow as tf
from PIL import Image


# Script to generate series of images that can later be converted to a gif
def draw_images(model, data, name):
    # Solver
    solver = Solver(model=model)
    # Session
    config = tf.ConfigProto(
        device_count={'GPU': 0}
    )
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    # Saver

    saver = tf.train.Saver()

    # To restore previous
    print("Restoring model")
    saver.restore(sess, 'models/model_%s.ckpt' % name)
    print("Model restored")

    z = []

    # Generate grid of z
    samples = 5
    r = 2.5
    for z_4 in np.linspace(-r, r, samples*20):
        for z_3 in np.linspace(-r, r, samples):
            for z_2 in np.linspace(-r, r, samples):
                for z_1 in np.linspace(-r, r, samples):
                    for z_0 in np.linspace(-r, r, samples):
                        z.append([z_0, z_1, z_2, z_3, z_4])

    z = np.array(z)
    z.reshape([-1, model.z_dim])
    code_5 = [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]]
    code_8 = [[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]]

    y_lab = np.reshape(code_8 * (samples**5*20), [(samples**5*20), 10])
    img = sess.run(solver.x_from_z, feed_dict={solver.z_provided: z, solver.y_labels: y_lab})

    for z4 in range(samples*20):
        blank_image = Image.new('RGB', (28 * samples**2 + (samples-1)*10, 28 * samples**2 + 60))
        for z3 in range(samples):
            for z2 in range(samples):
                for z1 in range(samples):
                    for z0 in range(samples):
                        index = samples**4 * z4 + samples**3 * z3 + samples**2 * z2 + samples * z1 + z0
                        im = img[index].reshape([28, 28])
                        cimg = Image.fromarray(np.uint8(im*255))

                        x = z0 * 28 + z2*(10+28*samples)
                        y = z1 * 28 + z3*(10+28*samples)
                        blank_image.paste(cimg, (x, y))

        blank_image.save("output/z4/Res_%d.png" % z4)


if __name__ == '__main__':
    scenario = 1
    y_dim = 10
    if scenario == 1:
        model = ModelDenseMnist(batch_size=5**5*20, z_dim=5, y_dim=y_dim, is_training=False)
        data = MNIST(mean=False)
        draw_images(model, data, name='Mnist_Dense_Adam')
    if scenario == 2:
        model = ModelConvMnist(batch_size=5**5*20, z_dim=5, y_dim=y_dim, is_training=False)
        data = MNIST(mean=False)
        draw_images(model, data, name='Mnist_Conv_Adam')
    if scenario == 3:
        model = ModelHqMnist(batch_size=100, z_dim=5, y_dim=y_dim, is_training=False)
        data = MNIST(mean=False)
        draw_images(model, data, name='Mnist_Hq')
    if scenario == 4:
        model = ModelConvCeleb(batch_size=128, z_dim=25, y_dim=None, is_training=False)
        data = CelebA()
        draw_images(model, data, name='Celeb_Conv_Momentum_noy')
