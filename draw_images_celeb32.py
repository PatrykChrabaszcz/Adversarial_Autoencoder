import numpy as np
from src.model_subpix_32 import ModelSubpix32
from src.model_conv_32 import ModelConv32
from src.model_subpix_res32 import ModelSubpixRes32
import matplotlib.pyplot as plt

from src.datasets import CelebA
from src.aae_solver import AaeSolver
from src.aae_gan_solver import AaeGanSolver

import tensorflow as tf
from PIL import Image
from apng import APNG


samples = 20


def draw_images(model, name, systematic=True, gan=False):
    # Solver
    if gan:
        solver = AaeGanSolver(model=model)
    else:
        solver = AaeSolver(model=model)
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

    if systematic:
        a = np.zeros([50, 50])
    else:
        a = np.random.uniform(-1, 1, [50, 50])

    for v in np.linspace(-3, 3, samples):
        for i in range(model.z_dim):
            b = np.reshape(np.copy(a[i, :]), [1, 50])
            b[0][i] = v
            z.append(b)

    z = np.concatenate(z)
    y = np.zeros([model.z_dim * samples, 1])

    img = sess.run(solver.x_from_z, feed_dict={solver.z_provided: z, solver.y_labels: y})

    files= []
    for img_index in range(samples):
        blank_image = Image.new('RGB', (32*10, 32*5), color=1)
        for x in range(10):
            for y in range(5):
                index = img_index*50 + x + y*10
                im = CelebA.transform2display(img[index])
                cimg = Image.fromarray(np.uint8(im * 255))
                blank_image.paste(cimg, (x*32, y*32))

        file = "output/celeb/Res_%d.png" % img_index
        files.append(file)
        blank_image.save(file)

        ap = APNG()
        # Create animated png
        for file in files:
            ap.append(file, delay=50)
        for file in files[::-1]:
            ap.append(file, delay=50)

        ap.save("output/celeb/Celeb_systematic2.apng" )


if __name__ == '__main__':
    scenario = 3

    celeb_z_dim = 50

    if scenario == 1:
        model = ModelSubpix32(batch_size=samples*celeb_z_dim, z_dim=celeb_z_dim, y_dim=None, is_training=False)
        draw_images(model, name='Celeb_Subpix_noy', systematic=False)
    if scenario == 2:
        model = ModelConv32(batch_size=samples * celeb_z_dim, z_dim=celeb_z_dim, y_dim=None, is_training=False)
        draw_images(model, name='Celeb_Conv_noy', systematic=True)
    if scenario == 3:
        model = ModelConv32(batch_size=samples * celeb_z_dim, z_dim=celeb_z_dim, y_dim=None, is_training=False)
        draw_images(model, name='Celeb_SubpixRes_noy', systematic=True)

    #
    # if scenario == 2:
    #     y_dim = None
    #     model = ModelDenseMnist(batch_size=samples ** 4 * frames, z_dim=5, y_dim=y_dim, is_training=False)
    #     draw_images(model, name='Mnist_Dense_noy')
    # if scenario == -2:
    #     y_dim = None
    #     model = ModelDenseMnist(batch_size=samples ** 4 * frames, z_dim=5, y_dim=y_dim, is_training=False)
    #     draw_images(model, name='Mnist_Dense_noy')
    #
    # if scenario == 3:
    #     y_dim = 10
    #     model = ModelConvMnist(batch_size=samples ** 4 * frames, z_dim=5, y_dim=y_dim, is_training=False)
    #     draw_images(model, name='Mnist_Conv_y')
    # if scenario == -3:
    #     y_dim = 10
    #     model = ModelConvMnist(batch_size=91, z_dim=5, y_dim=y_dim, is_training=False)
    #     compare_style(model, name='Mnist_Conv_y')
    #
    # if scenario == 4:
    #     y_dim = None
    #     model = ModelConvMnist(batch_size=samples ** 4 * frames, z_dim=5, y_dim=y_dim, is_training=False)
    #     draw_images(model, name='Mnist_Conv_noy')
    # if scenario == -4:
    #     y_dim = None
    #     model = ModelConvMnist(batch_size=91, z_dim=5, y_dim=y_dim, is_training=False)
    #     compare_style(model, name='Mnist_Conv_noy')
