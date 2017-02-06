import numpy as np
from tensorflow.python.framework import ops

from src.model_dense_mnist import ModelDenseMnist
from src.model_conv_mnist import ModelConvMnist
from src.aae_solver import AaeSolver
from src.aae_gan_solver import AaeGanSolver
from src.aae_wgan_solver import AaeWGanSolver
from src.datasets import MNIST

import tensorflow as tf
from PIL import Image
from apng import APNG
from itertools import product
import os

# Number of samples in manifold
samples = 5

# Number of frames in animation
frames = 10


def manifold_images(model, name, gan, y=False):
    # Solver
    if gan == 'WGan':
        print("WGan Solver")
        solver = AaeWGanSolver(model=model)
        gan = 'WGan_'
    elif gan == 'Gan':
        print("Gan Solver")
        solver = AaeGanSolver(model=model)
        gan = 'Gan_'
    else:
        solver = AaeSolver(model=model)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    # To restore previous
    print("Restoring model")
    saver.restore(sess, 'models/model_%s%s.ckpt' % (gan, name))
    print("Model restored")

    r = 1.8
    # Generate grid of z
    ls = np.linspace(-r, r, samples)
    lf = np.linspace(-3, 3, frames)
    z = np.array(list(product(lf, ls, ls, ls, ls)))

    if not os.path.exists('output/z_tmp'):
        os.makedirs('output/z_tmp')

    if y:
        rng = 10
    else:
        rng = 1

    for z_i in range(5):
        z_t = np.roll(z, z_i, axis=1)
        for code_v in range(rng):
            code = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
            code[0][code_v] = 1

            size = (samples**4*frames)
            y_lab = np.reshape(code * size, [size, 10])
            img = sess.run(solver.x_from_z,
                           feed_dict={solver.z_provided: z_t,
                                      solver.y_labels: y_lab})
            files = []

            w = 28 * samples**2 + (samples-1)*10
            h = (28 * samples**2 + (samples-1)*10)*frames
            b_i_canv = Image.new('L', (w, h), color=1)
            b_i_apng = Image.new('L', (w, w), color=1)
            ls = range(samples)
            lf = range(frames)
            prod = list(product(lf, ls, ls, ls, ls))

            for i, p in enumerate(prod):
                index = samples ** 4 * p[0] + samples ** 3 * p[1] + \
                        samples ** 2 * p[2] + samples * p[3] + p[4]
                im = img[index].reshape([28, 28])
                cimg = Image.fromarray(np.uint8(im * 255))
                x = p[4] * 28 + p[2] * (10 + 28 * samples)
                y = p[3] * 28 + p[1] * (10 + 28 * samples)
                b_i_apng.paste(cimg, (x, y))
                y_p = y + p[0] * (28 * samples ** 2 + (samples - 1) * 10)
                b_i_canv.paste(cimg, (x, y_p))

                if (i+1) % samples**4 == 0:
                    file = "output/z_tmp/Res_%d.png" % (i // samples**4)
                    files.append(file)
                    b_i_apng.save(file)

            ap = APNG()
            # Create animated png
            for file in files:
                ap.append(file, delay=50)
            for file in files[::-1]:
                ap.append(file, delay=50)
            if not os.path.exists('output/z_%d' % z_i):
                os.makedirs('output/z_%d' % z_i)

            ap.save("output/z_%d/%s_%d.apng" % (z_i, name, code_v))

            b_i_canv.save("output/z_%d/%s_%d.png" % (z_i, name, code_v))


def compare_style(model, name, gan, y=False):
    # Solver
    if gan == 'WGan':
        print("WGan Solver")
        solver = AaeWGanSolver(model=model)
        gan = 'WGan_'
    elif gan == 'Gan':
        print("Gan Solver")
        solver = AaeGanSolver(model=model)
        gan = 'Gan_'
    else:
        solver = AaeSolver(model=model)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    # To restore previous
    print("Restoring model")
    saver.restore(sess, 'models/model_%s%s.ckpt' % (gan, name))
    print("Model restored")

    b_i = []
    for i in range(91):
        b_i.append(Image.new('L', (28 * 5, 28 * 5)))

    ys = []

    for i in range(9):
        y = [0] * 10
        y[i] = 1.0
        ys.append(np.array(y).reshape([1, 10]))
        for j in range(1, 10):
            y = [0] * 10
            y[i] = 1 - np.sin(np.pi/20*j)
            y[i+1] = np.sin(np.pi/20*j)
            ys.append(np.array(y).reshape([1, 10]))

    y = [0] * 10
    y[9] = 1.0
    ys.append(np.array(y).reshape([1, 10]))
    y = np.concatenate(ys, axis=0)

    for i in range(25):
        z = np.random.uniform(-1.5, 1.5, size=[1, 5])
        z = np.tile(z, [91, 1])
        img = sess.run(solver.x_from_z, feed_dict={solver.z_provided: z, solver.y_labels: y})

        for j in range(91):
            im = img[j].reshape([28, 28])
            cimg = Image.fromarray(np.uint8(im * 255))
            x = 28 * (i%5)
            x2 = 28 * (i//5)
            b_i[j].paste(cimg, (x, x2))

    if not os.path.exists('output/z_tmp'):
        os.makedirs('output/z_tmp')
    if not os.path.exists('output/style'):
        os.makedirs('output/style')

    ap = APNG()
    files = []
    for i in range(91):
        file = "output/z_tmp/Res_%d.png" % i
        b_i[i].save(file)
        files.append(file)
    # Create animated png
    for file in files:
        ap.append(file, delay=100)

    ap.save("output/style/%s.apng" % name)


def draw_reconstruction(model, name, gan):
    # Solver
    if gan == 'WGan':
        print("WGan Solver")
        solver = AaeWGanSolver(model=model)
        gan = 'WGan_'
    elif gan == 'Gan':
        print("Gan Solver")
        solver = AaeGanSolver(model=model)
        gan = 'Gan_'
    else:
        solver = AaeSolver(model=model)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    # To restore previous
    print("Restoring model")

    saver.restore(sess, 'models/model_%s%s.ckpt' % (gan, name))
    print("Model restored")

    data = MNIST()
    x = data.test_images[:20, :]
    y = data.test_labels[:20, :]

    x_rec = sess.run(solver.x_reconstructed, feed_dict={solver.x_image: x,
                                                        solver.y_labels: y})

    b_i = Image.new('L', (28 * 20, 28 * 2))

    for i in range(20):
        im = x[i].reshape([28, 28])
        img_o = Image.fromarray(np.uint8(im * 255))
        im = x_rec[i].reshape([28, 28])
        img_r = Image.fromarray(np.uint8(im * 255))

        b_i.paste(img_o, (i*28, 0))
        b_i.paste(img_r, (i * 28, 28))

    if not os.path.exists('output/rec'):
        os.makedirs('output/rec')

    b_i.save('output/rec/%s_rec.png' % name)


# Could be improved by using one model with smaller batch size, right now each function
# requires different batch size setting so there are 3 models
# For higher 'samples' and 'frames' value it needs a lot of GPU memory.
# TODO: Fix all functions so that they can use model with smaller batch size
if __name__ == '__main__':
    scenario = 3

    if scenario == 1:
        print("Draw Mnist Conv Gan y")
        y_dim = 10
        name = 'Mnist_Conv_y'
        model = ModelConvMnist(batch_size=samples ** 4 * frames, z_dim=5, y_dim=y_dim, is_training=False)
        manifold_images(model, name, 'Gan', True)
        ops.reset_default_graph()
        model = ModelConvMnist(batch_size=91, z_dim=5, y_dim=y_dim, is_training=False)
        compare_style(model, name, 'Gan', True)
        ops.reset_default_graph()
        model = ModelConvMnist(batch_size=20, z_dim=5, y_dim=y_dim, is_training=False)
        draw_reconstruction(model, name, gan='Gan')

    if scenario == 2:
        print("Draw Mnist Dense y")
        y_dim = 10
        name = 'Mnist_Dense_y'
        model = ModelDenseMnist(batch_size=samples ** 4 * frames, z_dim=5, y_dim=y_dim, is_training=False)
        manifold_images(model, name, '', True)
        ops.reset_default_graph()
        model = ModelDenseMnist(batch_size=91, z_dim=5, y_dim=y_dim, is_training=False)
        compare_style(model, name, '', True)
        ops.reset_default_graph()
        model = ModelDenseMnist(batch_size=20, z_dim=5, y_dim=y_dim, is_training=False)
        draw_reconstruction(model, name, gan='')

    if scenario == 3:
        print("Draw Mnist Dense no y")
        y_dim = None
        name = 'Mnist_Dense_noy'
        model = ModelDenseMnist(batch_size=samples ** 4 * frames, z_dim=5, y_dim=y_dim, is_training=False)
        manifold_images(model, name, '', False)
        ops.reset_default_graph()
        model = ModelDenseMnist(batch_size=91, z_dim=5, y_dim=y_dim, is_training=False)
        compare_style(model, name, '', False)
        ops.reset_default_graph()
        model = ModelDenseMnist(batch_size=20, z_dim=5, y_dim=y_dim, is_training=False)
        draw_reconstruction(model, name, gan='')
