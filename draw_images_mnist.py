import numpy as np

from src.model_dense_mnist import ModelDenseMnist
from src.model_conv_mnist import ModelConvMnist
from src.aae_solver import AaeSolver
from src.aae_gan_solver import AaeGanSolver

import tensorflow as tf
from PIL import Image
from apng import APNG

samples = 3
frames = 40


# Dirty script to generate images
# TODO: Clean this up and make it generate everything in one go

def draw_images(model, name, gan=False):
    # Solver
    if gan:
        print("Gan Solver")
        solver = AaeGanSolver(model=model)
    else:
        solver = AaeSolver(model=model)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    # To restore previous
    print("Restoring model")
    saver.restore(sess, 'models/model_%s.ckpt' % name)
    print("Model restored")

    z = []

    # Generate grid of z
    r = 1.75
    for z_4 in np.linspace(-2, 2, frames):
        for z_3 in np.linspace(-r, r, samples):
            for z_2 in np.linspace(-r, r, samples):
                for z_1 in np.linspace(-r, r, samples):
                    for z_0 in np.linspace(-r, r, samples):
                        z.append([z_0, z_1, z_2, z_3, z_4])

    z = np.array(z)
    z.reshape([-1, model.z_dim])

    apng = True
    for code_v in range(10):
        code = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        code[0][code_v] = 1

        y_lab = np.reshape(code * (samples**4*frames), [(samples**4*frames), 10])
        img = sess.run(solver.x_from_z, feed_dict={solver.z_provided: z, solver.y_labels: y_lab})
        files = []

        if not apng:
            blank_image = \
                Image.new('L', (28 * samples**2 + (samples-1)*10, (28 * samples**2 + (samples-1)*10)*frames), color=1)

        for z4 in range(frames):
            if apng:
                blank_image = \
                    Image.new('L', (28 * samples**2 + (samples-1)*10, 28 * samples**2 + (samples-1)*10), color=1)

            for z3 in range(samples):
                for z2 in range(samples):
                    for z1 in range(samples):
                        for z0 in range(samples):
                            index = samples**4 * z4 + samples**3 * z3 + samples**2 * z2 + samples * z1 + z0
                            im = img[index].reshape([28, 28])
                            cimg = Image.fromarray(np.uint8(im*255))

                            x = z0 * 28 + z2*(10+28*samples)
                            y = z1 * 28 + z3*(10+28*samples)
                            if apng:
                                blank_image.paste(cimg, (x, y))
                            else:
                                blank_image.paste(cimg, (x, y + z4 * (28 * samples**2 + (samples-1)*10)))
            if apng:
                file = "output/z_tmp/Res_%d.png" % z4
                files.append(file)
                blank_image.save(file)

        if apng:
            ap = APNG()
            # Create animated png
            for file in files:
                ap.append(file, delay=50)
            for file in files[::-1]:
                ap.append(file, delay=50)
            ap.save("output/z4/AA2_%d.apng" % code_v)

        else:
            blank_image.save("output/z4/WGan2_Mnist_Gan_y_%d.png" % code_v)


def compare_style(model, name):
    # Solver
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

    ap = APNG()
    files = []
    for i in range(91):
        file = "output/style/Res_%d.png" % i
        b_i[i].save(file)
        files.append(file)
    # Create animated png
    for file in files:
        ap.append(file, delay=100)

    ap.save("output/style/Mnist_Conv_style3.apng")


if __name__ == '__main__':
    scenario = 3

    if scenario == 1:
        print("Draw mnist dense y")
        y_dim = 10
        model = ModelDenseMnist(batch_size=samples ** 4 * frames, z_dim=5, y_dim=y_dim, is_training=False)
        draw_images(model, name='Mnist_Dense_y_new')
    if scenario == -1:
        y_dim = 10
        model = ModelDenseMnist(batch_size=91, z_dim=5, y_dim=y_dim, is_training=False)
        compare_style(model, name='Mnist_Dense_y')

    if scenario == 2:
        y_dim = 10
        model = ModelConvMnist(batch_size=samples ** 4 * frames, z_dim=5, y_dim=y_dim, is_training=False)
        draw_images(model, name='Mnist_Conv_y')
    if scenario == -2:
        y_dim = None
        model = ModelConvMnist(batch_size=samples ** 4 * frames, z_dim=5, y_dim=y_dim, is_training=False)
        draw_images(model, name='Mnist_Dense_noy')

    if scenario == 3:
        y_dim = 10
        model = ModelConvMnist(batch_size=samples ** 4 * frames, z_dim=5, y_dim=y_dim, is_training=False)
        draw_images(model, name='Mnist_Conv_y_pretrainGan', gan=True)
    if scenario == -3:
        y_dim = 10
        model = ModelDenseMnist(batch_size=91, z_dim=5, y_dim=y_dim, is_training=False)
        compare_style(model, name='Mnist_Dense_y', gan=True)

