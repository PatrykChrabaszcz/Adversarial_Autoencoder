import numpy as np
import matplotlib.pyplot as plt

from src.datasets import MNIST, CelebA
from src.model_dense_mnist import ModelDenseMnist
from src.model_conv_mnist import ModelConvMnist

from src.model_conv_32 import ModelConv32
from src.aae_solver import AaeSolver
from src.aae_gan_solver import AaeGanSolver

import tensorflow as tf


# Script to preview distribution of latent Z
def plot_samples(model, data, name):
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
    z = []
    y = []
    pred_out = []

    i = 5
    for batch_x, batch_y in data.iterate_minibatches(model.batch_size, shuffle=True):
        z_enc, y_enc = sess.run([solver.z_encoded, solver.y_pred_enc],
                                feed_dict={solver.x_image: batch_x, solver.y_labels: batch_y})
        z.append(z_enc)
        pred_out.extend(y_enc)
        y.extend(batch_y)
        if i == 0:
            break
        i -= 1

    z = np.concatenate(z, axis=0)

    f, axarr = plt.subplots(model.z_dim, model.z_dim)
    for j in range(model.z_dim):
        for i in range(model.z_dim):
            ax = axarr[i, j]
            ax.set_title('Axis [%d, %d]' % (i, j))
            ax.scatter(z[:, j], z[:, i], c=pred_out)
            if i != model.z_dim-1:
                plt.setp(ax.get_xticklabels(), visible=False)
            if j != 0:
                plt.setp(ax.get_yticklabels(), visible=False)

    plt.show()

    f, axarr = plt.subplots(model.z_dim, model.z_dim)
    for i in range(model.z_dim):
        for j in range(model.z_dim):
            ax = axarr[i, j]
            ax.set_title('Axis [%d, %d]' % (i, j))
            ax.scatter(z[:, i], z[:, j], c=np.argmax(y, axis=1))
            if i != model.z_dim-1:
                plt.setp(ax.get_xticklabels(), visible=False)
            if j != 0:
                plt.setp(ax.get_yticklabels(), visible=False)
            ax.set_ylim([-3, 3])
            ax.set_xlim([-3, 3])
            ax.set_autoscalex_on(False)
            ax.set_autoscaley_on(False)
    plt.show()


if __name__ == '__main__':
    scenario = 2
    y_dim = 10
    if scenario == 1:
        y_dim = 10
        model = ModelDenseMnist(batch_size=128, z_dim=5, y_dim=y_dim, is_training=False)
        data = MNIST()
        plot_samples(model, data, name='Mnist_Dense_y')
    if scenario == 2:
        y_dim = None
        model = ModelDenseMnist(batch_size=128, z_dim=5, y_dim=y_dim, is_training=False)
        data = MNIST()
        plot_samples(model, data, name='Mnist_Dense_noy')
    if scenario == 3:
        y_dim = None
        model = ModelConvMnist(batch_size=128, z_dim=5, y_dim=y_dim, is_training=False)
        data = MNIST()
        plot_samples(model, data, name='Mnist_Conv_y')

    if scenario == 4:
        y_dim = None
        model = ModelConv32(batch_size=128, z_dim=10, y_dim=None, is_training=False)
        data = CelebA()
        plot_samples(model, data, name='Celeb_Conv_noy')

    if scenario == 6:
        y_dim = None
        model = ModelConv32(batch_size=128, z_dim=10, y_dim=None, is_training=False)
        data = CelebA()
        plot_samples(model, data, name='Celeb_Subpix_noy_bn')
