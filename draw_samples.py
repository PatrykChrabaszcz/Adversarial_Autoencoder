import numpy as np
import matplotlib.pyplot as plt

from src.datasets import MNIST, CelebA
from src.model_mnist_dense import ModelDenseMnist
from src.model_mnist_conv import ModelConvMnist
from src.model_mnist_hq import ModelHqMnist
from src.model_celeb_conv import ModelConvCeleb
from src.solver import Solver
import tensorflow as tf


def plot_samples(model, data, name):
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
    y = []
    pred_out = []

    i = 10
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
            plt.ylim([-5, 5])
            plt.xlim([-5, 5])

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
    scenario = 3
    y_dim = 10
    if scenario == 1:
        model = ModelDenseMnist(batch_size=128, z_dim=5, y_dim=y_dim, is_training=False)
        data = MNIST(mean=False)
        plot_samples(model, data, name='Mnist_Dense_Adam_noy')
    if scenario == 2:
        model = ModelConvMnist(batch_size=128, z_dim=5, y_dim=y_dim, is_training=False)
        data = MNIST(mean=False)
        plot_samples(model, data, name='Mnist_Conv_Adam')
    if scenario == 3:
        model = ModelHqMnist(batch_size=100, z_dim=5, y_dim=y_dim, is_training=False)
        data = MNIST(mean=False)
        plot_samples(model, data, name='Mnist_Hq')
    if scenario == 4:
        model = ModelConvCeleb(batch_size=128, z_dim=25, y_dim=None, is_training=False)
        data = CelebA()
        plot_samples(model, data, name='Celeb_Conv_Momentum_noy')
