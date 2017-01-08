import time

from model_conv_celeb import ModelConvCeleb
from model_conv_mnist import ModelConvMnist
from model_dense_mnist import ModelDenseMnist
from model_hq import ModelHqMnist

from src.datasets import MNIST, CelebA
from src.solver import *


def test(model, data, name, restore=False):
    # Solver
    solver = Solver(model=model)
    # Session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # Saver
    saver = tf.train.Saver()

    # Training part
    n_epochs = 2000

    # To restore previous
    if restore:
        saver.restore(sess, 'models/model_%s_disc.ckpt' % name)

    for epoch in range(n_epochs):
        start_time = time.time()

        print("Starting epoch %d" % epoch)
        loss_disc_sum = 0
        steps = 0

        for batch_x, batch_y in data.iterate_minibatches(model.batch_size, shuffle=True):

            # Discriminator update
            loss_d, _ = sess.run([solver.disc_loss, solver.disc_optimizer],
                                 feed_dict={solver.x_image: batch_x, solver.disc_lr: 0.01})
            loss_disc_sum += loss_d

            steps += 1

        print('Epoch took %d seconds.' % (time.time()-start_time))

        print("Discrimination Lost %f" % (loss_disc_sum/steps))

        if (epoch+1) % 10 == 0:
            saver.save(sess, 'models/model_%s_disc.ckpt' % name)


if __name__ == '__main__':
    scenario = 1

    if scenario == 1:
        model = ModelDenseMnist(batch_size=100, z_dim=2)
        data = MNIST()
        test(model, data, name='Mnist_Dense', restore=True)
    if scenario == 2:
        model = ModelConvMnist(batch_size=100, z_dim=2)
        data = MNIST()
        test(model, data, name='Mnist_Conv', restore=True)
    if scenario == 3:
        model = ModelConvCeleb(batch_size=100, z_dim=20)
        data = CelebA()
        test(model, data, name='Celeb_Conv', restore=True)
    if scenario == 4:
        model = ModelHqMnist(batch_size=100, z_dim=2)
        data = MNIST()
        test(model, data, name='Mnist_Hq', restore=True)
