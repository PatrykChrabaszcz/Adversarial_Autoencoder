import time

from src.model_mnist_conv import ModelConvMnist
from src.model_mnist_dense import ModelDenseMnist
from src.model_mnist_hq import ModelHqMnist
#from model_celeb_conv import ModelConvCeleb

from src.datasets import MNIST
from src.solver import *



def train(model, data, name, restore=False):
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
        saver.restore(sess, 'models/model_%s.ckpt' % name)

    for epoch in range(n_epochs):
        start_time = time.time()

        print("Starting epoch %d" % epoch)
        loss_rec_sum = 0
        loss_disc_sum = 0
        loss_enc_sum = 0
        steps = 0

        for batch_x, batch_y in data.iterate_minibatches(model.batch_size, shuffle=True):
            # Reconstruction update
            l_r, _ = sess.run([solver.rec_loss, solver.rec_optimizer],
                              feed_dict={solver.x_image: batch_x, solver.rec_lr: 0.01})
            loss_rec_sum += l_r
            # Discriminator update
            l_d, _ = sess.run([solver.disc_loss, solver.disc_optimizer],
                              feed_dict={solver.x_image: batch_x, solver.disc_lr: 0.1})
            loss_disc_sum += l_d

            # Encoder update
            enc_loss, _ = sess.run([solver.enc_loss, solver.enc_optimizer],
                                   feed_dict={solver.x_image: batch_x, solver.enc_lr: 0.1})
            loss_enc_sum += enc_loss

            steps += 1

        print('Epoch took %d seconds.' % (time.time()-start_time))

        print("Reconstruction Lost %f" % (loss_rec_sum/steps))
        print("Discrimination Lost %f" % (loss_disc_sum/steps))
        print("Encoder Lost %f \n" % (loss_enc_sum/steps))

        if (epoch+1) % 10 == 0:
            pass
        saver.save(sess, 'models/model_%s.ckpt' % name)


if __name__ == '__main__':
    scenario = 1

    if scenario == 1:
        model = ModelDenseMnist(batch_size=100, z_dim=4)
        data = MNIST()
        train(model, data, name='Mnist_Dense')
    if scenario == 2:
        model = ModelConvMnist(batch_size=100, z_dim=4)
        data = MNIST()
        train(model, data, name='Mnist_Conv', restore=True)
    if scenario == 3:
        model = ModelHqMnist(batch_size=100, z_dim=25)
        data = MNIST()
        train(model, data, name='Mnist_Hq')
    # if scenario == 4:
    #     model = ModelConvCeleb(batch_size=100, z_dim=20)
    #     data = CelebA()
    #     train(model, data, name='Celeb_Conv')
