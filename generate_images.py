import numpy as np
from src.datasets import MNIST
from src.model_mnist_hq import ModelHqMnist
from src.model_mnist_dense import ModelDenseMnist
from src.model_mnist_conv import ModelConvMnist
from src.solver import Solver
import tensorflow as tf

model = ModelConvMnist(batch_size=100, z_dim=4)
data = MNIST()
# Solver
solver = Solver(model=model)
# Session
sess = tf.Session()
sess.run(tf.global_variables_initializer())
# Saver
saver = tf.train.Saver()

# To restore previous
saver.restore(sess, 'models/model_Mnist_Conv.ckpt')

data = MNIST()
batch = MNIST().train_images[:100]
x_rec, z = sess.run([solver.x_reconstructed, solver.z_encoded], feed_dict={solver.x_image: batch})


np.save("x_rec", x_rec)
#np.save("xhq_rec", xhq_rec)
