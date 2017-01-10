import numpy as np

from src.datasets import MNIST, CelebA
from src.model_mnist_dense import ModelDenseMnist
from src.model_mnist_conv import ModelConvMnist
from src.model_mnist_hq import ModelHqMnist
from src.model_celeb_conv import ModelConvCeleb
from src.solver import Solver
import tensorflow as tf


model = ModelConvCeleb(batch_size=128, z_dim=25, y_dim=40)
data = CelebA()
# Solver
solver = Solver(model=model)
# Session
sess = tf.Session()
sess.run(tf.global_variables_initializer())
# Saver
saver = tf.train.Saver()

# To restore previous
print("Restoring model")
saver.restore(sess, 'models/model_Celeb_Conv_Momentum.ckpt')
print("Model restored")
z = []
y = []
for batch_x, batch_y in data.iterate_minibatches(model.batch_size, shuffle=True):

    z_enc, y_enc = sess.run([solver.z_encoded, solver.y_pred_enc],
                            feed_dict={solver.x_image: batch_x, solver.y_labels: batch_y})
    z.append(z_enc)
    y.extend(y_enc)

z = np.concatenate(z, axis=0)

np.save("z.npy", z)
np.save("y.npy", y)
