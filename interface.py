from solver import *
from datasets import MNIST, CelebA
import numpy as np

model = ModelConvMnist(batch_size=100, z_dim=4)
solver = Solver(model=model)
# Session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Saver
saver = tf.train.Saver()
saver.restore(sess, '/home/chrabasp/Workspace/MNIST_AA/models/mnist_model_20.ckpt')

dataset = MNIST()
z = []
y = []
for test_x, y_b in dataset.iterate_minibatches(batchsize=model.batch_size):
    z.append(sess.run(solver.z_encoded, feed_dict={solver.x_image: test_x}))
    y.extend(y_b)

z = np.concatenate(z)
np.save('z2', z)
np.save('y2', y)
