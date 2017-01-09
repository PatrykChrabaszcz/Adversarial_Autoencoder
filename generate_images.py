import numpy as np
from src.datasets import MNIST
from src.model_mnist_hq import ModelHqMnist
from src.model_mnist_dense import ModelDenseMnist
from src.model_mnist_conv import ModelConvMnist
from src.solver import Solver
import tensorflow as tf

model = ModelConvMnist(batch_size=100, z_dim=5, y_dim=10)
data = MNIST()
# Solver
solver = Solver(model=model)
# Session
sess = tf.Session()
sess.run(tf.global_variables_initializer())
# Saver
saver = tf.train.Saver()

# To restore previous
print("Reading model")
saver.restore(sess, 'models/model_Mnist_Conv.ckpt')

data = MNIST()
x_img = MNIST().train_images[:100]
y_lab = MNIST().train_labels[:100]


print("Start to generate")
x_rec, z = sess.run([solver.x_reconstructed, solver.z_encoded],
                    feed_dict={solver.x_image: x_img, solver.y_labels: y_lab})
np.save("x_rec", x_rec)


#print("High def generation ...")
#xhq_rec = sess.run(solver.xhq_from_z, feed_dict={solver.z_provided: z})
#np.save("xhq_rec", xhq_rec)
