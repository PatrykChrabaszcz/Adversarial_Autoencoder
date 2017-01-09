import numpy as np
from src.datasets import MNIST
from src.model_mnist_hq import ModelHqMnist
from src.model_mnist_dense import ModelDenseMnist
from src.model_mnist_conv import ModelConvMnist
from src.solver import Solver
import tensorflow as tf
import os
print(os.getcwd())
model = ModelDenseMnist(batch_size=100, z_dim=5, y_dim=10)
data = MNIST()
# Solver
solver = Solver(model=model)
# Session
config = tf.ConfigProto(
        #device_count = {'GPU': 0}
    )
sess = tf.Session(config=config)
sess.run(tf.initialize_all_variables())
# Saver
saver = tf.train.Saver()

# To restore previous
print("Reading model")
saver.restore(sess, os.path.join(os.getcwd(), 'models/model_Mnist_Dense_Momentum.ckpt'))

data = MNIST()
x_img = MNIST().train_images[:100]
y_lab = MNIST().train_labels[:100]


print("Start to generate")
y_lab[0, 5] = 0
y_lab[0, 0] = 1
x_rec, z = sess.run([solver.x_reconstructed, solver.z_encoded],
                    feed_dict={solver.x_image: x_img, solver.y_labels: y_lab})

np.save("x_rec", x_rec)


#print("High def generation ...")
#xhq_rec = sess.run(solver.xhq_from_z, feed_dict={solver.z_provided: z})
#np.save("xhq_rec", xhq_rec)
