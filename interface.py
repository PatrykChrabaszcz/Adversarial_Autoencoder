from PyQt5.QtWidgets import (QWidget, QPushButton, QFileDialog, QCheckBox,
                             QVBoxLayout, QLabel, QHBoxLayout, QSlider, QComboBox, QApplication)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import sys
from scipy.misc import imread, imsave
import tensorflow as tf
from src.solver import Solver
from src.datasets import MNIST, CelebA
from src.model_mnist_dense import ModelDenseMnist
from src.model_mnist_conv import ModelConvMnist
from src.model_celeb_conv import ModelConvCeleb
import numpy as np

class Window(QWidget):
    def __init__(self):
        super().__init__()

        self.batch_size = 100

        l_main = QHBoxLayout()


        # Left sidebar
        l_left = QVBoxLayout()
        l_main.addLayout(l_left)

        # Middle layout
        self.l_mid = QVBoxLayout()
        l_main.addLayout(self.l_mid)

        # Right sidebar
        self.l_right = QVBoxLayout()
        l_main.addLayout(self.l_right)

        self.l_oimg = QLabel()
        self.l_right.addWidget(self.l_oimg)

        # Choose model to work with
        self.cb_model = QComboBox(self)
        self.cb_model.addItems(['models/model_Mnist_Dense_Momentum.ckpt',
                                'models/model_Mnist_Dense_Momentum_noy.ckpt',
                                'models/model_Mnist_Conv_Momentum.ckpt',
                                'models/model_Mnist_Conv_Momentum_noy.ckpt',
                                'Celeb_Conv_with_y',
                                'Celeb_conv_without_y'])
        l_left.addWidget(self.cb_model)

        b_model = QPushButton('LoadModel', self)
        b_model.clicked.connect(self.load_model)
        l_left.addWidget(b_model)

        self.l_model = QLabel(self)
        self.l_model.setText('Load model')
        l_left.addWidget(self.l_model)

        # Choose input image
        self.iimg = None
        b_iimg = QPushButton('Load image', self)
        b_iimg.clicked.connect(self.load_image)
        l_left.addWidget(b_iimg)

        self.l_iimg = QLabel('', self)
        l_left.addWidget(self.l_iimg)

        l_left.insertStretch(-1)

        # Set geometry
        self.setGeometry(300, 300, 300, 200)
        l_main.insertStretch(-1)
        self.setLayout(l_main)

        self.data = None
        self.solver = None
        config = tf.ConfigProto(
                device_count={'GPU': 0}
            )
        self.sess = tf.Session(config=config)

        self.z_dim = None
        self.y_dim = None
        self.z_sliders = []
        self.curr_z = []
        self.curr_zl = QLabel('')

        self.y_checks = []
        self.curr_y = []

    def load_model(self, clicked):
        self.l_model.setText('Loading data ...')

        s_m = str(self.cb_model.currentText())
        if 'Mnist' in s_m:
            self.data = MNIST(mean=False)
            self.z_dim = 5
            if 'Dense' in s_m:
                ModelClass = ModelDenseMnist
                self.batch_size = 1
            else:
                ModelClass = ModelConvMnist
            if 'noy' in s_m:
                self.y_dim = None
            else:
                self.y_dim = 10
        elif 'Celeb' in s_m:
            self.data = CelebA()
            ModelClass = ModelConvCeleb
            self.z_dim = 25
            if 'noy' in s_m:
                self.y_dim = None
            else:
                self.y_dim = 40
        else:
            self.l_model.setText('Could not load model')
            return

        model = ModelClass(batch_size=self.batch_size, z_dim=self.z_dim, y_dim=self.y_dim, is_training=False)

        self.solver = Solver(model=model)

        self.sess.run(tf.global_variables_initializer())
        # Saver
        saver = tf.train.Saver()
        # To restore previous
        self.l_model.setText('Loading model ...')
        saver.restore(self.sess, s_m)
        self.l_model.setText('Loaded.')

        self.z_y_init()
        self.control_init()

    def control_init(self):
        l = QHBoxLayout()
        self.l_mid.addLayout(l)

        b_run_d = QPushButton('Run decoder')
        b_run_d.clicked.connect(self.run_decoder)
        b_run_e = QPushButton('Run encoder')
        b_run_e.clicked.connect(self.run_encoder)
        l.addWidget(b_run_d)
        l.addWidget(b_run_e)

    def z_y_init(self):
        # Add dynamic changes
        l = QVBoxLayout()
        self.l_mid.addLayout(l)
        for i in range(self.z_dim):
            h_l = QHBoxLayout()
            l_z = QLabel('Z: %d' % i)
            s_z = QSlider(Qt.Horizontal)
            s_z.setMinimum(-3000)
            s_z.setMaximum(3000)
            s_z.valueChanged.connect(self.get_sliders)
            self.curr_z.append(0)
            self.z_sliders.append(s_z)
            h_l.addWidget(l_z)
            h_l.addWidget(s_z)
            l.addLayout(h_l)

        l.addWidget(self.curr_zl)
        for i in range(self.y_dim):
            if not i % 5:
                h_l = QHBoxLayout()
                l.addLayout(h_l)
            c_y = QCheckBox()
            c_y.setText('y %d' % i)
            c_y.stateChanged.connect(self.get_y)
            h_l.addWidget(c_y)
            self.curr_y.append(0)
            self.y_checks.append(c_y)
        l.insertStretch(-1)
        self.z_string()

    def z_string(self):
        s = ''
        for i, z in enumerate(self.curr_z):
            s += 'z%d: %0.3f \t' % (i, z)
            if not (i + 1) % 3:
                s += '\n'
            self.curr_zl.setText(s)

    def get_sliders(self, v):
        for i, z_s in enumerate(self.z_sliders):
            self.curr_z[i] = z_s.value()/1000.

        self.z_string()
        self.run_decoder()

    def set_sliders(self):
        pass

    def get_y(self, b):
        for i, c_y in enumerate(self.y_checks):
            if c_y.isChecked():
                self.curr_y[i] = 1
            else:
                self.curr_y[i] = 0
        self.run_decoder()

    def set_y(self):
        pass


    def load_image(self, clicked):
        f_name = QFileDialog.getOpenFileName(self, 'Open Image',
                                             '', 'Image files (*.png)')[0]
        self.iimg = imread(f_name)
        self.l_iimg.setPixmap(QPixmap(f_name).scaled(128, 128))

    def run_encoder(self):
        x = np.tile(self.iimg, [self.batch_size, 1])
        #y =
        #z =

    def run_decoder(self):
        z = np.tile(self.curr_z, [self.batch_size, 1])
        y = np.tile(self.curr_y, [self.batch_size, 1])

        x_rec = self.sess.run(self.solver.x_from_z, feed_dict={self.solver.z_provided: z, self.solver.y_labels: y})
        img = x_rec[1]
        img = np.reshape(img, [28, 28])
        imsave('dec_output.png', img)
        self.l_oimg.setPixmap(QPixmap('dec_output.png').scaled(128, 128))

    def draw_sample(self):
        self.iimg = self.data.train_images[0]
        self.curr_y = self.data.train_labels[0]

        imsave('sample.png', self.i_img)
        self.l_iimg.setPixmap(QPixmap('sample.png'))


def main():
    app = QApplication(sys.argv)
    w = Window()
    w.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
