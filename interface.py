from PyQt5.QtWidgets import (QWidget, QPushButton, QFileDialog, QCheckBox,
                             QVBoxLayout, QLabel, QHBoxLayout, QSlider, QComboBox, QApplication)
from PyQt5.QtGui import QPixmap, QColor
from PyQt5.QtCore import Qt, QTimer
from PIL import ImageQt
import sys
from scipy.misc import imread, imsave, toimage
import tensorflow as tf
from src.solver import Solver
from src.datasets import MNIST, CelebA
from src.model_mnist_dense import ModelDenseMnist
from src.model_mnist_conv import ModelConvMnist
from src.model_mnist_hq import ModelHqMnist
from src.model_celeb_conv import ModelConvCeleb
import numpy as np
from functools import partial


class Window(QWidget):
    def __init__(self):
        super().__init__()

        self.batch_size = 1

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
        self.l_oimg.setPixmap(QPixmap())

        # Choose model to work with
        self.cb_model = QComboBox(self)
        self.cb_model.addItems(['models/model_Mnist_Dense_Adam.ckpt',
                                'models/model_Mnist_Dense_Adam_noy.ckpt',
                                'models/model_Mnist_Conv_Adam.ckpt',
                                'models/model_Mnist_Conv_Adam_noy.ckpt',
                                'models/model_Mnist_Hq.ckpt',
                                'Celeb_Conv_with_y',
                                'Celeb_conv_without_y'])
        l_left.addWidget(self.cb_model)

        b_model = QPushButton('LoadModel', self)
        b_model.clicked.connect(self.load_model)
        l_left.addWidget(b_model)

        self.l_model = QLabel(self)
        self.l_model.setText('Model not loaded')
        l_left.addWidget(self.l_model)

        # Manage input images
        # Two input images
        self.iimg = [None, None]

        # Latent representation of input images, computed after loading was done
        self.img_z = [None, None]

        # Label of input image if exists
        self.img_y = [None, None]

        self.curr_index = 0

        # Load input image or sample from the model
        b_iimgl = [QPushButton('Load first image', self), QPushButton('Load second image', self)]
        b_iimgs = [QPushButton('Sample first image', self), QPushButton('Sample second image', self)]

        # Display images
        self.l_iimg = [QLabel('', self), QLabel('', self)]

        # Functions for loading and sampling image
        self.imgl_slots = [partial(self.load_image, 0), partial(self.load_image, 1)]
        self.imgs_slots = [partial(self.sample_image, 0), partial(self.sample_image, 1)]

        # Set up layout and connect functions to buttons
        for i in range(2):
            l_left.addWidget(b_iimgl[i])
            l_left.addWidget(b_iimgs[i])
            l_left.addWidget(self.l_iimg[i])
            b_iimgl[i].clicked.connect(self.imgl_slots[i])
            b_iimgs[i].clicked.connect(self.imgs_slots[i])
        l_left.insertStretch(-1)

        # Set window geometry
        self.resize(800, 600)
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
        self.image_size = None
        self.image_channels = None
        self._t = QTimer()
        self._t.setInterval(10)
        self._t.timeout.connect(self.animationStep)

    def load_model(self, clicked):
        self.l_model.setText('Loading data ...')

        s_m = str(self.cb_model.currentText())
        if 'Mnist' in s_m:
            self.data = MNIST(mean=False)
            self.z_dim = 5
            if 'Dense' in s_m:
                ModelClass = ModelDenseMnist
            elif 'Conv' in s_m:
                ModelClass = ModelConvMnist
            elif 'Hq' in s_m:
                ModelClass = ModelHqMnist
            if 'noy' in s_m:
                self.y_dim = None
            else:
                self.y_dim = 10
            self.image_size = 28
            self.image_channels = 1
        elif 'Celeb' in s_m:
            self.data = CelebA()
            ModelClass = ModelConvCeleb
            self.z_dim = 25
            if 'noy' in s_m:
                self.y_dim = None
            else:
                self.y_dim = 40
            self.image_size = 32
            self.image_channels = 3
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
        l.addWidget(b_run_d)

        b_run_a = QPushButton('Run animation')
        b_run_a.clicked.connect(self.run_animation)
        l.addWidget(b_run_a)

        pix = QPixmap(self.image_size*5, self.image_size*5)
        pix.fill(QColor(0, 0, 0))
        self.l_oimg.setPixmap(pix)
        for i in range(2):
            self.l_iimg[i].setPixmap(pix)

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
        if self.y_dim is not None:
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
        if self.img_z[self.curr_index] is not None:
            for i, z in enumerate(self.img_z[self.curr_index][0]):
                s += 'z%d: %0.3f \t' % (i, z)
                if not (i + 1) % 3:
                    s += '\n'
        self.curr_zl.setText(s)

    def get_sliders(self, v):
        for i, z_s in enumerate(self.z_sliders):
            self.img_z[self.curr_index][0][i] = z_s.value()/1000.

        self.z_string()
        self.run_decoder()

    def set_sliders(self):
        z = self.img_z[self.curr_index][0]
        for i, z_s in enumerate(self.z_sliders):
            z_s.blockSignals(True)
            z_s.setValue(float(z[i])*1000)
            z_s.blockSignals(False)

    def get_y(self, b):
        for i, c_y in enumerate(self.y_checks):
            if c_y.isChecked():
                self.img_y[self.curr_index][0][i] = 1
            else:
                self.img_y[self.curr_index][0][i] = 0
        self.run_decoder()

    def set_y(self):
        y = self.img_y[self.curr_index][0]
        for i, c_y in enumerate(self.y_checks):
            c_y.blockSignals(True)
            c_y.setChecked(bool(y[i]))
            c_y.blockSignals(False)

    def load_image(self, index,  clicked):
        f_name = QFileDialog.getOpenFileName(self, 'Open Image',
                                             '', 'Image files (*.png)')[0]
        if f_name is '':
            return

        # Get Image
        img = imread(f_name)/np.float32(256)
        self.iimg[index] = img
        self.img_z[index] = self.sess.run(self.solver.z_encoded, feed_dict={self.solver.x_image: img})
        self.img_y[index] = np.array([1] + [0] * 9).reshape([1, 10])
        if 'Mnist' in str(self.cb_model.currentText()):
            img = np.reshape(img, [28, 28])
        px = QPixmap.fromImage(ImageQt.ImageQt(toimage(img)))
        self.l_iimg[index].setPixmap(px.scaled(self.image_size*5, self.image_size*5))

        self.curr_index = index
        self.set_sliders()
        self.set_y()

    def sample_image(self, index, clicked):
        z = self.sess.run(self.solver.z_sampled)
        y = np.random.randint(0, 10)
        y = np.array([0 if i != y else 1 for i in range(10)]).reshape([1, 10])
        img = self.sess.run(self.solver.x_from_z,
                            feed_dict={self.solver.z_provided: z, self.solver.y_labels: y})

        self.iimg[index] = img
        if 'Mnist' in str(self.cb_model.currentText()):
            img = np.reshape(img, [28, 28])

        px = QPixmap.fromImage(ImageQt.ImageQt(toimage(img)))
        self.l_iimg[index].setPixmap(px.scaled(self.image_size*5, self.image_size*5))
        self.img_z[index] = z
        self.img_y[index] = y
        self.curr_index = index
        self.set_sliders()
        self.set_y()
    # def run_encoder(self, index):
    #     self.img_z[index] = self.sess.run(self.solver.z_encoded, feed_dict={self.solver.x_image: self.iimg[index]})
    #     if index == 0:
    #         self.set_sliders()


    def run_animation(self):
        self._anim_steps = 400
        self._anim_step = 0
        dy = (self.img_y[1] - self.img_y[0])/(self._anim_steps-1)
        dz = (self.img_z[1] - self.img_z[0])/(self._anim_steps-1)
        self._anim_y = [self.img_y[0] + i * dy for i in range(self._anim_steps)]
        self._anim_z = [self.img_z[0] + i * dz for i in range(self._anim_steps)]
        self._t.start()

    def animationStep(self):
        x_rec = self.sess.run(self.solver.x_from_z, feed_dict={self.solver.z_provided: self._anim_z[self._anim_step],
                                                               self.solver.y_labels: self._anim_y[self._anim_step]})
        img = x_rec
        img = np.reshape(img, [28, 28])
        px = QPixmap.fromImage(ImageQt.ImageQt(toimage(img)))
        self.l_oimg.setPixmap(px.scaled(self.image_size * 5, self.image_size * 5))
        self._anim_step += 1
        if self._anim_step == self._anim_steps:
            self._t.stop()

    def run_decoder(self):
        z = self.img_z[self.curr_index]
        y = self.img_y[self.curr_index]
        x_rec = self.sess.run(self.solver.x_from_z, feed_dict={self.solver.z_provided: z, self.solver.y_labels: y})
        img = x_rec
        img = np.reshape(img, [28, 28])
        px = QPixmap.fromImage(ImageQt.ImageQt(toimage(img)))
        self.l_oimg.setPixmap(px.scaled(self.image_size*5, self.image_size*5))

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
