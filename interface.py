from PyQt5.QtWidgets import (QWidget, QPushButton, QFileDialog, QCheckBox,
                             QVBoxLayout, QLabel, QHBoxLayout, QSlider, QComboBox, QApplication)
from PyQt5.QtGui import QPixmap, QColor, QImage
from PyQt5.QtCore import Qt, QTimer
from PIL import ImageQt, Image
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

        self._l_main = QHBoxLayout(self)

        # Start Menu
        self._w_start = QWidget(self)
        self._l_start = QVBoxLayout(self._w_start)
        self._l_main.addWidget(self._w_start)

        l_model = QLabel(self._w_start)
        l_model.setText('Select model to work with')
        self._l_start.addWidget(l_model)

        self.cb_model = QComboBox(self._w_start)
        self.cb_model.addItems(['models/model_Mnist_Dense_Adam.ckpt',
                                'models/model_Mnist_Dense_Adam_noy.ckpt',
                                'models/model_Mnist_Conv_y.ckpt',
                                'models/model_Mnist_Conv_Adam_n.ckpt',
                                'models/model_Mnist_Conv_Adam_noy_n.ckpt',
                                'models/model_Mnist_Conv_Adam.ckpt',
                                'models/model_Mnist_Conv_Adam_new.ckpt',
                                'models/model_Mnist_Conv_Adam_noy.ckpt',
                                'models/model_Celeb_Conv_Adam_sigmoid_50_noy.ckpt',
                                'models/model_Celeb_Conv_Adam_tanh_50_noy.ckpt',
                                'models/model_Celeb_Conv_Adam_tanh_50.ckpt',
                                'models/model_Mnist_Hq.ckpt'
                                ])
        self._l_start.addWidget(self.cb_model)

        b_model = QPushButton('Load Model', self._w_start)
        b_model.clicked.connect(self.load_model)
        self._l_start.addWidget(b_model)

        # Input Images
        self.iimg = [None, None]
        # Latent representation of input images, computed after loading was done
        self.img_z = [None, None]
        # Label of input image if exists
        self.img_y = [None, None]

        self.z_dim = None
        self.y_dim = None
        self.image_channels = None
        self.image_size = None

        # Widgets to display images
        self._l_oimg = [QLabel('', self), QLabel('', self)]
        self._l_iimg = [QLabel('', self), QLabel('', self)]
        self._l_anim = QLabel('', self)

        self.z_sliders = []
        self.y_checks = []
        self.curr_index = 0

        self.data = None
        self.solver = None

        config = tf.ConfigProto(
                device_count={'GPU': 0}
            )
        self.sess = tf.Session(config=config)

        self._t = QTimer()
        self._t.setInterval(10)
        self._t.timeout.connect(self.animationStep)

    def load_model(self, clicked):
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
                self.z_dim = 10
            self.y_dim = 10
            self.image_size = 28
            self.image_channels = 1
        elif 'Celeb' in s_m:
            self.data = CelebA(mean=True)
            ModelClass = ModelConvCeleb
            self.z_dim = 50
            self.y_dim = 40
            self.image_size = 32
            self.image_channels = 3

        if 'noy' in s_m:
            self.y_dim = None

        model = ModelClass(batch_size=self.batch_size, z_dim=self.z_dim, y_dim=self.y_dim, is_training=False)

        self.solver = Solver(model=model)

        self.sess.run(tf.global_variables_initializer())
        # Saver
        saver = tf.train.Saver()
        # Restore previous
        saver.restore(self.sess, s_m)

        self.build_interface()

    def build_interface(self):
        # Hide starting widget
        self._w_start.hide()

        # Reserve a place to display input and output images
        pix = QPixmap(self.image_size * 5, self.image_size * 5)
        pix.fill(QColor(0, 0, 0))

        # Left sidebar
        l_left = QVBoxLayout()
        self._l_start.deleteLater()
        self._l_main.addLayout(l_left)
        b_iimgl = [QPushButton('Load first image', self), QPushButton('Load second image', self)]
        b_iimgrz = [QPushButton('Sample random first z', self), QPushButton('Sample random second z', self)]
        b_iimgri = [QPushButton('Sample random first image', self), QPushButton('Sample random second image', self)]
        imgl_slots = [partial(self.load_image, 0), partial(self.load_image, 1)]
        imgrz_slots = [partial(self.sample_random_z, 0), partial(self.sample_random_z, 1)]
        imgri_slots = [partial(self.sample_random_image, 0), partial(self.sample_random_image, 1)]

        for i in range(2):
            self._l_iimg[i].setPixmap(pix)
            self._l_oimg[i].setPixmap(pix)
            l_left.addWidget(b_iimgl[i])
            l_left.addWidget(b_iimgrz[i])
            l_left.addWidget(b_iimgri[i])
            l = QHBoxLayout()
            l.addWidget(self._l_iimg[i])
            l.addWidget(self._l_oimg[i])
            b_iimgl[i].clicked.connect(imgl_slots[i])
            b_iimgrz[i].clicked.connect(imgrz_slots[i])
            b_iimgri[i].clicked.connect(imgri_slots[i])
            l_left.addLayout(l)

        # Middle layout
        l_mid = QVBoxLayout()
        self._l_main.addLayout(l_mid)

        l = QHBoxLayout()
        l_mid.addLayout(l)

        b_run_d = QPushButton('Run decoder')
        b_run_d.clicked.connect(self.run_decoder)
        l.addWidget(b_run_d)

        b_run_a = QPushButton('Run animation')
        b_run_a.clicked.connect(self.run_animation)
        l.addWidget(b_run_a)
        l_mid.insertStretch(-1)

        # Build z sliders
        l = QHBoxLayout()
        l_mid.addLayout(l)

        for i in range(self.z_dim):
            if not i % 25:
                lv = QVBoxLayout()
                l.addLayout(lv)
                l.insertStretch(-1)
            h_l = QHBoxLayout()
            lv.addLayout(h_l)
            l_z1 = QLabel('Z: %d,  -3' % i)
            l_z2 = QLabel('Z: %d,  3' % i)
            s_z = QSlider(Qt.Horizontal)
            s_z.setMinimum(-3000)
            s_z.setMaximum(3000)
            s_z.valueChanged.connect(self.get_sliders)
            self.z_sliders.append(s_z)
            h_l.addWidget(l_z1)
            h_l.addWidget(s_z)
            h_l.addWidget(l_z2)

        # Build y checkboxes
        if self.y_dim is not None:
            for i in range(self.y_dim):
                if not i % 20:
                    lv = QVBoxLayout()
                    l.addLayout(lv)
                    l.insertStretch(-1)
                c_y = QCheckBox()
                c_y.setText('y %d' % i)
                c_y.stateChanged.connect(self.get_y)
                lv.addWidget(c_y)
                self.y_checks.append(c_y)
        l.insertStretch(-1)

        # Right sidebar
        l_right = QVBoxLayout()
        self._l_main.addLayout(l_right)

        self._l_anim.setPixmap(pix)
        l_right.addWidget(QLabel('Animation', self))
        l_right.addWidget(self._l_anim)
        l_right.insertStretch(-1)


    def get_sliders(self, v):
        for i, z_s in enumerate(self.z_sliders):
            self.img_z[self.curr_index][0][i] = z_s.value()/1000.

        self.run_decoder(self.curr_index)

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
        self.run_decoder(self.curr_index)

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
        if self.data.name == 'Celeb':
            img = img[:, :, :3]
            img = np.reshape(img, [1, 32, 32, 3])
            if self.data.mean_image is not None:
                img -= self.data.mean_image

        self.iimg[index] = img
        self.img_z[index] = self.sess.run(self.solver.z_encoded, feed_dict={self.solver.x_image: img})
        self.img_y[index] = np.array([0] * 40).reshape([1, 40])
        if 'Mnist' in str(self.cb_model.currentText()):
            img = np.reshape(img, [28, 28])
        px = self.toQImage(img)
        self._l_iimg[index].setPixmap(px)

        self.curr_index = index
        self.set_sliders()
        self.set_y()
        self.run_decoder(index)

    def sample_random_z(self, index, clicked):
        z = self.sess.run(self.solver.z_sampled)
        if self.data.name == 'Mnist':
            y = np.random.randint(0, 10)
            y = np.array([0 if i != y else 1 for i in range(10)]).reshape([1, 10])
        elif self.data.name == 'Celeb':
            y = np.random.randint(0, 1, [1, 40])

        img = self.sess.run(self.solver.x_from_z,
                            feed_dict={self.solver.z_provided: z, self.solver.y_labels: y})
        self.iimg[index] = img
        px = self.toQImage(img)
        self._l_iimg[index].setPixmap(px)

        self.img_z[index] = z
        self.img_y[index] = y
        self.curr_index = index
        self.set_sliders()
        self.set_y()
        self.run_decoder(index)

    def run_animation(self):
        self._anim_steps = 400
        self._anim_step = 0
        dy = (self.img_y[1] - self.img_y[0])/(self._anim_steps-1)
        dz = (self.img_z[1] - self.img_z[0])/(self._anim_steps-1)
        self._anim_y = [self.img_y[0] + i * dy for i in range(self._anim_steps)]
        self._anim_z = [self.img_z[0] + i * dz for i in range(self._anim_steps)]
        self._t.start()

    def animationStep(self):
        img = self.sess.run(self.solver.x_from_z, feed_dict={self.solver.z_provided: self._anim_z[self._anim_step],
                                                             self.solver.y_labels: self._anim_y[self._anim_step]})
        px = self.toQImage(img)
        self._l_anim.setPixmap(px)
        self._anim_step += 1
        if self._anim_step == self._anim_steps:
            self._t.stop()

    def run_decoder(self, index):
        z = self.img_z[index]
        y = self.img_y[index]
        img = self.sess.run(self.solver.x_from_z, feed_dict={self.solver.z_provided: z, self.solver.y_labels: y})

        px = self.toQImage(img)
        self._l_oimg[index].setPixmap(px)

    def sample_random_image(self, index, clicked):
        i = np.random.randint(0, self.data.test_images.shape[0])
        img = self.data.test_images[i]
        img = np.reshape(img, [1, 784])
        y = self.data.test_labels[i]
        y = np.reshape(y, [1, 10])
        self.iimg[index] = img
        self.img_z[index] = self.sess.run(self.solver.z_encoded,
                                          feed_dict={self.solver.x_image: img, self.solver.y_labels: y})
        self.img_y[index] = y
        if 'Mnist' in str(self.cb_model.currentText()):
            img = np.reshape(img, [28, 28])
        px = self.toQImage(img)
        self._l_iimg[index].setPixmap(px)

        self.curr_index = index
        self.set_sliders()
        self.set_y()
        self.run_decoder(index)


    def toQImage(self, image):
        if self.data.name == 'Mnist':
            img = np.reshape(image, [self.image_size, self.image_size])
            mode = 'L'
        elif self.data.name == 'Celeb':
            image += self.data.mean_image
            img = np.reshape(image, [self.image_size, self.image_size, self.image_channels])
            mode = 'RGB'
        pilimage = Image.fromarray(np.uint8(img*255), mode)
        imageq = ImageQt.ImageQt(pilimage)
        qimage = QImage(imageq)
        pix = QPixmap(qimage)
        pix = pix.scaled(5*self.image_size, 5*self.image_size)

        return pix


def main():
    app = QApplication(sys.argv)
    w = Window()
    w.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
