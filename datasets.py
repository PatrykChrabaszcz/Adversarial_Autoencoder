import numpy as np
import os
import sys
'''
Image arrays have the shape (N, 3, 32, 32), where N is the size of the
corresponding set. This is the format used by Lasagne/Theano. To visualize the
images, you need to change the axis order, which can be done by calling
np.rollaxis(image_array[n, :, :, :], 0, start=3).

Each image has an associated 40-dimensional attribute vector. The names of the
attributes are stored in self.attr_names.
'''

celeb_path = "CELEB"
mnist_path = "MNIST"


class CelebA:
    def __init__(self):
        self.train_images = np.float32(np.load(os.path.join(celeb_path, "train_images_32.npy"))) / 255.0
        self.train_labels = np.uint8(np.load(os.path.join(celeb_path, "train_labels_32.npy")))
        self.val_images = np.float32(np.load(os.path.join(celeb_path, "val_images_32.npy"))) / 255.0

        self.val_labels = np.uint8(np.load(os.path.join(celeb_path, "val_labels_32.npy")))
        self.train_images = np.rollaxis(self.train_images, 1, 4)
        self.val_images = np.rollaxis(self.train_images, 1, 4)
        with open(os.path.join(celeb_path, "attr_names.txt")) as f:
            self.attr_names = f.readlines()[0].split()

    def iterate_minibatches(self, batchsize, shuffle=False, test=False):
        if test:
            inputs = self.test_images
            targets = self.test_labels
        else:
            inputs = self.train_images
            targets = self.train_labels
        assert len(inputs) == len(targets)
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt]

class MNIST:
    def __init__(self):
        data = self.load_dataset()
        self.train_images = data['x_train']
        self.train_labels = data['y_train']
        self.test_images = data['x_test']
        self.test_labels = data['y_test']

    def iterate_minibatches(self, batchsize, shuffle=False, test=False):
        if test:
            inputs = self.test_images
            targets = self.test_labels
        else:
            inputs = self.train_images
            targets = self.train_labels
        assert len(inputs) == len(targets)
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt]

    def load_dataset(self):
        # We first define a download function, supporting both Python 2 and 3.
        if sys.version_info[0] == 2:
            from urllib import urlretrieve
        else:
            from urllib.request import urlretrieve

        def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
            print("Downloading %s" % filename)
            urlretrieve(source + filename, os.path.join(mnist_path, filename))

        # We then define functions for loading MNIST images and labels.
        # For convenience, they also download the requested files if needed.
        import gzip

        def load_mnist_images(filename):
            if not os.path.exists(filename):
                download(filename)
            # Read the inputs in Yann LeCun's binary format.
            with gzip.open(os.path.join(mnist_path, filename), 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8, offset=16)
            # The inputs are vectors now, we reshape them to monochrome 2D images,
            # following the shape convention: (examples, channels, rows, columns)
            data = data.reshape(-1, 1, 28, 28)
            # The inputs come as bytes, we convert them to float32 in range [0,1].
            # (Actually to range [0, 255/256], for compatibility to the version
            # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
            return data / np.float32(256)

        def load_mnist_labels(filename):
            if not os.path.exists(filename):
                download(filename)
            # Read the labels in Yann LeCun's binary format.
            with gzip.open(os.path.join(mnist_path, filename), 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8, offset=8)
            # The labels are vectors of integers now, that's exactly what we want.
            return data

        # We can now download and read the training and test set images and labels.
        x_train = load_mnist_images('train-images-idx3-ubyte.gz')
        y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
        x_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
        y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

        # We just return all the arrays in order, as expected in main().
        # (It doesn't matter how we do this as long as we can read them again.)

        x_train = np.reshape(x_train, [-1, 784])
        x_test = np.reshape(x_test, [-1, 784])
        return {'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test}