import numpy as np
import os
import sys
celeb_path = 'CELEB'
mnist_path = 'MNIST'


class CelebA:
    # By default compute mean (Output layer from the network uses tanh activation)
    def __init__(self, mean=True):
        self.train_images = np.float32(np.load(os.path.join(celeb_path, 'train_images_32.npy'))) / 255.0
        self.train_labels = np.uint8(np.load(os.path.join(celeb_path, 'train_labels_32.npy')))
        self.val_images = np.float32(np.load(os.path.join(celeb_path, 'val_images_32.npy'))) / 255.0
        self.val_labels = np.uint8(np.load(os.path.join(celeb_path, 'val_labels_32.npy')))
        self.train_images = np.rollaxis(self.train_images, 1, 4)
        self.val_images = np.rollaxis(self.train_images, 1, 4)
        self.mean_image = False

        if mean:
            self.mean_image = np.mean(self.train_images, axis=0)
            self.train_images = self.train_images-self.mean_image
            self.test_images = self.train_images-self.mean_image

        with open(os.path.join(celeb_path, 'attr_names.txt')) as f:
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
            yield (
             inputs[excerpt], targets[excerpt])


class MNIST:
    # By default do not compute mean (Output layer from the network uses sigmoid activation)
    def __init__(self, mean=False):
        data = self.load_dataset()
        self.train_images = data['x_train']
        self.test_images = data['x_test']
        self.mean_image = None

        if mean:
            self.mean_image = np.mean(data['x_train'], axis=0)
            self.train_images = data['x_train']-self.mean_image
            self.test_images = data['x_test']-self.mean_image

        y_tr = data['y_train']
        y_te = data['y_test']
        # One hot encoding
        y = np.zeros((y_tr.size, 10))
        y[np.arange(y_tr.size), y_tr] = 1
        self.train_labels = y
        y = np.zeros((y_te.size, 10))
        y[np.arange(y_te.size), y_te] = 1
        self.test_labels = y

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
            yield (inputs[excerpt], targets[excerpt])

    @staticmethod
    def load_dataset():
        if sys.version_info[0] == 2:
            from urllib import urlretrieve
        else:
            from urllib.request import urlretrieve

        def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
            print('Downloading %s' % filename)
            urlretrieve(source + filename, os.path.join(mnist_path, filename))

        import gzip

        def load_mnist_images(filename):
            if not os.path.exists(os.path.join(mnist_path, filename)):
                download(filename)
            with gzip.open(os.path.join(mnist_path, filename), 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8, offset=16)
            data = data.reshape(-1, 1, 28, 28)
            return data / np.float32(256)

        def load_mnist_labels(filename):
            if not os.path.exists(os.path.join(mnist_path, filename)):
                download(filename)
            with gzip.open(os.path.join(mnist_path, filename), 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8, offset=8)
            return data

        x_train = load_mnist_images('train-images-idx3-ubyte.gz')
        y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
        x_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
        y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')
        x_train = np.reshape(x_train, [-1, 784])
        x_test = np.reshape(x_test, [-1, 784])
        return {'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test}
