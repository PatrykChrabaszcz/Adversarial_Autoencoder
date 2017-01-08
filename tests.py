# Test coordinates function from model_hq_mnist

import numpy as np

def test_coordinates():

        batch_size = 3
        z = np.array([[1, 5], [5, 5], [3, 2]])
        z = np.tile(z, [16,1])
        z = np.reshape(z, [4*4*batch_size, 2])
        px = 4
        scale = 1
        scale_f = float(scale)
        x = np.arange(px * scale) / scale_f
        x = np.array([[i] * batch_size for i in x]).reshape([px * scale * batch_size, 1])
        x = np.tile(x, [px, 1])
        y = np.array([[i / scale_f] * px * batch_size for i in range(px * scale)]). \
            reshape([px * px * scale * batch_size, 1])

        res = np.concatenate([x, y], axis=1)
        res = np.concatenate([res, z], axis=1)
        print(res)


if __name__ == '__main__':
    test_coordinates()
