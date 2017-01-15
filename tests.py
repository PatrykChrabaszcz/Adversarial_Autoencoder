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


def test_coordinates_org():
        x_dim = 4
        y_dim = 4
        scale = 1.0
        n_pixel = x_dim * y_dim
        batch_size = 3
        x_range = scale * (np.arange(x_dim) - (x_dim - 1) / 2.0) / (x_dim - 1) / 0.5
        y_range = scale * (np.arange(y_dim) - (y_dim - 1) / 2.0) / (y_dim - 1) / 0.5
        x_mat = np.matmul(np.ones((y_dim, 1)), x_range.reshape((1, x_dim)))
        y_mat = np.matmul(y_range.reshape((y_dim, 1)), np.ones((1, x_dim)))
        r_mat = np.sqrt(x_mat * x_mat + y_mat * y_mat)
        x_mat = np.tile(x_mat.flatten(), batch_size).reshape(batch_size, n_pixel, 1)
        y_mat = np.tile(y_mat.flatten(), batch_size).reshape(batch_size, n_pixel, 1)
        r_mat = np.tile(r_mat.flatten(), batch_size).reshape(batch_size, n_pixel, 1)
        res = np.concatenate([x_mat, y_mat], axis=1)
        print(x_mat)


def _coordinates(self, scale=1):
        px = 28
        scale_f = float(scale)
        x = np.arange(px * scale) / scale_f
        x = np.array([[i] * self.batch_size for i in x]).reshape([px * scale * self.batch_size, 1])
        x = np.tile(x, [px * scale, 1])
        y = np.array([[i / scale_f] * px * scale * self.batch_size for i in range(px * scale)]). \
                reshape([px * px * scale * scale * self.batch_size, 1])
        return np.concatenate([x, y], axis=1)


if __name__ == '__main__':
        print("MINE:")
        test_coordinates()

        print("ORG:")
        test_coordinates_org()
