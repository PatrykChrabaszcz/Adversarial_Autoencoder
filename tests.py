import numpy as np

from itertools import product

samples = 2
frames = 3
# Generate grid of z
r = 1.75
ls = np.linspace(-r, r, samples)
lf = np.linspace(-2, 2, frames)
z = np.array(list(product(lf, ls, ls, ls, ls)))

print(z)

z=[]
for z_4 in np.linspace(-2, 2, frames):
    for z_3 in np.linspace(-r, r, samples):
        for z_2 in np.linspace(-r, r, samples):
            for z_1 in np.linspace(-r, r, samples):
                for z_0 in np.linspace(-r, r, samples):
                    z.append([z_0, z_1, z_2, z_3, z_4])

z = np.array(z)
z.reshape([-1, 5])

print(z)