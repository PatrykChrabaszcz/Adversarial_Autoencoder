import os
from scipy.misc import imread
import numpy as np
names = [
    'Brightfield_Microspores',
    'Brightfield_Pollen',
    'Brightfield_Protoplasts',
    'Fluorescence_GOWT1',
    'Fluorescence_HeLa',
    'PhaseContrast_HKPV',
    'PhaseContrast_U373'
]
numbers = [
    152,
    1162,
    629,
    291,
    1621,
    255,
    203
]

data_path = '/mhome/chrabasp/Download/VAE/VAE'
# Read images

def main():
    # Original images
    x = []

    # Reconstructed images (masked)
    x_rec = []

    for image_name in os.listdir(data_path):
        if 'Brightfield_Pollen' in image_name:
            if 'mask' not in image_name:

                image = imread(os.path.join(data_path, image_name))
                image_masked = imread(os.path.join(data_path, '%s_masked.tif' % image_name[:-4]))
                x.append(image.reshape([1, 4096]))
                x_rec.append(image_masked.reshape([1, 4096]))


    x = np.concatenate(x)
    x_rec = np.concatenate(x_rec)

    np.save('cell_x', x)
    np.save('cell_x_masked', x_rec)


if __name__ == '__main__':

    main()
