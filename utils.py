import os
from scipy.misc import imread
import numpy as np


path = '/home/chrabasp/Download/img_align_celeba'

def crop_folder():
    size = 128
    images = []
    i = 0
    for file in os.listdir(path):
        image = imread(os.path.join(path, file))
        height = image.shape[0]
        width = image.shape[1]
        h_o = (height-size)//2
        w_o = (width-size)//2
        img = image[h_o:(h_o+size), w_o:(w_o+size), :]
        images.append(img)
        i += 1
        if not i % 1000:
            print('Processed %d images' % i, end='\r')
    images = np.array(images)
    np.save('celeb128.npy', images)


if __name__ == '__main__':
    crop_folder()